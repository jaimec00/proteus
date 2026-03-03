import asyncio
import io
import json
import logging
import tarfile

import numpy as np
import zstandard as zstd
from pathlib import Path
from cloudpathlib import S3Path

from proteus.types import Dict
from proteus.data import DataPath
from proteus.utils.s3_utils import upload_bytes_to_s3

logger = logging.getLogger(__name__)


class ShardWriter:
	"""packs PDB blobs into tar shards on-the-fly and uploads them to S3"""

	def __init__(
		self, s3_prefix: S3Path, shard_size_bytes: int, source: str, s3_client,
		checkpoint_path: Path = None, resume_index_rows: list[dict] = None, resume_shard_id: int = 0,
	):
		self._s3_prefix = s3_prefix
		self._shard_size_bytes = shard_size_bytes
		self._source = source
		self._s3_client = s3_client
		self._checkpoint_path = checkpoint_path

		self._shard_id = resume_shard_id
		self._buf = io.BytesIO()
		self._tar = tarfile.open(fileobj=self._buf, mode="w")
		self._entry_count = 0
		self._pending_uploads: list[asyncio.Task] = []
		self._index_rows: list[dict] = resume_index_rows or []
		self._shard_row_start = len(self._index_rows)

	def add(self, pdb_id: str, blob: bytes, chain_ids: list[str], meta: dict):
		"""append a blob to the current shard, recording index rows per chain"""
		# record byte offset before writing
		header_offset = self._buf.tell()
		data_offset = header_offset + 512  # tar header is 512 bytes

		# add blob to tar
		info = tarfile.TarInfo(name=f"{pdb_id}.npz.zst")
		info.size = len(blob)
		self._tar.addfile(info, io.BytesIO(blob))
		self._entry_count += 1

		# verify tar layout: data should start right after a 512-byte header
		expected_end = data_offset + len(blob) + (-len(blob) % 512)
		assert self._buf.tell() == expected_end, \
			f"unexpected tar layout for {pdb_id}: expected {expected_end}, got {self._buf.tell()}"

		# record one index row per chain
		shard_name = f"{self._source}/{self._shard_id:06d}"
		for chain_id in chain_ids:
			self._index_rows.append({
				"pdb": pdb_id,
				"chain": chain_id,
				"source": meta["source"],
				"shard_id": shard_name,
				"offset": data_offset,
				"size": len(blob),
				"resolution": meta["resolution"],
				"method": meta["method"],
				"deposit_date": meta["deposit_date"],
			})

		# flush if over size target
		if self._buf.tell() >= self._shard_size_bytes:
			self._flush_shard()

	def _flush_shard(self):
		"""close current tar, start background upload + checkpoint, reset buffer"""
		self._tar.close()
		shard_bytes = self._buf.getvalue()

		shard_key = f"{DataPath.SHARDS}/{self._source}/{self._shard_id:06d}.tar"
		s3_path = self._s3_prefix / shard_key

		# snapshot the rows belonging to this shard before resetting
		shard_rows = self._index_rows[self._shard_row_start:]
		task = asyncio.create_task(
			self._upload_and_checkpoint(shard_bytes, s3_path, shard_rows)
		)
		self._pending_uploads.append(task)
		logger.info(f"flushing shard {self._shard_id:06d} ({len(shard_bytes)} bytes, {self._entry_count} entries)")

		# reset for next shard
		self._shard_id += 1
		self._shard_row_start = len(self._index_rows)
		self._buf = io.BytesIO()
		self._tar = tarfile.open(fileobj=self._buf, mode="w")
		self._entry_count = 0

	async def _upload_and_checkpoint(self, shard_bytes: bytes, s3_path, rows: list[dict]):
		"""upload shard to S3, then append its index rows to the checkpoint file"""
		await upload_bytes_to_s3(shard_bytes, s3_path, self._s3_client)
		if self._checkpoint_path:
			with open(self._checkpoint_path, "a") as f:
				for row in rows:
					f.write(json.dumps(row) + "\n")

	async def finalize(self) -> list[dict]:
		"""flush last shard if non-empty, await all uploads, return index rows"""
		if self._entry_count > 0:
			self._flush_shard()

		if self._pending_uploads:
			await asyncio.gather(*self._pending_uploads)
			logger.info(f"all {self._shard_id} shards uploaded")

		return self._index_rows


def _serialize_pdb_blob(pdb_id: str, data: Dict, zstd_level: int = 10) -> bytes:
	"""pack all per-chain arrays + pdb metadata into a single zstd-compressed npz blob.

	layout inside the npz:
	  - {chain_id}/coords, {chain_id}/atom_mask, {chain_id}/bfactor, {chain_id}/sequence
	  - _meta (json bytes)
	  - _chains (list of chain IDs, for ordering)
	"""
	arrays = {}
	chain_ids = list(data["chains"].keys())
	for chain_id, chain_data in data["chains"].items():
		arrays[f"{chain_id}/coords"] = chain_data["coords"]
		arrays[f"{chain_id}/atom_mask"] = chain_data["atom_mask"]
		arrays[f"{chain_id}/bfactor"] = chain_data["bfactor"]
		arrays[f"{chain_id}/sequence"] = np.array(chain_data["sequence"])

	meta = {
		"resolution": data["resolution"],
		"method": data["method"],
		"deposit_date": data["deposit_date"],
		"source": data["source"],
		"chains": chain_ids,
		"assemblies": [
			{"chains": a["chains"], "transforms": a["transforms"].tolist()}
			for a in data["assemblies"]
		],
	}
	arrays["_meta"] = np.void(json.dumps(meta).encode())

	buf = io.BytesIO()
	np.savez(buf, **arrays)
	compressor = zstd.ZstdCompressor(level=zstd_level)
	return compressor.compress(buf.getvalue())


def _deserialize_pdb_blob(blob: bytes) -> Dict:
	"""inverse of _serialize_pdb_blob. returns the same dict structure."""
	decompressor = zstd.ZstdDecompressor()
	raw = decompressor.decompress(blob)
	npz = np.load(io.BytesIO(raw), allow_pickle=False)

	meta = json.loads(bytes(npz["_meta"]))
	chains = {}
	for chain_id in meta["chains"]:
		chains[chain_id] = {
			"coords": npz[f"{chain_id}/coords"],
			"atom_mask": npz[f"{chain_id}/atom_mask"],
			"bfactor": npz[f"{chain_id}/bfactor"],
			"sequence": str(npz[f"{chain_id}/sequence"]),
		}

	assemblies = [
		{"chains": a["chains"], "transforms": np.array(a["transforms"], dtype=np.float32)}
		for a in meta["assemblies"]
	]

	return {
		"chains": chains,
		"assemblies": assemblies,
		"resolution": meta["resolution"],
		"method": meta["method"],
		"deposit_date": meta["deposit_date"],
		"source": meta["source"],
	}
