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
from proteus.utils.s3_utils import upload_bytes_to_s3
from proteus.data.data_constants import DataPath, IndexCol, ChainKey, ProteinKey, NpzKey

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
				IndexCol.PDB: pdb_id,
				IndexCol.CHAIN: chain_id,
				IndexCol.SOURCE: meta[ProteinKey.SOURCE],
				IndexCol.SHARD_ID: shard_name,
				IndexCol.OFFSET: data_offset,
				IndexCol.SIZE: len(blob),
				IndexCol.RESOLUTION: meta[ProteinKey.RESOLUTION],
				IndexCol.METHOD: meta[ProteinKey.METHOD],
				IndexCol.DEPOSIT_DATE: meta[ProteinKey.DEPOSIT_DATE],
				IndexCol.MEAN_PLDDT: meta[ProteinKey.MEAN_PLDDT],
				IndexCol.PTM: meta[ProteinKey.PTM],
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
	chain_ids = list(data[ProteinKey.CHAINS].keys())
	for chain_id, chain_data in data[ProteinKey.CHAINS].items():
		arrays[f"{chain_id}/{ChainKey.COORDS}"] = chain_data[ChainKey.COORDS]
		arrays[f"{chain_id}/{ChainKey.ATOM_MASK}"] = chain_data[ChainKey.ATOM_MASK]
		arrays[f"{chain_id}/{ChainKey.BFACTOR}"] = chain_data[ChainKey.BFACTOR]
		arrays[f"{chain_id}/{ChainKey.PLDDT}"] = chain_data[ChainKey.PLDDT]
		arrays[f"{chain_id}/{ChainKey.OCCUPANCY}"] = chain_data[ChainKey.OCCUPANCY]
		arrays[f"{chain_id}/{ChainKey.SEQUENCE}"] = np.array(chain_data[ChainKey.SEQUENCE])

	# chain-to-chain similarity arrays
	if ProteinKey.CHAIN_TM_SCORES in data:
		arrays[ProteinKey.CHAIN_TM_SCORES] = data[ProteinKey.CHAIN_TM_SCORES]
	if ProteinKey.CHAIN_SEQ_IDENTITY in data:
		arrays[ProteinKey.CHAIN_SEQ_IDENTITY] = data[ProteinKey.CHAIN_SEQ_IDENTITY]

	meta = {
		ProteinKey.RESOLUTION: data[ProteinKey.RESOLUTION],
		ProteinKey.METHOD: data[ProteinKey.METHOD],
		ProteinKey.DEPOSIT_DATE: data[ProteinKey.DEPOSIT_DATE],
		ProteinKey.SOURCE: data[ProteinKey.SOURCE],
		ProteinKey.MEAN_PLDDT: data[ProteinKey.MEAN_PLDDT],
		ProteinKey.PTM: data[ProteinKey.PTM],
		ProteinKey.CHAINS: chain_ids,
		ProteinKey.ASSEMBLIES: [
			{ProteinKey.CHAINS: a[ProteinKey.CHAINS], ProteinKey.ASMB_XFORMS: a[ProteinKey.ASMB_XFORMS].tolist()}
			for a in data[ProteinKey.ASSEMBLIES]
		],
	}
	arrays[NpzKey.META] = np.void(json.dumps(meta).encode())

	buf = io.BytesIO()
	np.savez(buf, **arrays)
	compressor = zstd.ZstdCompressor(level=zstd_level)
	return compressor.compress(buf.getvalue())


def _deserialize_pdb_blob(blob: bytes) -> Dict:
	"""inverse of _serialize_pdb_blob. returns the same dict structure."""
	decompressor = zstd.ZstdDecompressor()
	raw = decompressor.decompress(blob)
	npz = np.load(io.BytesIO(raw), allow_pickle=False)

	meta = json.loads(bytes(npz[NpzKey.META]))
	chains = {}
	for chain_id in meta[ProteinKey.CHAINS]:
		chains[chain_id] = {
			ChainKey.COORDS: npz[f"{chain_id}/{ChainKey.COORDS}"],
			ChainKey.ATOM_MASK: npz[f"{chain_id}/{ChainKey.ATOM_MASK}"],
			ChainKey.BFACTOR: npz[f"{chain_id}/{ChainKey.BFACTOR}"],
			ChainKey.PLDDT: npz[f"{chain_id}/{ChainKey.PLDDT}"],
			ChainKey.OCCUPANCY: npz[f"{chain_id}/{ChainKey.OCCUPANCY}"],
			ChainKey.SEQUENCE: str(npz[f"{chain_id}/{ChainKey.SEQUENCE}"]),
		}

	assemblies = [
		{ProteinKey.CHAINS: a[ProteinKey.CHAINS], ProteinKey.ASMB_XFORMS: np.array(a[ProteinKey.ASMB_XFORMS], dtype=np.float32)}
		for a in meta[ProteinKey.ASSEMBLIES]
	]

	result = {
		ProteinKey.CHAINS: chains,
		ProteinKey.ASSEMBLIES: assemblies,
		ProteinKey.RESOLUTION: meta[ProteinKey.RESOLUTION],
		ProteinKey.METHOD: meta[ProteinKey.METHOD],
		ProteinKey.DEPOSIT_DATE: meta[ProteinKey.DEPOSIT_DATE],
		ProteinKey.SOURCE: meta[ProteinKey.SOURCE],
		ProteinKey.MEAN_PLDDT: meta[ProteinKey.MEAN_PLDDT],
		ProteinKey.PTM: meta[ProteinKey.PTM],
	}

	if ProteinKey.CHAIN_TM_SCORES in npz:
		result[ProteinKey.CHAIN_TM_SCORES] = npz[ProteinKey.CHAIN_TM_SCORES]
	if ProteinKey.CHAIN_SEQ_IDENTITY in npz:
		result[ProteinKey.CHAIN_SEQ_IDENTITY] = npz[ProteinKey.CHAIN_SEQ_IDENTITY]

	return result
