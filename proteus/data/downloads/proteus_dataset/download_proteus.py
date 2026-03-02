'''
entry point for the experimental structure data collection and cleaning pipeline
'''

import gzip
import io
import json
import logging
import tarfile
import aiohttp
import aioboto3
import asyncio
import requests
import hydra
from pathlib import Path
from cloudpathlib import S3Path
from tqdm import tqdm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
import gemmi
import shutil
import subprocess

from proteus.types import Dict, List
from proteus.static.constants import resname_2_one, noncanonical_parent, atoms as atom14_order
from proteus.utils.s3_utils import upload_bytes_to_s3, REGION, get_session
from proteus.data.downloads.proteus_dataset.conf.download import (
	DataPipelineCfg,
	ExperimentalDataDownloadCfg,
	FoldSeekCfg,
	register_download_configs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
register_download_configs()

class FoldSeek:
	def __init__(self, cfg: FoldSeekCfg):
		self.input_path = Path(cfg.input_path)
		self.db_path = Path(cfg.db_path)
		self.raw_db_path = self.db_path / "raw_db"
		self.cluster_db_path = self.db_path / "cluster_db"
		self.cluster_tsv_path = self.db_path / "clusters.tsv"
		self.tmp_dir = self.db_path / "tmp"

		# shared
		self.verbosity = str(cfg.verbosity)

		# createdb
		self.distance_threshold = str(cfg.distance_threshold)
		self.mask_bfactor_threshold = str(cfg.mask_bfactor_threshold)
		self.coord_store_mode = str(cfg.coord_store_mode)
		self.chain_name_mode = str(cfg.chain_name_mode)

		# clustering
		self.tmscore_threshold = str(cfg.tmscore_threshold)
		self.tmscore_threshold_mode = str(cfg.tmscore_threshold_mode)
		self.lddt_threshold = str(cfg.lddt_threshold)
		self.coverage = str(cfg.coverage)
		self.cov_mode = str(cfg.cov_mode)
		self.cluster_mode = str(cfg.cluster_mode)
		self.sensitivity = str(cfg.sensitivity)
		self.e_value = str(cfg.e_value)
		self.max_seqs = str(cfg.max_seqs)
		self.min_aln_len = str(cfg.min_aln_len)
		self.max_seq_len = str(cfg.max_seq_len)
		self.split = str(cfg.split)
		self.split_memory_limit = cfg.split_memory_limit
		self.cluster_steps = str(cfg.cluster_steps)
		self.cluster_reassign = str(int(cfg.cluster_reassign))

	def create_db(self):
		self.db_path.mkdir(parents=True, exist_ok=True)
		cmd = [
			"foldseek", "createdb",
			str(self.input_path),
			str(self.raw_db_path),
			"--db-extraction-mode", "0",              # chain extraction
			"--input-format", "2",                    # mmCIF
			"--distance-threshold", self.distance_threshold,
			"--mask-bfactor-threshold", self.mask_bfactor_threshold,
			"--coord-store-mode", self.coord_store_mode,
			"--chain-name-mode", self.chain_name_mode,
			"-v", self.verbosity,
		]

		db_output = subprocess.run(cmd)

		# delete the raw cifs after we create the db
		db_output.check_returncode()
		shutil.rmtree(self.input_path)

	def run_cluster(self):
		"""run foldseek cluster, clean up tmp dir. does not parse results."""
		raw_db = str(self.raw_db_path)
		cluster_db = str(self.cluster_db_path)
		tmp_dir = str(self.tmp_dir)

		cmd = [
			"foldseek", "cluster",
			raw_db, cluster_db, tmp_dir,
			"--alignment-type", "0",                    # 3Di only (structure, no sequence)
			"--min-seq-id", "0.0",                      # no sequence identity filter
			"--tmscore-threshold", self.tmscore_threshold,
			"--tmscore-threshold-mode", self.tmscore_threshold_mode,
			"--lddt-threshold", self.lddt_threshold,
			"-c", self.coverage,
			"--cov-mode", self.cov_mode,
			"--cluster-mode", self.cluster_mode,
			"-e", self.e_value,
			"-s", self.sensitivity,
			"--max-seqs", self.max_seqs,
			"--min-aln-len", self.min_aln_len,
			"--max-seq-len", self.max_seq_len,
			"--split", self.split,
			"--split-memory-limit", self.split_memory_limit,
			"--cluster-steps", self.cluster_steps,
			"--cluster-reassign", self.cluster_reassign,
			"--remove-tmp-files", "1",
			"-v", self.verbosity,
		]

		result = subprocess.run(cmd)
		result.check_returncode()
		if self.tmp_dir.exists():
			shutil.rmtree(self.tmp_dir)

	def parse_clusters(self) -> Dict[str, str]:
		"""run createtsv, clean up db files, return {chain_id: cluster_representative}.
		leaves the TSV on disk so it can serve as a resume signal."""

		raw_db, cluster_db, tsv_path = str(self.raw_db_path), str(self.cluster_db_path), str(self.cluster_tsv_path)

		result = subprocess.run([
			"foldseek", "createtsv",
			raw_db, raw_db, cluster_db, tsv_path,
			"-v", self.verbosity,
		])
		result.check_returncode()

		# clean up db files now that we have the tsv
		for f in self.db_path.glob(self.raw_db_path.name + "*"):
			f.unlink()
		for f in self.db_path.glob(self.cluster_db_path.name + "*"):
			f.unlink()

		return self.load_clusters()

	def load_clusters(self) -> Dict[str, str]:
		"""read existing clusters.tsv into {chain_id: cluster_representative}"""
		clusters = {}
		for line in self.cluster_tsv_path.read_text().splitlines():
			rep, member = line.split("\t")
			clusters[member] = rep
		return clusters

	def cleanup_tsv(self):
		"""remove the clusters TSV after index has been safely uploaded"""
		if self.cluster_tsv_path.exists():
			self.cluster_tsv_path.unlink()

class DataPipeline:
	def __init__(self, cfg: DataPipelineCfg):
		self.experimental_dl = ExperimentalDataDownload(cfg.experimental_dl)
		self.foldseek = FoldSeek(cfg.foldseek)
		self.foldseek_input_path = Path(cfg.foldseek.input_path)
		self.foldseek_db_path = Path(cfg.foldseek.db_path)
		self.s3_path = S3Path(cfg.s3_path)
		self.local_path = Path(cfg.local_path)

	def download(self):
		return self.experimental_dl.download()

	def run(self):
		index_rows = self.download()
		index_path = self.local_path / "index.parquet"

		# build columnar dict from list of row dicts
		columns = {}
		for key in index_rows[0]:
			columns[key] = [row[key] for row in index_rows]

		table = pa.table(columns)
		pq.write_table(table, index_path)
		logger.info(f"saved index with {len(index_rows)} rows to {index_path}")

		# clustering: resume from whichever stage completed last
		if self.foldseek.cluster_tsv_path.exists():
			logger.info("found existing clusters.tsv, skipping foldseek")
			clusters = self.foldseek.load_clusters()
		else:
			has_cluster_db = any(self.foldseek.db_path.glob(self.foldseek.cluster_db_path.name + "*"))
			if has_cluster_db:
				logger.info("found existing cluster db, skipping createdb and cluster")
			else:
				has_raw_db = any(self.foldseek.db_path.glob(self.foldseek.raw_db_path.name + "*"))
				if not has_raw_db:
					self.foldseek.create_db()
				self.foldseek.run_cluster()
			clusters = self.foldseek.parse_clusters()

		# add cluster_id column
		cluster_ids = [
			clusters.get(f"{pdb}_{chain}", "")
			for pdb, chain in zip(columns["pdb"], columns["chain"])
		]
		table = table.append_column("cluster_id", pa.array(cluster_ids))
		pq.write_table(table, index_path)

		# upload index to s3
		s3_index_path = self.s3_path / "shards" / "index.parquet"
		upload_bytes_to_s3_sync(index_path.read_bytes(), s3_index_path)
		logger.info(f"uploaded index to {s3_index_path}")

		# cleanup only after S3 upload succeeds
		self.foldseek.cleanup_tsv()
		if self.experimental_dl.checkpoint_path.exists():
			self.experimental_dl.checkpoint_path.unlink()
			logger.info("removed checkpoint file after successful completion")


def upload_bytes_to_s3_sync(data: bytes, s3_path: S3Path):
	"""synchronous upload of bytes to S3"""
	session = get_session(aio=False)
	client = session.client("s3", region_name=REGION)
	client.put_object(Bucket=s3_path.bucket, Key=s3_path.key, Body=data)


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

	def add(self, pid: str, blob: bytes, chain_ids: list[str], meta: dict):
		"""append a blob to the current shard, recording index rows per chain"""
		# record byte offset before writing
		header_offset = self._buf.tell()
		data_offset = header_offset + 512  # tar header is 512 bytes

		# add blob to tar
		info = tarfile.TarInfo(name=f"{pid}.npz.zst")
		info.size = len(blob)
		self._tar.addfile(info, io.BytesIO(blob))
		self._entry_count += 1

		# verify tar layout: data should start right after a 512-byte header
		expected_end = data_offset + len(blob) + (-len(blob) % 512)
		assert self._buf.tell() == expected_end, \
			f"unexpected tar layout for {pid}: expected {expected_end}, got {self._buf.tell()}"

		# record one index row per chain
		shard_name = f"{self._source}/{self._shard_id:06d}"
		for chain_id in chain_ids:
			self._index_rows.append({
				"pdb": pid,
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

		shard_key = f"shards/{self._source}/{self._shard_id:06d}.tar"
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


class ExperimentalDataDownload:
	def __init__(self, cfg: ExperimentalDataDownloadCfg):
		assert cfg.min_chain_length >= 4, \
			f"min_chain_length must be >= 4 for foldseek clustering, got {cfg.min_chain_length}"
		self.methods = set(cfg.methods)
		self.max_resolution = cfg.max_resolution
		self.max_entries = cfg.max_entries
		self.min_chain_length = cfg.min_chain_length
		self.semaphore_limit = cfg.semaphore_limit
		self.chunk_size = cfg.chunk_size
		self.shard_size_bytes = cfg.shard_size_mb * 1024 * 1024
		self.s3_path = S3Path(cfg.s3_path)
		self.local_path = Path(cfg.local_path)
		self.checkpoint_path = Path(cfg.checkpoint_path)

	def download(self):
		experimental_ids = self._get_experimental_ids()
		return asyncio.run(self._download_async(experimental_ids))

	def _load_checkpoint(self) -> tuple[list[dict], set[str], int]:
		"""read checkpoint from previous run. returns (index_rows, done_pdb_ids, next_shard_id)"""
		checkpoint_path = self.checkpoint_path
		if not checkpoint_path.exists():
			return [], set(), 0
		index_rows = []
		done_pids = set()
		max_shard = -1
		for line in checkpoint_path.read_text().splitlines():
			row = json.loads(line)
			index_rows.append(row)
			done_pids.add(row["pdb"])
			shard_num = int(row["shard_id"].rsplit("/", 1)[1])
			max_shard = max(max_shard, shard_num)
		next_shard_id = max_shard + 1 if max_shard >= 0 else 0
		logger.info(f"resume: {len(done_pids)} PDBs already done across {next_shard_id} shards")
		return index_rows, done_pids, next_shard_id

	async def _download_async(self, experimental_ids: List[str]):
		self.local_path.mkdir(parents=True, exist_ok=True)

		# load checkpoint from previous run (if any)
		index_rows, done_pids, next_shard_id = self._load_checkpoint()
		remaining = [pid for pid in experimental_ids if pid.lower() not in done_pids]
		logger.info(f"{len(remaining)} to download ({len(done_pids)} already done)")

		pbar = tqdm(total=len(remaining), desc="downloading")

		connector = aiohttp.TCPConnector(limit=self.semaphore_limit, enable_cleanup_closed=True)
		timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=60)
		s3_session = get_session(aio=True)
		succeeded, failed = 0, 0
		async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session, \
			s3_session.client("s3", region_name=REGION) as s3_client:

			shard_writer = ShardWriter(
				s3_prefix=self.s3_path,
				shard_size_bytes=self.shard_size_bytes,
				source="experimental",
				s3_client=s3_client,
				checkpoint_path=self.checkpoint_path,
				resume_index_rows=index_rows,
				resume_shard_id=next_shard_id,
			)
			shard_lock = asyncio.Lock()

			async def _task(pdb_id):
				result = await self._maybe_pdbredo_else_rcsb(
					pdb_id, session, shard_writer, shard_lock,
				)
				pbar.update(1)
				return result

			for i in range(0, len(remaining), self.chunk_size):
				chunk = remaining[i:i + self.chunk_size]
				tasks = [_task(pdb_id) for pdb_id in chunk]
				results = await asyncio.gather(*tasks, return_exceptions=True)
				for pdb_id, result in zip(chunk, results):
					if isinstance(result, Exception):
						logger.error(f"{pdb_id}: {result}")
						failed += 1
					elif result is None:
						failed += 1
					else:
						succeeded += 1

			index_rows = await shard_writer.finalize()
		pbar.close()

		logger.info(f"done: {succeeded} succeeded, {failed} skipped out of {len(remaining)}")
		return index_rows

	async def _maybe_pdbredo_else_rcsb(
		self, pdb_id: str, session: aiohttp.ClientSession,
		shard_writer: "ShardWriter", shard_lock: asyncio.Lock,
	):
		data = await self._download_pdbredo(pdb_id, session)
		if data is None:
			data = await self._download_rcsb(pdb_id, session)
		if data is None:
			return None

		pid = pdb_id.lower()

		# write ca cifs locally for foldseek
		for chain_id, chain_data in data["chains"].items():
			cif_path = self.local_path / f"{pid}_{chain_id}.cif.gz"
			cif_path.write_bytes(gzip.compress(chain_data["cif"].encode()))

		# pack into blob and add to shard
		blob = _serialize_pdb_blob(pid, data)
		chain_ids = list(data["chains"].keys())
		meta = {
			"resolution": data["resolution"],
			"method": data["method"],
			"deposit_date": data["deposit_date"],
			"source": data["source"],
		}
		async with shard_lock:
			shard_writer.add(pid, blob, chain_ids, meta)

		return pdb_id

	async def _fetch(
		self, session: aiohttp.ClientSession, url: str,
		max_retries: int = 3, retry_statuses: frozenset[int] | None = None,
	) -> bytes | None:
		"""fetch url with retries and exponential backoff. returns None on non-retryable errors.

		retry_statuses: HTTP status codes to retry (default: 500+). pass empty set to
		never retry on status codes (still retries network/timeout errors).
		"""
		if retry_statuses is None:
			retry_statuses = frozenset(range(500, 600))
		for attempt in range(max_retries):
			try:
				async with session.get(url) as resp:
					if resp.status in retry_statuses:
						raise aiohttp.ClientResponseError(
							resp.request_info, resp.history,
							status=resp.status, message=f"server error {resp.status}",
						)
					if resp.status != 200:
						return None
					return await resp.read()
			except (aiohttp.ClientError, asyncio.TimeoutError):
				if attempt == max_retries - 1:
					raise
				delay = 2 ** attempt
				logger.warning(f"retry {attempt + 1}/{max_retries} for {url} (waiting {delay}s)")
				await asyncio.sleep(delay)

	async def _download_pdbredo(self, pdb_id: str, session: aiohttp.ClientSession):
		# pdb-redo returns 500 (not 404) for missing entries, so treat any
		# non-200 as "not found" without retrying. still retry on network errors.
		pid = pdb_id.lower()
		raw = await self._fetch(
			session,
			f"https://pdb-redo.eu/db/{pid}/{pid}_final.cif",
			retry_statuses=frozenset(),
		)
		if raw is None:
			return None
		content = raw.decode("utf-8")
		data = _parse_mmcif(content, self.methods, self.max_resolution, self.min_chain_length)
		if data is None:
			return None
		data |= {"source": "pdb-redo"}
		return data

	async def _download_rcsb(self, pdb_id: str, session: aiohttp.ClientSession):
		pid = pdb_id.lower()
		raw = await self._fetch(
			session,
			f"https://files.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/{pid[1:3]}/{pid}.cif.gz",
		)
		if raw is None:
			return None
		with gzip.open(io.BytesIO(raw), "rt") as f:
			content = f.read()
		data = _parse_mmcif(content, self.methods, self.max_resolution, self.min_chain_length)
		if data is None:
			return None
		data |= {"source": "rcsb"}
		return data

	def _get_experimental_ids(self) -> List[str]:
		# query RCSB Search API for PDB IDs matching our method and resolution criteria
		method_nodes = [
			{
				"type": "terminal",
				"service": "text",
				"parameters": {
					"attribute": "exptl.method",
					"operator": "exact_match",
					"value": method,
				},
			}
			for method in self.methods
		]

		query = {
			"query": {
				"type": "group",
				"logical_operator": "and",
				"nodes": [
					{
						"type": "group",
						"logical_operator": "or",
						"nodes": method_nodes,
					},
					{
						"type": "terminal",
						"service": "text",
						"parameters": {
							"attribute": "rcsb_entry_info.resolution_combined",
							"operator": "less_or_equal",
							"value": self.max_resolution,
						},
					},
				],
			},
			"return_type": "entry",
			"request_options": {"return_all_hits": True},
		}

		logger.info("querying RCSB search API for experimental IDs ...")
		resp = requests.post(
			"https://search.rcsb.org/rcsbsearch/v2/query",
			json=query,
			timeout=120,
		)

		if resp.status_code == 200 and resp.content:
			results = resp.json()
			pdbids = [hit["identifier"].upper() for hit in results.get("result_set", [])]
			logger.info(f"rcsb search returned {len(pdbids)} entries")
		else:
			# fallback: get all PDB IDs from holdings and let _parse_mmcif filter
			logger.warning(f"rcsb search API returned {resp.status_code}, falling back to holdings")
			pdbids = self._get_holdings_ids()
			logger.info(f"holdings returned {len(pdbids)} entries")

		if self.max_entries > 0:
			pdbids = pdbids[:self.max_entries]

		return pdbids

	def _get_holdings_ids(self) -> List[str]:
		"""get all PDB IDs from wwPDB holdings file"""
		url = "https://files.wwpdb.org/pub/pdb/holdings/current_file_holdings.json.gz"
		logger.info(f"retrieving rcsb metadata at {url} ...")
		resp = requests.get(url, timeout=120)
		resp.raise_for_status()
		with gzip.open(io.BytesIO(resp.content), "rt") as f:
			holdings = json.load(f)
		return list(holdings.keys())

def _parse_mmcif(content: str, methods: List[str], max_resolution: float, min_chain_length: int = 4) -> Dict | None:
	structure = gemmi.read_structure_string(content)

	# filter by experimental method
	method = structure.info['_exptl.method'] if '_exptl.method' in structure.info else ''
	method = method.strip("'\" ")
	if method not in methods:
		return None

	# filter by resolution
	if structure.resolution > max_resolution:
		return None

	model = structure[0]

	# per-chain data
	chains_data = {}
	for chain in model:
		polymer = chain.get_polymer()
		if not polymer:
			continue

		# collect residues, resolving microheterogeneity
		resolved = []
		for res_or_group in polymer:
			if hasattr(res_or_group, '__iter__') and not hasattr(res_or_group, 'name'):
				candidates = [r for r in res_or_group if r.name in resname_2_one]
				if candidates:
					resolved.append(_best_residue(candidates))
			elif res_or_group.name in resname_2_one:
				resolved.append(res_or_group)

		if len(resolved) < min_chain_length:
			continue

		L = len(resolved)
		coords = np.zeros((L, 14, 3), dtype=np.float32)
		mask = np.zeros((L, 14), dtype=bool)
		bfactors = np.zeros(L, dtype=np.float32)
		seq = []
		# backbone atoms for foldseek (needs at least N, CA, C to compute 3Di)
		backbone_names = ["N", "CA", "C", "O", "CB"]
		cif_lines = [
			f"data_{chain.name}",
			"loop_",
			"_atom_site.group_PDB",
			"_atom_site.id",
			"_atom_site.type_symbol",
			"_atom_site.label_atom_id",
			"_atom_site.label_alt_id",
			"_atom_site.label_comp_id",
			"_atom_site.label_asym_id",
			"_atom_site.label_entity_id",
			"_atom_site.label_seq_id",
			"_atom_site.Cartn_x",
			"_atom_site.Cartn_y",
			"_atom_site.Cartn_z",
			"_atom_site.occupancy",
			"_atom_site.B_iso_or_equiv",
			"_atom_site.auth_seq_id",
			"_atom_site.auth_asym_id",
			"_atom_site.pdbx_PDB_model_num",
		]
		cif_atom_id = 0

		for i, res in enumerate(resolved):
			resname = res.name
			seq.append(resname_2_one[resname])
			parent = noncanonical_parent.get(resname, resname)
			expected_atoms = atom14_order[parent]

			for j, atom_name in enumerate(expected_atoms):
				atom = _best_atom(res, atom_name)
				if atom is not None:
					coords[i, j] = [atom.pos.x, atom.pos.y, atom.pos.z]
					mask[i, j] = True
					if atom_name == "CA":
						bfactors[i] = atom.b_iso

					if atom_name in backbone_names:
						cif_atom_id += 1
						elem = atom_name[0]
						cif_lines.append(
							f"ATOM {cif_atom_id} {elem} {atom_name} . {resname} "
							f"{chain.name} 1 {i + 1} "
							f"{atom.pos.x:.3f} {atom.pos.y:.3f} {atom.pos.z:.3f} "
							f"1.00 {atom.b_iso:.2f} {i + 1} {chain.name} 1"
						)

		chains_data[chain.name] = {
			"sequence": "".join(seq),
			"coords": coords,
			"atom_mask": mask,
			"bfactor": bfactors,
			"cif": "\n".join(cif_lines) + "\n",
		}

	# assembly / biounit info with Nx4x4 homogeneous transforms
	assemblies = []
	for assembly in structure.assemblies:
		biounit_chains = []
		transforms = []
		for gen in assembly.generators:
			biounit_chains.extend(list(gen.subchains) or list(gen.chains))
			for oper in gen.operators:
				mat4 = np.eye(4, dtype=np.float32)
				mat4[:3, :3] = np.array(oper.transform.mat.tolist(), dtype=np.float32)
				mat4[:3, 3] = np.array(oper.transform.vec.tolist(), dtype=np.float32)
				transforms.append(mat4)
		assemblies.append({
			"chains": biounit_chains,
			"transforms": np.stack(transforms) if transforms else np.empty((0, 4, 4), dtype=np.float32),
		})

	if not chains_data:
		return None

	date_key = '_pdbx_database_status.recvd_initial_deposition_date'
	deposit_date = structure.info[date_key] if date_key in structure.info else ''

	return {
		"chains": chains_data,
		"assemblies": assemblies,
		"resolution": structure.resolution,
		"method": method,
		"deposit_date": deposit_date,
	}


ZSTD_LEVEL = 10

def _serialize_pdb_blob(pid: str, data: Dict) -> bytes:
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
	compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL)
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


def _best_atom(res, atom_name: str):
	'''find the atom with the highest occupancy'''
	best = None
	for atom in res:
		if atom.name != atom_name:
			continue
		if best is None or atom.occ > best.occ:
			best = atom
	return best

def _best_residue(residues):
	'''for microheterogeneity, pick the residue with highest average occupancy'''
	if len(residues) == 1:
		return residues[0]
	best_res, best_occ = None, -1.0
	for res in residues:
		atoms = [a for a in res]
		if not atoms:
			continue
		avg_occ = sum(a.occ for a in atoms) / len(atoms)
		if avg_occ > best_occ:
			best_occ = avg_occ
			best_res = res
	return best_res


@hydra.main(version_base=None, config_name="default")
def main(cfg: DataPipelineCfg):
	pipeline = DataPipeline(cfg)
	pipeline.run()


if __name__ == "__main__":
	main()
