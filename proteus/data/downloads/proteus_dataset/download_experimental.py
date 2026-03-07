import gzip
import io
import json
import logging
import re
import subprocess
import asyncio

import aiohttp
import aioboto3
import requests
from pathlib import Path
from cloudpathlib import S3Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from proteus.types import Dict, List, Tuple, Set
from proteus.utils.s3_utils import REGION
from proteus.data.downloads.proteus_dataset.conf.experimental_download import ExperimentalDataDownloadCfg
from proteus.data.downloads.proteus_dataset.data_writing import ShardWriter, _serialize_pdb_blob
from proteus.data.downloads.proteus_dataset.data_parsing import _parse_mmcif

logger = logging.getLogger(__name__)


class ExperimentalDataDownload:
	def __init__(self, cfg: ExperimentalDataDownloadCfg):

		if cfg.min_chain_length < 4:
			raise RuntimeError(f"min_chain_length must be >= 4 for foldseek clustering, got {cfg.min_chain_length}")

		# filters
		self.methods = set(cfg.methods)
		self.max_resolution = cfg.max_resolution
		self.min_chain_length = cfg.min_chain_length

		# for debugging / testin
		self.max_entries = cfg.max_entries

		# other necessary stuff
		self.semaphore_limit = cfg.semaphore_limit
		self.chunk_size = cfg.chunk_size
		self.shard_size_bytes = cfg.shard_size_mb * 1024 * 1024
		self.zstd_level = cfg.zstd_level

		# paths
		self.s3_path = S3Path(cfg.s3_path)
		self.local_path = Path(cfg.local_path)
		self.checkpoint_path = Path(cfg.checkpoint_path)

	def download(self):

		# get ids, pdb redo returns all in the db, rcsb returns all with filters
		pdbredo_ids = self._get_pdbredo_ids()
		rcsb_ids = self._get_rcsb_ids()
		
		# filter out pdbs from pdb redo that arent in rcsb set, since rcsb set applied the filter
		pdbredo_set = pdbredo_ids & rcsb_ids

		# experimental should not contain any that are in pdbredo, since we give priority to higher quality pdb-redo
		rcsb_set = rcsb_ids - pdbredo_set

		pdbredo, rcsb = list(pdbredo_set), list(rcsb_set)

		# apply max_entries limit split evenly across both sources
		if self.max_entries > 0:
			half = self.max_entries // 2
			rcsb = rcsb[:half]
			pdbredo = pdbredo[:self.max_entries - len(rcsb)]

		# combine and specify where the boundary is
		combined = rcsb + pdbredo
		boundary = len(rcsb)
		logger.info(f"{len(rcsb)} rcsb-only, {len(pdbredo)} pdb-redo entries")

		return asyncio.run(self._download_async(combined, boundary))

	def _load_checkpoint(self) -> Tuple[List[dict], Set[str], int]:
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

	async def _download_async(self, combined: List[str], boundary: int = 0):
		self.local_path.mkdir(parents=True, exist_ok=True)

		# load checkpoint from previous run (if any)
		index_rows, done_pids, next_shard_id = self._load_checkpoint()
		rcsb_only = combined[:boundary]
		remaining = [pid for pid in combined if pid not in done_pids]

		# recompute boundary after removing already-done entries
		boundary = sum(1 for pid in rcsb_only if pid not in done_pids)
		logger.info(f"{len(remaining)} to download ({len(done_pids)} already done)")

		pbar = tqdm(total=len(remaining), desc="downloading")

		connector = aiohttp.TCPConnector(limit=self.semaphore_limit, enable_cleanup_closed=True)
		timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=60)
		s3_session = aioboto3.Session()
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

			async def _task(pdb_id, is_pdb_redo):
				result = await self._download_entry(
					pdb_id, session, shard_writer, shard_lock, is_pdb_redo,
				)
				pbar.update(1)
				return result

			for i in range(0, len(remaining), self.chunk_size):
				chunk_indices = range(i, min(i + self.chunk_size, len(remaining)))
				tasks = [_task(remaining[j], is_pdb_redo=(j >= boundary)) for j in chunk_indices]
				results = await asyncio.gather(*tasks, return_exceptions=True)
				for j, result in zip(chunk_indices, results):
					if isinstance(result, Exception):
						logger.error(f"{remaining[j]}: {result}")
						failed += 1
					elif result is None:
						failed += 1
					else:
						succeeded += 1

			index_rows = await shard_writer.finalize()
		pbar.close()

		logger.info(f"done: {succeeded} succeeded, {failed} skipped out of {len(remaining)}")
		return index_rows

	async def _download_entry(
		self, pdb_id: str, session: aiohttp.ClientSession,
		shard_writer: "ShardWriter", shard_lock: asyncio.Lock,
		is_pdb_redo: bool = False,
	):
		data = None
		if is_pdb_redo:
			data = await self._download_pdbredo(pdb_id, session)
		if data is None:
			data = await self._download_rcsb(pdb_id, session)
		if data is None:
			return None

		# write ca cifs locally for foldseek
		for chain_id, chain_data in data["chains"].items():
			cif_path = self.local_path / f"{pdb_id}_{chain_id}.cif.gz"
			cif_path.write_bytes(gzip.compress(chain_data["cif"].encode()))

		# pack into blob and add to shard
		blob = _serialize_pdb_blob(pdb_id, data, self.zstd_level)
		chain_ids = list(data["chains"].keys())
		meta = {
			"resolution": data["resolution"],
			"method": data["method"],
			"deposit_date": data["deposit_date"],
			"source": data["source"],
		}
		async with shard_lock:
			shard_writer.add(pdb_id, blob, chain_ids, meta)

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
					logger.error(f"failed to fetch {url} after {max_retries} retries, skipping")
					return None
				delay = 2 ** attempt
				logger.warning(f"retry {attempt + 1}/{max_retries} for {url} (waiting {delay}s)")
				await asyncio.sleep(delay)

	async def _download_pdbredo(self, pdb_id: str, session: aiohttp.ClientSession):
		# pdb-redo returns 500 (not 404) for missing entries, so treat any
		# non-200 as "not found" without retrying. still retry on network errors.
		raw = await self._fetch(
			session,
			f"https://pdb-redo.eu/db/{pdb_id}/{pdb_id}_final.cif",
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
		raw = await self._fetch(
			session,
			f"https://files.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/{pdb_id[1:3]}/{pdb_id}.cif.gz",
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

	def _get_pdbredo_ids(self) -> set[str]:
		"""fetch the set of PDB IDs available on PDB-REDO via parallel rsync listing.
		lists 2-char prefix dirs, then queries each prefix in parallel to get 4-char PDB IDs.
		returns lowercase 4-char IDs. on failure returns empty set (graceful degradation)."""

		pdb_id_re = re.compile(r'[0-9][a-z0-9]{3}')
		rsync_base = "rsync://rsync.pdb-redo.eu/pdb-redo"

		def _list_prefix(prefix: str) -> list[str]:
			r = subprocess.run(
				["rsync", "--list-only", "--no-motd", f"{rsync_base}/{prefix}/"],
				capture_output=True, text=True, timeout=30,
			)
			r.check_returncode()
			ids = []
			for line in r.stdout.splitlines():
				if not line.startswith("d"):
					continue
				name = line.split()[-1]
				if pdb_id_re.fullmatch(name):
					ids.append(name)
			return ids

		try:
			logger.info("fetching pdb-redo entry list via rsync ...")
			# get 2-char prefix directories
			result = subprocess.run(
				["rsync", "--list-only", "--no-motd", f"{rsync_base}/"],
				capture_output=True, text=True, timeout=120,
			)
			result.check_returncode()
			prefixes = []
			for line in result.stdout.splitlines():
				if not line.startswith("d"):
					continue
				name = line.split()[-1]
				if re.fullmatch(r'[0-9a-z]{2}', name):
					prefixes.append(name)
			logger.info(f"pdb-redo: {len(prefixes)} prefix dirs, listing entries ...")

			# query each prefix in parallel
			ids: set[str] = set()
			with ThreadPoolExecutor(max_workers=64) as pool:
				futures = {pool.submit(_list_prefix, p): p for p in prefixes}
				for fut in as_completed(futures):
					try:
						ids.update(fut.result())
					except Exception as e:
						logger.warning(f"pdb-redo prefix {futures[fut]}: {e}")

			logger.info(f"pdb-redo listing: {len(ids)} entries")
			return ids
		except Exception as e:
			logger.warning(f"failed to fetch pdb-redo listing: {e}. all entries will use rcsb.")
			return set()

	def _get_rcsb_ids(self) -> Set[str]:
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
			pdbids = {hit["identifier"].lower() for hit in results.get("result_set", [])}
			logger.info(f"rcsb search returned {len(pdbids)} entries")
		else:
			# fallback: get all PDB IDs from holdings and let _parse_mmcif filter
			logger.warning(f"rcsb search API returned {resp.status_code}, falling back to holdings")
			pdbids = self._get_holdings_ids()
			logger.info(f"holdings returned {len(pdbids)} entries")

		return pdbids

	def _get_holdings_ids(self) -> Set[str]:
		"""get all PDB IDs from wwPDB holdings file"""
		url = "https://files.wwpdb.org/pub/pdb/holdings/current_file_holdings.json.gz"
		logger.info(f"retrieving rcsb metadata at {url} ...")
		resp = requests.get(url, timeout=120)
		resp.raise_for_status()
		with gzip.open(io.BytesIO(resp.content), "rt") as f:
			holdings = json.load(f)
		return {pid.lower() for pid in holdings.keys()}