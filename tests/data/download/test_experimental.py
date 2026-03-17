import gzip
import json
import asyncio
import time
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

import aiohttp
from aioresponses import aioresponses

from proteus.data.data_constants import ExpMethods, ClusterInputType, IndexCol
from proteus.data.downloads.proteus_dataset.download.experimental import ExperimentalDataDownload


# ---------------------------------------------------------------------------
# TestExperimentalInit
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestExperimentalInit:

	def test_min_chain_length_too_small(self, exp_cfg):
		exp_cfg.min_chain_length = 3
		with pytest.raises(RuntimeError, match="min_chain_length must be >= 4"):
			ExperimentalDataDownload(exp_cfg, required_inputs=set())

	def test_min_chain_length_ok(self, exp_cfg):
		exp_cfg.min_chain_length = 4
		dl = ExperimentalDataDownload(exp_cfg, required_inputs=set())
		assert dl.min_chain_length == 4


# ---------------------------------------------------------------------------
# TestDownloadSetOps
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestDownloadSetOps:

	def test_deduplication_logic(self, exp_download):
		with patch.object(exp_download, "_get_pdbredo_ids", return_value={"1abc", "2def", "3ghi"}), \
			patch.object(exp_download, "_get_rcsb_ids", return_value={"1abc", "2def", "4jkl", "5mno"}), \
			patch.object(exp_download, "_download_async", new_callable=AsyncMock) as mock_dl:

			mock_dl.return_value = []
			exp_download.download()

			args = mock_dl.call_args
			combined = args[0][0]
			boundary = args[0][1]

			# rcsb-only: {4jkl, 5mno}, pdbredo intersection: {1abc, 2def}
			rcsb_only = set(combined[:boundary])
			pdbredo = set(combined[boundary:])
			assert rcsb_only == {"4jkl", "5mno"}
			assert pdbredo == {"1abc", "2def"}

	def test_max_entries_splitting(self, exp_cfg):
		exp_cfg.max_entries = 6
		dl = ExperimentalDataDownload(exp_cfg, required_inputs=set())

		with patch.object(dl, "_get_pdbredo_ids", return_value=set(f"pdb{i}" for i in range(10))), \
			patch.object(dl, "_get_rcsb_ids", return_value=set(f"rcsb{i}" for i in range(10)) | set(f"pdb{i}" for i in range(10))), \
			patch.object(dl, "_download_async", new_callable=AsyncMock) as mock_dl:

			mock_dl.return_value = []
			dl.download()

			combined = mock_dl.call_args[0][0]
			assert len(combined) <= 6


# ---------------------------------------------------------------------------
# TestFetch
# ---------------------------------------------------------------------------

@pytest.mark.cpu
@pytest.mark.asyncio
class TestFetch:

	async def test_retry_on_server_error(self, exp_download):
		with aioresponses() as m:
			url = "http://test.example.com/data"
			m.get(url, status=500)
			m.get(url, status=500)
			m.get(url, status=200, body=b"ok")

			connector = aiohttp.TCPConnector(limit=0)
			timeout = aiohttp.ClientTimeout(total=10)
			exp_download._semaphore = asyncio.Semaphore(10)
			async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
				with patch("proteus.data.downloads.proteus_dataset.download.experimental.asyncio.sleep", new_callable=AsyncMock):
					result = await exp_download._fetch(session, url, max_retries=3)

			assert result == b"ok"

	async def test_non_retryable_returns_none(self, exp_download):
		with aioresponses() as m:
			url = "http://test.example.com/missing"
			m.get(url, status=404)

			connector = aiohttp.TCPConnector(limit=0)
			timeout = aiohttp.ClientTimeout(total=10)
			exp_download._semaphore = asyncio.Semaphore(10)
			async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
				result = await exp_download._fetch(session, url, max_retries=3)

			assert result is None

	async def test_timeout_retries(self, exp_download):
		with aioresponses() as m:
			url = "http://test.example.com/slow"
			m.get(url, exception=asyncio.TimeoutError())
			m.get(url, exception=asyncio.TimeoutError())
			m.get(url, status=200, body=b"finally")

			connector = aiohttp.TCPConnector(limit=0)
			timeout = aiohttp.ClientTimeout(total=10)
			exp_download._semaphore = asyncio.Semaphore(10)
			async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
				with patch("proteus.data.downloads.proteus_dataset.download.experimental.asyncio.sleep", new_callable=AsyncMock):
					result = await exp_download._fetch(session, url, max_retries=3)

			assert result == b"finally"

	async def test_all_retries_exhausted(self, exp_download):
		with aioresponses() as m:
			url = "http://test.example.com/fail"
			m.get(url, status=500)
			m.get(url, status=500)
			m.get(url, status=500)

			connector = aiohttp.TCPConnector(limit=0)
			timeout = aiohttp.ClientTimeout(total=10)
			exp_download._semaphore = asyncio.Semaphore(10)
			async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
				with patch("proteus.data.downloads.proteus_dataset.download.experimental.asyncio.sleep", new_callable=AsyncMock):
					result = await exp_download._fetch(session, url, max_retries=3)

			assert result is None


# ---------------------------------------------------------------------------
# TestDownloadIntegration
# ---------------------------------------------------------------------------

@pytest.mark.cpu
@pytest.mark.asyncio
class TestDownloadIntegration:

	async def test_full_flow(self, exp_cfg, mmcif_builder, tmp_path):
		"""mock S3, mock IDs, mock wwPDB URL with gzip mmCIF, verify end-to-end"""
		# build a valid mmCIF and gzip it
		cif_content = mmcif_builder(pdb_id="1abc", method=ExpMethods.XRAY, resolution=2.0, chains=[("A", "ACDEFG")])
		cif_gz = gzip.compress(cif_content.encode())

		exp_cfg.max_entries = -1
		dl = ExperimentalDataDownload(exp_cfg, required_inputs={ClusterInputType.MMCIF, ClusterInputType.FASTA})

		with aioresponses() as m:
			# mock the wwPDB URL
			m.get(
				"https://files.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/ab/1abc.cif.gz",
				body=cif_gz,
			)

			# mock S3
			s3_mock = AsyncMock()

			with patch("proteus.data.downloads.proteus_dataset.download.experimental.aioboto3") as mock_aioboto3:
				mock_session = MagicMock()
				mock_aioboto3.Session.return_value = mock_session
				mock_cm = AsyncMock()
				mock_cm.__aenter__ = AsyncMock(return_value=s3_mock)
				mock_cm.__aexit__ = AsyncMock(return_value=False)
				mock_session.client.return_value = mock_cm

				# call _download_async directly to avoid asyncio.run() inside running loop
				# combined = ["1abc"], boundary = 1 means all are rcsb-only
				rows = await dl._download_async(["1abc"], boundary=1)

		assert len(rows) >= 1
		assert rows[0][IndexCol.PDB] == "1abc"

		# check raw files written
		local = tmp_path / "local"
		mmcif_files = list((local / "raw_mmcif").glob("*.cif.gz"))
		fasta_files = list((local / "raw_fasta").glob("*.fasta"))
		assert len(mmcif_files) >= 1
		assert len(fasta_files) >= 1

		# checkpoint written
		checkpoint = tmp_path / "checkpoint.jsonl"
		assert checkpoint.exists()


# ---------------------------------------------------------------------------
# TestPdbredoCache
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestPdbredoCache:

	def test_cache_hit_skips_rsync(self, exp_download):
		"""valid cache within TTL should skip rsync entirely"""
		local = exp_download.local_path
		local.mkdir(parents=True, exist_ok=True)
		cache_path = local / "pdbredo_ids_cache.json"
		cache_path.write_text(json.dumps({
			"timestamp": time.time(),
			"ids": ["1abc", "2def"],
		}))

		with patch("subprocess.run") as mock_rsync:
			ids = exp_download._get_pdbredo_ids()

		mock_rsync.assert_not_called()
		assert ids == {"1abc", "2def"}

	def test_cache_expired_runs_rsync(self, exp_download):
		"""cache older than TTL should trigger rsync"""
		local = exp_download.local_path
		local.mkdir(parents=True, exist_ok=True)
		cache_path = local / "pdbredo_ids_cache.json"
		cache_path.write_text(json.dumps({
			"timestamp": time.time() - (exp_download.pdbredo_cache_ttl_hours + 1) * 3600,
			"ids": ["old"],
		}))

		# mock rsync to return a prefix list then entry list
		def mock_run(cmd, **kwargs):
			result = MagicMock()
			result.returncode = 0
			result.check_returncode = MagicMock()
			rsync_url = cmd[-1]
			if rsync_url.endswith("pdb-redo/"):
				result.stdout = "drwxr-xr-x 4096 2024/01/01 00:00:00 ab\n"
			else:
				result.stdout = "drwxr-xr-x 4096 2024/01/01 00:00:00 1abc\n"
			return result

		with patch("proteus.data.downloads.proteus_dataset.download.experimental.subprocess.run", side_effect=mock_run):
			ids = exp_download._get_pdbredo_ids()

		assert "1abc" in ids
		assert "old" not in ids

	def test_cache_corrupted_falls_back(self, exp_download):
		"""corrupted cache file should gracefully fall back to rsync"""
		local = exp_download.local_path
		local.mkdir(parents=True, exist_ok=True)
		cache_path = local / "pdbredo_ids_cache.json"
		cache_path.write_text("not valid json {{{")

		def mock_run(cmd, **kwargs):
			result = MagicMock()
			result.returncode = 0
			result.check_returncode = MagicMock()
			rsync_url = cmd[-1]
			if rsync_url.endswith("pdb-redo/"):
				result.stdout = "drwxr-xr-x 4096 2024/01/01 00:00:00 ab\n"
			else:
				result.stdout = "drwxr-xr-x 4096 2024/01/01 00:00:00 2def\n"
			return result

		with patch("proteus.data.downloads.proteus_dataset.download.experimental.subprocess.run", side_effect=mock_run):
			ids = exp_download._get_pdbredo_ids()

		assert "2def" in ids
