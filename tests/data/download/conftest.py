import pytest

from proteus.data.data_constants import ExpMethods, ClusterInputType
from proteus.data.downloads.proteus_dataset.conf.download import ExperimentalDataDownloadCfg
from proteus.data.downloads.proteus_dataset.download.experimental import ExperimentalDataDownload


@pytest.fixture
def exp_cfg(tmp_path):
	return ExperimentalDataDownloadCfg(
		s3_path="s3://test-bucket/data",
		local_path=str(tmp_path / "local"),
		checkpoint_path=str(tmp_path / "checkpoint.jsonl"),
		shard_size_mb=1,
		zstd_level=1,
		semaphore_limit=2,
		chunk_size=10,
		methods=[ExpMethods.XRAY],
		max_resolution=3.5,
		max_entries=10,
		min_chain_length=4,
		pdbredo_cache_ttl_hours=24,
	)


@pytest.fixture
def exp_download(exp_cfg):
	return ExperimentalDataDownload(exp_cfg, required_inputs={ClusterInputType.MMCIF, ClusterInputType.FASTA})
