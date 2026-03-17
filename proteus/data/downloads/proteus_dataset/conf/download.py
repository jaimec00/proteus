from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from proteus.data.data_constants import ExpMethods


@dataclass
class DownloadMethodCfg:
	s3_path: str = "${...s3_path}"
	local_path: str = "${...local_path}"
	checkpoint_path: str = "${...local_path}/checkpoint.jsonl"
	shard_size_mb: int = 256
	zstd_level: int = 10
	semaphore_limit: int = 32
	pipeline_concurrency: int = 256


@dataclass
class ExperimentalDataDownloadCfg(DownloadMethodCfg):
	# filtering
	methods: list = field(default_factory=lambda: [ExpMethods.XRAY, ExpMethods.CRYO_EM])
	max_resolution: float = 3.5
	max_entries: int = 10_000  # -1 = all, testing for now
	min_chain_length: int = 8 # skip chains shorter than this (foldseek requires >= 4)
	pipeline_concurrency: int = 256
	pdbredo_cache_ttl_hours: int = 24
	skip_tm_at_seqsim: float = 0.95
	_impl_cls: str = "proteus.data.downloads.proteus_dataset.download.ExperimentalDataDownload"


def register_download():
	cs = ConfigStore.instance()
	cs.store("rcsb-pdb", node=[ExperimentalDataDownloadCfg()], group="download_methods")
