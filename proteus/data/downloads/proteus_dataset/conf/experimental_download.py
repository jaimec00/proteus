from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from proteus.data.data_constants import ExpMethods

@dataclass
class ExperimentalDataDownloadCfg:
	# filtering
	methods: list = field(default_factory=lambda: [ExpMethods.XRAY, ExpMethods.CRYO_EM])
	max_resolution: float = 3.5
	max_entries: int = 1024  # -1 = all, testing for now
	min_chain_length: int = 8 # skip chains shorter than this (foldseek requires >= 4)

	# pipeline
	semaphore_limit: int = 32
	chunk_size: int = 1000
	shard_size_mb: int = 256
	zstd_level: int = 10
	s3_path: str = "${..s3_path}"
	local_path: str = "${foldseek.input_path}"
	checkpoint_path: str = "${..local_path}/checkpoint.jsonl"

def register_experimental_download():
	cs = ConfigStore.instance()
	cs.store("default", ExperimentalDataDownloadCfg, group="experimental_dl")
