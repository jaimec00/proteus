from dataclasses import dataclass, field
from hydra.conf import RunDir, SweepDir, HydraConf
from hydra.core.config_store import ConfigStore

@dataclass
class ExperimentalDataDownloadCfg:
	# filtering
	methods: list = field(default_factory=lambda: ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"])
	max_resolution: float = 3.5
	max_entries: int = -1  # -1 = all

	# pipeline
	semaphore_limit: int = 32
	chunk_size: int = 1000
	s3_path: str = "s3://proteus-data-bucket"
	local_path: str = "/home/ubuntu/proteus/data/tmp"

@dataclass
class DataPipelineCfg:
	experimental_dl: ExperimentalDataDownloadCfg = field(default_factory=ExperimentalDataDownloadCfg)

@dataclass
class DownloadHydra(HydraConf):
	run: RunDir = field(default_factory=lambda: RunDir("/home/ubuntu/proteus/data/download/logs/${now:%Y-%m-%d}/${now:%H-%M-%S}"))

def register_download_configs():
	cs = ConfigStore.instance()
	cs.store("default", DataPipelineCfg)
	cs.store("config", DownloadHydra, group="hydra")
