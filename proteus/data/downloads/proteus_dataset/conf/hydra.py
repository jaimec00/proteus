from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from hydra.conf import RunDir, SweepDir, HydraConf

@dataclass
class DownloadHydra(HydraConf):
	run: RunDir = field(default_factory=lambda: RunDir("/home/ubuntu/proteus/data/download/logs/${now:%Y-%m-%d}/${now:%H-%M-%S}"))

def register_hydra():
	cs = ConfigStore.instance()
	cs.store("config", DownloadHydra, group="hydra")
