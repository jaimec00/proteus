from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from proteus.data.downloads.proteus_dataset.conf.download import DownloadMethodCfg
from proteus.data.downloads.proteus_dataset.conf.cluster import ClusterMethodCfg


@dataclass
class DataPipelineBaseCfg:
	download_methods: list[DownloadMethodCfg] = MISSING
	cluster_methods: list[ClusterMethodCfg] = MISSING
	s3_path: str = MISSING
	local_path: str = MISSING
