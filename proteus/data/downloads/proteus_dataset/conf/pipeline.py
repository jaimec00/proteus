from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from proteus.data.downloads.proteus_dataset.conf.experimental_download import ExperimentalDataDownloadCfg
from proteus.data.downloads.proteus_dataset.conf.foldseek import FoldSeekCfg


@dataclass
class DataPipelineBaseCfg:
	experimental_dl: ExperimentalDataDownloadCfg = MISSING
	foldseek: FoldSeekCfg = MISSING
	s3_path: str = MISSING
	local_path: str = MISSING
