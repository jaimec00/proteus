from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
import hydra

from proteus.data.downloads.proteus_dataset.conf import register_download_configs, DataPipelineBaseCfg
from proteus.data.downloads.proteus_dataset.data_pipeline import DataPipeline

defaults = [
	"_self_",
	{"experimental_dl": "default"},
	{"foldseek": "default"},
]

@dataclass
class DataPipelineCfg(DataPipelineBaseCfg):
	defaults: list = field(default_factory=defaults) 
	s3_path: str = "s3://proteus-data-bucket"
	local_path: str = "/home/ubuntu/proteus/data/tmp"

register_download_configs()
cs = ConfigStore.instance()
cs.store("default", DataPipelineCfg)

@hydra.main(version_base=None, config_name="default")
def main(cfg: DataPipelineCfg):
	pipeline = DataPipeline(cfg)
	pipeline.run()

if __name__ == "__main__":
	main()
