from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import hydra

from proteus.data.downloads.proteus_dataset.conf import register_download_configs, DataPipelineBaseCfg
from proteus.data.downloads.proteus_dataset.data_pipeline import DataPipeline

defaults = [
	"_self_",
	{"cluster_methods": "sweep"},
	{"download_methods": "rcsb-pdb"},
]

@dataclass
class DataPipelineCfg(DataPipelineBaseCfg):
	defaults: list = field(default_factory=lambda: defaults)
	s3_path: str = "s3://proteus-data-bucket"
	local_path: str = "/home/ubuntu/proteus/data/tmp"

register_download_configs()
cs = ConfigStore.instance()
cs.store("default", DataPipelineCfg)

@hydra.main(version_base=None, config_name="default")
def main(cfg: DataPipelineCfg):
    profile = False
    if profile:
        from proteus.data.downloads.proteus_dataset.profile_run import RunProfiler, print_summary, print_stage_summary
        from pathlib import Path
        with RunProfiler(path="profile.jsonl", interval=0.5) as profiler:
            pipeline = DataPipeline(cfg)
            pipeline.run(profiler=profiler)

        print_summary(Path("profile.jsonl"))
        stage_path = Path(cfg.local_path) / "stage_times.jsonl"
        if stage_path.exists():
            print_stage_summary(stage_path)
    else:
        pipeline = DataPipeline(cfg)
        pipeline.run()


if __name__ == "__main__":
	main()
