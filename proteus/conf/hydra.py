from dataclasses import dataclass, field
from hydra.conf import RunDir, SweepDir, HydraConf, JobConf
from hydra.core.config_store import ConfigStore

@dataclass
class Hydra(HydraConf):
  run: RunDir = field(default_factory=lambda: RunDir("/home/ubuntu/proteinDiffVirgina/experiments/run/${now:%Y-%m-%d}/${now:%H-%M-%S}"))
  sweep: SweepDir = field(default_factory=lambda: SweepDir(
    dir = "/home/ubuntu/proteinDiffVirgina/experiments/sweep/${now:%Y-%m-%d}/${now:%H-%M-%S}",
    subdir = "${hydra:job.override_dirname}"
  ))
  job: JobConf = field(default_factory=lambda: JobConf(
    config=JobConf.JobConfig(
      override_dirname=JobConf.JobConfig.OverrideDirname(
        exclude_keys=["logger.experiment_name"]
      )
    )
  ))

  
def register_hydra():
  cs = ConfigStore.instance()
  cs.store("config", Hydra, group="hydra")