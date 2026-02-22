from proteus.training.logger import LoggerCfg
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

@dataclass
class DefaultLogger(LoggerCfg):
    run_name: str = "${hydra:job.override_dirname}"
    experiment_name: str = "debug"
    overwrite: bool = True
    log_system_metrics: bool = True
    system_metrics_sample_interval: int = 20
    system_metrics_log_interval: int = 10
    log_interval: int = 50

def register_logger():
    cs = ConfigStore.instance()
    cs.store("default", DefaultLogger, group="logger")
