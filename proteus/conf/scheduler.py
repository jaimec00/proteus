from proteus.training.scheduler import SchedulerCfg, LRSchedules
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import II

@dataclass
class SqrtScheduler(SchedulerCfg):
    d_model: int = II("model.d_model")
    lr_type: str = LRSchedules.SQRT
    lr_step: float = 1e-4
    warmup_steps: int = 5_000

@dataclass
class StaticScheduler(SchedulerCfg):
    d_model: int = II("model.d_model")
    lr_type: str = LRSchedules.STATIC
    lr_step: float = 2.0e-5

def register_scheduler():
    cs = ConfigStore.instance()
    cs.store("sqrt", SqrtScheduler, group="scheduler")
    cs.store("static", StaticScheduler, group="scheduler")