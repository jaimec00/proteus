
from proteus.training.optim import OptimCfg
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class AdamW(OptimCfg):
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    weight_decay: float = 0.01

@dataclass
class AdamWSkipAA(AdamW):
    skip_weight_decay: list = field(default_factory=lambda: ["tokenizer.aa_magnitudes"])

@dataclass
class Adam(OptimCfg):
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    weight_decay: float = 0.0

def register_optim():
    cs = ConfigStore.instance()
    cs.store("adamw", AdamW, group="optim")
    cs.store("adamw_skip_aa", AdamW, group="optim")
    cs.store("adam", Adam, group="optim")
