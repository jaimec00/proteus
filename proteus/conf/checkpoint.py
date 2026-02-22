from proteus.training.training_run import CheckpointCfg

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class NoCheckpoint(CheckpointCfg):
    load_from_checkpoint: str = ""
    reset_state: bool = False

def register_checkpoint():
    cs = ConfigStore.instance()
    cs.store(name="no_ckpt", node=NoCheckpoint, group="checkpoint")