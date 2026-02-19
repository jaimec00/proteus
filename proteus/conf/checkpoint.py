from proteus.training.training_run import CheckpointCfg

from dataclasses import dataclass

@dataclass
class NoCheckpoint(CheckpointCfg):
    pass

def register_checkpoint():
    cs = ConfigStore.instance()
    cs.store(name="no_ckpt", node=NoCheckpoint, group="checkpoint")