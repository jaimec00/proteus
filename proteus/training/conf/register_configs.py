from proteus.training.conf.data import register_data
from proteus.training.conf.logger import register_logger
from proteus.training.conf.optim import register_optim
from proteus.training.conf.profiler import register_profiler
from proteus.training.conf.scheduler import register_scheduler
from proteus.training.conf.losses import register_losses
from proteus.training.conf.training_params import register_training_params
from proteus.training.conf.checkpoint import register_checkpoint
from proteus.training.conf.hydra import register_hydra

def register_configs():
    register_data()
    register_logger()
    register_optim()
    register_profiler()
    register_scheduler()
    register_losses()
    register_training_params()
    register_checkpoint()
    register_hydra()