from dataclasses import dataclass
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Optimizer
import enum


class LRSchedules(enum.StrEnum):
    STATIC = "static"
    SQRT = "sqrt"

@dataclass
class SchedulerCfg:
    d_model: int
    lr_type: str = LRSchedules.STATIC
    lr_step: float = 5e-3

def setup_scheduler(cfg: SchedulerCfg, optim: Optimizer):

    if cfg.lr_type == LRSchedules.SQRT:

        # compute the scale
        if cfg.lr_step == 0.0:
            scale = cfg.d_model**(-0.5)
        else:
            scale = cfg.warmup_steps**(0.5) * cfg.lr_step # scale needed so max lr is what was specified

        def sqrt(step):
            '''sqrt scheduler from attn paper'''
            step = step # in case job gets cancelled and want to start from where left off
            return scale * min((step+1)**(-0.5), (step+1)*(cfg.warmup_steps**(-1.5)))

        scheduler = lr_scheduler.LambdaLR(optim, sqrt)

    elif cfg.lr_type == LRSchedules.STATIC:
        def static(step):
            return cfg.lr_step
        scheduler = lr_scheduler.LambdaLR(optim, static)

    else:
        raise ValueError(f"invalid lr_type: {cfg.lr_type}. options are ['{LRSchedules.SQRT}', '{LRSchedules.STATIC}']")

    return scheduler