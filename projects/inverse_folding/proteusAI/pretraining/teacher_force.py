
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from proteus.training import TrainingRun, TrainingRunCfg
from proteus.conf.register_configs import register_configs
from proteus.model.composed.inverse_folding.proteusAI import proteusAICfg
from proteus.model.tokenizer import WaveFunctionTokenizerCfg
from proteus.utils.masking import MaskerCfg
from proteus.model.transformer import (
    TransformerBlockCfg, 
    TransformerModelCfg, 
    MHACfg,
)
from proteus.data.construct_registry import ConstructFunctionNames
from proteus.types import Any

defaults = [
    "_self_",
    {"data": "medium_seq"},
    {"logger": "default"},
    {"losses": "cel_aa"},
    {"optim": "adamw"},
    {"scheduler": "sqrt"},
    {"profiler": "no_profile"},
    {"checkpoint": "no_ckpt"},
    {"training_params": "default"},
]

@dataclass
class proteusAIPretrainBaseCfg(TrainingRunCfg):
    defaults: list = field(default_factory=lambda: defaults)
    construct_function: str = ConstructFunctionNames.PROTEUS
    model: proteusAICfg = MISSING

# everything is interpolated, can be more specific if you like
D_WF, D_MODEL = 256, 512
MIN_WL, MAX_WL, BASE_WL = 3.0, 25.0, 20.0
TRANSFORMER_LAYERS, HEADS = 4, 4

# experiments with no masking to see 
# minimum capacity needed to make predictions
MASK_RATE = 0.0 

@dataclass
class proteusAIPretrainTeacherForceCfg(proteusAIPretrainBaseCfg):
    model: proteusAICfg = field(default_factory=lambda: proteusAICfg(
        d_model=D_MODEL,
        tokenizer=WaveFunctionTokenizerCfg(
            d_wf=D_WF, 
            min_wl=MIN_WL, 
            max_wl=MAX_WL, 
            base_wl=BASE_WL
        ),
        masker=MaskerCfg(mask_rate=MASK_RATE),
        transformer=TransformerModelCfg(
            transformer_block=TransformerBlockCfg(
                attn=MHACfg(heads=HEADS)
            ),
            layers = TRANSFORMER_LAYERS,
        ),
    ))

register_configs()
cs = ConfigStore.instance()
cs.store(name="teacher_force", node=proteusAIPretrainTeacherForceCfg)

@hydra.main(version_base=None, config_name="teacher_force")
def main(cfg):
    TrainingRun(cfg)

if __name__ == "__main__":
    main()