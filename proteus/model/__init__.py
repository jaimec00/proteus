from .base import Base
from . import transformer, model_utils

import enum, auto
class OutputNames(enum.StrEnum):
    SEQ_LOGITS = auto()
    SEQ_LABELS = auto()
    LOSS_MASK = auto()
    AA_MAGNITUDES = auto()

__all__ = [
    "Base",
    "transformer",
    "model_utils",
    "OutputNames"
]