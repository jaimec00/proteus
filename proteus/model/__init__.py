from .base import Base
from . import transformer, model_utils

import enum
class OutputNames(enum.StrEnum):
    SEQ_LOGITS = enum.auto()
    SEQ_LABELS = enum.auto()
    LOSS_MASK = enum.auto()
    AA_MAGNITUDES = enum.auto()
    WF_RAW = enum.auto()
    NO_MASK = enum.auto()

__all__ = [
    "Base",
    "transformer",
    "model_utils",
    "OutputNames"
]