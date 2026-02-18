from proteus.static.constants import aa_2_lbl
from proteus.types import Int, Bool, Tuple, T
from dataclasses import dataclass
import torch


@dataclass
class MaskerCfg:
    mask_rate: float = 0.15

class Masker:
    def __init__(self, cfg: MaskerCfg):
        self.mask_rate = cfg.mask_rate

    def mask_labels(self, labels: Int[T, "BL"], also_mask: Bool[T, "BL"]) -> Tuple[Int[T, "BL"], Bool[T, "BL"]]:
        
        if self.mask_rate == 0.0: # not bert style, just predict aa directly
            is_masked = torch.ones_like(also_mask)
            masked_labels = labels
        else:
            is_masked = (torch.rand_like(labels, dtype=torch.float) <= self.mask_rate) | also_mask
            masked_labels = labels.masked_fill(is_masked, aa_2_lbl("<mask>"))
            
        return masked_labels, is_masked