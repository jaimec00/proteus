from proteus.static.constants import aa_2_lbl, canonical_aas
from proteus.types import Int, Bool, Tuple, T, Optional
from dataclasses import dataclass
import torch


@dataclass
class MaskerCfg:
    mask_rate: float = 0.15

class Masker:
    def __init__(self, cfg: MaskerCfg):
        self.mask_rate = cfg.mask_rate

    def mask_labels(self, labels: Int[T, "BL"], also_mask: Optional[Bool[T, "BL"]] = None, only_mask: Optional[Bool[T, "BL"]] = None) -> Tuple[Int[T, "BL"], Bool[T, "BL"]]:
        
        also_mask = also_mask if also_mask is not None else torch.zeros_like(labels, dtype=torch.bool)
        only_mask = only_mask if only_mask is not None else torch.ones_like(labels, dtype=torch.bool)
        if self.mask_rate == 0.0: # not bert style, just predict aa directly
            is_masked = torch.ones_like(also_mask)
            masked_labels = labels
        else:
            is_masked = ((torch.rand_like(labels, dtype=torch.float) <= self.mask_rate) | also_mask) & only_mask
            
            # TODO: make mask method configurable (dedicated mask token vs random aa)
            # masked_labels = labels.masked_fill(is_masked, aa_2_lbl("<mask>"))

            rand_aas = torch.randint(0, len(canonical_aas), labels.shape, device=labels.device)
            masked_labels = labels.where(is_masked, rand_aas)
            
        return masked_labels, is_masked