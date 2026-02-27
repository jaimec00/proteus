from proteus.static.constants import aa_2_lbl, canonical_aas
from proteus.types import Int, Bool, Tuple, T, Optional
from dataclasses import dataclass
import omegaconf
import torch
import enum

class NoiseScheduleTypes(enum.StrEnum):
    NONE = enum.auto()
    CONSTANT = enum.auto()
    LINEAR = enum.auto()

@dataclass
class NoiseScheduleCfg:
    schedule: str = omegaconf.MISSING
    max_t: int = 1
    rand_mask: bool = True # found that dedicated mask overfits
    corruption_rate: float = 0.00

@dataclass
class NoNoiseScheduleCfg(NoiseScheduleCfg):
    schedule: str = NoiseScheduleTypes.NONE

@dataclass
class LinearNoiseScheduleCfg(NoiseScheduleCfg):
    schedule: str = NoiseScheduleTypes.LINEAR
    max_t: int = 10
    rand_mask: bool = True

@dataclass
class ConstantNoiseScheduleCfg(NoiseScheduleCfg):
    schedule: str = NoiseScheduleTypes.CONSTANT
    rand_mask: bool = True
    corruption_rate: float = 0.00


class NoiseSchedule:
    def __init__(self, cfg: NoiseScheduleCfg):
        self.schedule = cfg.schedule
        self.max_t = cfg.max_t
        self.corruption_rate = cfg.corruption_rate
        self.rand_mask = cfg.rand_mask

    @torch.no_grad()
    def corrupt_labels(
        self, 
        labels: Int[T, "BL"], 
        sample_idx: Int[T, "BL"],
        cu_seqlens: Int[T, "BL"],
        also_mask: Optional[Bool[T, "BL"]] = None, 
        only_mask: Optional[Bool[T, "BL"]] = None,
    ) -> Tuple[Int[T, "BL"], Bool[T, "BL"]]:
        
        t = self._sample_t(sample_idx, cu_seqlens)
        corruption = self._get_corruption(t)
        is_corrupted = self._sample_corruption(corruption, also_mask, only_mask)        
        corrupted_labels = self._corrupt(labels, is_corrupted)

        return corrupted_labels, is_corrupted, t

    @torch.no_grad()
    def _sample_t(self, sample_idx, cu_seqlens):
        num_samples = cu_seqlens.size(0) - 1
        samples_t = torch.randint(0, self.max_t, (num_samples,), device=cu_seqlens.device)
        t = torch.gather(samples_t, 0, sample_idx)
        return t

    @torch.no_grad()
    def _get_corruption(self, t):
        return getattr(self, self.schedule)(t)

    @torch.no_grad()
    def _sample_corruption(self, corruption, also_mask, only_mask):
        also_mask = also_mask if also_mask is not None else torch.zeros_like(corruption, dtype=torch.bool)
        only_mask = only_mask if only_mask is not None else torch.ones_like(corruption, dtype=torch.bool)
        is_corrupted = (torch.bernoulli(corruption).bool() | also_mask) & only_mask
        return is_corrupted

    @torch.no_grad()
    def _corrupt(self, labels, is_corrupted):
        if self.rand_mask:
            rand_aas = torch.randint(0, len(canonical_aas), labels.shape, device=labels.device)
            corrupted_labels = labels.where(is_corrupted, rand_aas)
        else:
            corrupted_labels = labels.masked_fill(is_corrupted, aa_2_lbl("<mask>"))

        return corrupted_labels

    @torch.no_grad()
    def none(self, t):
        return torch.zeros_like(t)

    @torch.no_grad()
    def linear(self, t):
        return t / self.max_t

    @torch.no_grad()
    def constant(self, t):
        return torch.full_like(t, self.mask_rate)
