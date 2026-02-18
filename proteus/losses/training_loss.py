from __future__ import annotations

import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F

from proteus.static.constants import lbl_2_aa
from typing import Dict, List, Optional, Tuple, Any
from proteus.types import Float, Int, Bool, T

from dataclasses import dataclass, field
from omegaconf import MISSING, OmegaConf, DictConfig

# ----------------------------------------------------------------------------------------------------------------------
# config dataclasses

@dataclass
class LossTermCfg:
	fn: str = MISSING
	inputs: List[str] = MISSING
	kwargs: Dict[str, float] = field(default_factory=dict)
	weight: float = 0.0
	reductions: List[str] = field(default_factory=lambda: ["sum"])

@dataclass
class LossFnCfg:
	pass

# ----------------------------------------------------------------------------------------------------------------------
# loss output

@dataclass
class LossOutput:
	full_loss: torch.Tensor
	values: Dict[str, torch.Tensor]
	counts: Dict[str, torch.Tensor]

# ----------------------------------------------------------------------------------------------------------------------
# loss holder

class LossHolder:
	"""running accumulator for loss outputs — all ops stay on-device until get_metrics()"""

	def __init__(self) -> None:
		self.values: Dict[str, torch.Tensor] = {}
		self.counts: Dict[str, torch.Tensor] = {}
		self._last_full_loss: Optional[torch.Tensor] = None
		# data metrics
		self.total_tokens: int = 0
		self.total_loss_tokens: Optional[torch.Tensor] = None
		self.total_samples: int = 0
		self.num_batches: int = 0

	def add(self, output: LossOutput) -> None:
		"""accumulate loss output in-place, no cpu syncs"""
		self._last_full_loss = output.full_loss
		for k, v in output.values.items():
			reduction = k.split("/")[1]
			if k not in self.values:
				self.values[k] = v.clone()
				self.counts[k] = output.counts[k].clone()
			elif reduction == "max":
				self.values[k] = torch.maximum(self.values[k], v)
			elif reduction == "min":
				self.values[k] = torch.minimum(self.values[k], v)
			elif reduction == "last":
				self.values[k] = v.clone()
				self.counts[k] = output.counts[k].clone()
			else:
				self.values[k] = self.values[k] + v
				self.counts[k] = self.counts[k] + output.counts[k]

	def add_data(self, data_batch) -> None:
		"""accumulate data metrics from a batch"""
		self.total_tokens += data_batch.tokens
		loss_tokens = data_batch.loss_tokens
		if self.total_loss_tokens is None:
			self.total_loss_tokens = loss_tokens.clone()
		else:
			self.total_loss_tokens = self.total_loss_tokens + loss_tokens
		self.total_samples += data_batch.samples
		self.num_batches += 1

	def get_metrics(self) -> Dict[str, float]:
		"""final reduction — single cpu/gpu sync point for logging"""
		metrics = {}
		for k in self.values:
			reduction = k.split("/")[1]
			if reduction in ("max", "min", "last"):
				metrics[k] = self.values[k].item()
			else:
				metrics[k] = self.values[k].item() / max(1, self.counts[k].item())
		return metrics

	def get_last_loss(self) -> torch.Tensor:
		return self._last_full_loss

	def clear(self) -> None:
		self.values.clear()
		self.counts.clear()
		self._last_full_loss = None
		self.total_tokens = 0
		self.total_loss_tokens = None
		self.total_samples = 0
		self.num_batches = 0


class TrainingRunLosses:

	def __init__(self, loss_fn_cfg: LossFnCfg) -> None:
		if isinstance(loss_fn_cfg, DictConfig):
			loss_fn_cfg = OmegaConf.to_object(loss_fn_cfg)
		self.loss_fn: LossFn = LossFn(loss_fn_cfg)
		self.tmp: LossHolder = LossHolder()

# ----------------------------------------------------------------------------------------------------------------------
# loss function

class LossFn(nn.Module):
	def __init__(self, cfg: LossFnCfg):
		super().__init__()
		# discover loss terms from cfg dataclass fields
		self.terms: Dict[str, LossTermCfg] = {}
		for f in dataclasses.fields(cfg):
			val = getattr(cfg, f.name)
			if isinstance(val, LossTermCfg):
				assert hasattr(self, val.fn), f"LossFn has no '{val.fn}' method"
				self.terms[f.name] = val

	def forward(self, outputs: Dict[str, Any]) -> LossOutput:
		full_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device if outputs else "cpu")
		values: Dict[str, torch.Tensor] = {}
		counts: Dict[str, torch.Tensor] = {}

		for term_name, term in self.terms.items():

			# gather inputs
			args = [outputs[k] for k in term.inputs]
			fn_result = getattr(self, term.fn)(*args, **term.kwargs)

			multi_output = len(fn_result) > 1
			# first reduction in the list is used for backprop contribution
			applied_weight = False

			for fn_key, (unreduced, mask) in fn_result.items():
				for reduction in term.reductions:
					scalar, stored, count = self._reduce(unreduced, mask, reduction)

					# build the log key: term/reduction or term/reduction/sub_key
					if multi_output:
						log_key = f"{term_name}/{reduction}/{fn_key}"
					else:
						log_key = f"{term_name}/{reduction}"

					# weighted contribution to full_loss (first key, first reduction only)
					if not applied_weight and term.weight != 0.0:
						full_loss = full_loss + scalar * term.weight
						applied_weight = True

					# detach stored values — only used for logging, no need to retain graph
					values[log_key] = stored.detach()
					counts[log_key] = count

				applied_weight = True

		# log raw full_loss scalar (count=1 so accumulator averages over batches)
		values["full_loss/sum"] = full_loss.detach()
		counts["full_loss/sum"] = full_loss.detach().new_tensor(1)

		return LossOutput(full_loss=full_loss, values=values, counts=counts)

	@staticmethod
	def _reduce(
		unreduced: torch.Tensor, mask: torch.Tensor, reduction: str
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""returns (scalar_for_loss, stored_value, count) — all on-device tensors, no CPU syncs"""
		if reduction == "sum":
			stored = (unreduced * mask).sum()
			return stored, stored, mask.new_tensor(1)
		elif reduction == "mean":
			count = mask.sum()
			stored = (unreduced * mask).sum()
			scalar = stored / count.clamp(min=1)
			return scalar, stored, count
		elif reduction == "max":
			stored = unreduced.masked_fill(mask == 0, float('-inf')).max()
			return stored, stored, mask.new_tensor(1)
		elif reduction == "min":
			stored = unreduced.masked_fill(mask == 0, float('inf')).min()
			return stored, stored, mask.new_tensor(1)
		elif reduction == "last":
			return unreduced, unreduced, mask.new_tensor(1)
		else:
			raise ValueError(f"unknown reduction: {reduction}")

	# ----- loss methods — all return Dict[str, Tuple[Tensor, Tensor]] -----

	def cel(self, logits, labels, mask):
		labels = labels.masked_fill(~mask, 0)
		unreduced = F.cross_entropy(logits, labels, reduction="none")
		return {"cel": (unreduced, mask.float())}

	def focal_loss(self, logits, labels, mask, alpha=1.0, gamma=2.0):
		labels = labels.masked_fill(~mask, 0)
		cel = F.cross_entropy(logits, labels, reduction="none")
		p_t = torch.exp(-cel)
		return {"focal_loss": (alpha * (1 - p_t) ** gamma * cel, mask.float())}

	def matches(self, logits, labels, mask):
		return {"matches": ((torch.argmax(logits, dim=-1) == labels).float(), mask.float())}

	def probs(self, logits, labels, mask):
		labels_safe = labels.masked_fill(~mask, 0)
		p = torch.gather(torch.softmax(logits, dim=-1), -1, labels_safe.unsqueeze(-1)).squeeze(-1)
		return {"probs": (p, mask.float())}

	def aa_magnitudes(self, aa_magnitudes):
		one = aa_magnitudes.new_tensor(1.0)
		return {
			lbl_2_aa(i).replace("<", "").replace(">", ""): (aa_magnitudes[:, i].mean(), one)
			for i in range(aa_magnitudes.shape[1])
		}

	def per_label_cel(self, logits, labels, mask):
		"""cross-entropy loss per amino acid label"""
		labels_safe = labels.masked_fill(~mask, 0)
		unreduced = F.cross_entropy(logits, labels_safe, reduction="none")
		n_labels = logits.shape[-1]
		return {
			lbl_2_aa(i).replace("<", "").replace(">", ""): (unreduced, ((labels == i) & mask).float())
			for i in range(n_labels)
		}

	def per_label_probs(self, logits, labels, mask):
		"""predicted probability of the true label, per amino acid"""
		labels_safe = labels.masked_fill(~mask, 0)
		p = torch.gather(torch.softmax(logits, dim=-1), -1, labels_safe.unsqueeze(-1)).squeeze(-1)
		n_labels = logits.shape[-1]
		return {
			lbl_2_aa(i).replace("<", "").replace(">", ""): (p, ((labels == i) & mask).float())
			for i in range(n_labels)
		}

	def per_label_matches(self, logits, labels, mask):
		"""accuracy per amino acid label"""
		correct = (torch.argmax(logits, dim=-1) == labels).float()
		n_labels = logits.shape[-1]
		return {
			lbl_2_aa(i).replace("<", "").replace(">", ""): (correct, ((labels == i) & mask).float())
			for i in range(n_labels)
		}
