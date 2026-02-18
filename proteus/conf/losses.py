from proteus.losses.training_loss import LossFnCfg, LossTermCfg
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class SeqCelLossFn(LossFnCfg):
	seq_cel: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="cel", inputs=["seq_logits", "seq_labels", "loss_mask"],
		weight=1.0, reductions=["mean"],
	))
	seq_probs: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="probs", inputs=["seq_logits", "seq_labels", "loss_mask"],
		reductions=["mean"],
	))
	seq_acc: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="matches", inputs=["seq_logits", "seq_labels", "loss_mask"],
		reductions=["mean"],
	))
	aa_magnitudes: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="aa_magnitudes", inputs=["aa_magnitudes"],
		reductions=["last"],
	))
	per_label_cel: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="per_label_cel", inputs=["seq_logits", "seq_labels", "loss_mask"],
		reductions=["mean"],
	))
	per_label_probs: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="per_label_probs", inputs=["seq_logits", "seq_labels", "loss_mask"],
		reductions=["mean"],
	))
	per_label_acc: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="per_label_matches", inputs=["seq_logits", "seq_labels", "loss_mask"],
		reductions=["mean"],
	))

@dataclass
class SeqFocalLossFn(LossFnCfg):
	seq_cel: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="cel", inputs=["seq_logits", "seq_labels", "loss_mask"],
		reductions=["sum", "mean"],
	))
	seq_focal_loss: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="focal_loss", inputs=["seq_logits", "seq_labels", "loss_mask"],
		kwargs={"alpha": 1.0, "gamma": 2.0},
		weight=1.0, reductions=["sum", "mean"],
	))
	seq_probs: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="probs", inputs=["seq_logits", "seq_labels", "loss_mask"],
		reductions=["mean"],
	))
	seq_acc: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="matches", inputs=["seq_logits", "seq_labels", "loss_mask"],
		reductions=["mean"],
	))
	per_label_cel: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="per_label_cel", inputs=["seq_logits", "seq_labels", "loss_mask"],
		reductions=["mean"],
	))
	per_label_probs: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="per_label_probs", inputs=["seq_logits", "seq_labels", "loss_mask"],
		reductions=["mean"],
	))
	per_label_acc: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="per_label_matches", inputs=["seq_logits", "seq_labels", "loss_mask"],
		reductions=["mean"],
	))

def register_losses():
	cs = ConfigStore.instance()
	cs.store(name="cel_loss", node=SeqCelLossFn, group="losses")
	cs.store(name="focal_loss", node=SeqFocalLossFn, group="losses")
