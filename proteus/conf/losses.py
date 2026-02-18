from proteus.losses.training_loss import LossFnCfg, LossTermCfg, Reductions
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class SeqCelLossFn(LossFnCfg):
	seq_cel: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="cel", inputs=["seq_logits", "seq_labels", "loss_mask"],
		weight=1.0, reductions=[Reductions.SUM, Reductions.MEAN],
	))
	seq_probs: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="probs", inputs=["seq_logits", "seq_labels", "loss_mask"],
		reductions=[Reductions.MEAN],
	))
	seq_acc: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="matches", inputs=["seq_logits", "seq_labels", "loss_mask"],
		reductions=[Reductions.MEAN],
	))
	aa_magnitudes: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="aa_magnitudes", inputs=["aa_magnitudes"],
		reductions=[Reductions.LAST],
	))
	per_label_cel: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="per_label_cel", inputs=["seq_logits", "seq_labels", "loss_mask"],
		reductions=[Reductions.MEAN],
	))
	per_label_probs: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="per_label_probs", inputs=["seq_logits", "seq_labels", "loss_mask"],
		reductions=[Reductions.MEAN],
	))
	per_label_acc: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn="per_label_matches", inputs=["seq_logits", "seq_labels", "loss_mask"],
		reductions=[Reductions.MEAN],
	))

def register_losses():
	cs = ConfigStore.instance()
	cs.store(name="cel_loss", node=SeqCelLossFn, group="losses")
