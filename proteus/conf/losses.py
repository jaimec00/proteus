from proteus.losses.training_loss import LossFnCfg, LossTermCfg, Reductions, LossFnNames
from proteus.model import OutputNames

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class SeqCelLossFn(LossFnCfg):
	seq_cel: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn=LossFnNames.CEL, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
		weight=1.0, reductions=[Reductions.SUM, Reductions.MEAN],
	))
	seq_probs: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn=LossFnNames.PROBS, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
		reductions=[Reductions.MEAN],
	))
	seq_acc: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn=LossFnNames.MATCHES, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
		reductions=[Reductions.MEAN],
	))
	aa_magnitudes: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn=LossFnNames.AA_MAGNITUDES, inputs=[OutputNames.AA_MAGNITUDES],
		reductions=[Reductions.LAST],
	))
	per_label_cel: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn=LossFnNames.PER_LABEL_CEL, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
		reductions=[Reductions.MEAN],
	))
	per_label_probs: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn=LossFnNames.PER_LABEL_PROBS, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
		reductions=[Reductions.MEAN],
	))
	per_label_acc: LossTermCfg = field(default_factory=lambda: LossTermCfg(
		fn=LossFnNames.PER_LABEL_MATCHES, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
		reductions=[Reductions.MEAN],
	))

def register_losses():
	cs = ConfigStore.instance()
	cs.store(name="cel_loss", node=SeqCelLossFn, group="losses")
