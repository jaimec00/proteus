from proteus.losses.training_loss import LossFnCfg, LossTermCfg, MaskCfg, Reductions, LossFnNames
from proteus.model import OutputNames

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class SeqCEL(LossFnCfg):
    seq_cel: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.CEL, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
        weight=1.0, reductions=[Reductions.MEAN, Reductions.SUM],
    ))
    seq_probs: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.PROBS, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
        reductions=[Reductions.MEAN],
    ))
    seq_acc: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.MATCHES, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
        reductions=[Reductions.MEAN],
    ))

def _per_label():
    return MaskCfg(
        fn=LossFnNames.PER_LABEL_MASKS,
        inputs=[OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK, OutputNames.SEQ_LOGITS],
    )

@dataclass
class SeqCELPerLabel(LossFnCfg):
    seq_cel: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.CEL, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
        masks=[_per_label()],
        weight=1.0, reductions=[Reductions.MEAN, Reductions.SUM],
    ))
    seq_probs: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.PROBS, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
        masks=[_per_label()],
        reductions=[Reductions.MEAN],
    ))
    seq_acc: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.MATCHES, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
        masks=[_per_label()],
        reductions=[Reductions.MEAN],
    ))

def _seqlen_bins():
    return MaskCfg(
        fn=LossFnNames.BINNED_MASKS,
        inputs=[OutputNames.CU_SEQLENS, OutputNames.LOSS_MASK],
        kwargs={"bins": [[0, 128], [128, 256], [256, 512], [512, 768], [768, 1024]]},
    )

@dataclass
class SeqLenBinned(LossFnCfg):
    seq_cel: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.CEL, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
        masks=[_seqlen_bins()],
        weight=1.0, reductions=[Reductions.MEAN, Reductions.SUM],
    ))
    seq_probs: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.PROBS, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
        masks=[_seqlen_bins()],
        reductions=[Reductions.MEAN],
    ))
    seq_acc: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.MATCHES, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
        masks=[_seqlen_bins()],
        reductions=[Reductions.MEAN],
    ))

# both per-label and seqlen-binned masks on each term
@dataclass
class SeqCELPerLabelAndSeqLenBinned(LossFnCfg):
    seq_cel: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.CEL, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
        masks=[_per_label(), _seqlen_bins()],
        weight=1.0, reductions=[Reductions.MEAN, Reductions.SUM],
    ))
    seq_probs: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.PROBS, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
        masks=[_per_label(), _seqlen_bins()],
        reductions=[Reductions.MEAN],
    ))
    seq_acc: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.MATCHES, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
        masks=[_per_label(), _seqlen_bins()],
        reductions=[Reductions.MEAN],
    ))

def _is_masked():
    return MaskCfg(
        fn=LossFnNames.PASS_MASK,
        inputs=[OutputNames.IS_MASKED, OutputNames.LOSS_MASK],
    )
@dataclass
class SeqCELMasked(LossFnCfg):
    seq_cel: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.CEL, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
        masks=[_is_masked()],
        weight=1.0, reductions=[Reductions.MEAN, Reductions.SUM],
    ))
    seq_probs: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.PROBS, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
        masks=[_is_masked()],
        reductions=[Reductions.MEAN],
    ))
    seq_acc: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.MATCHES, inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK],
        masks=[_is_masked()],
        reductions=[Reductions.MEAN],
    ))


@dataclass
class AAMags(LossFnCfg):
    aa_magnitudes: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.AA_MAGNITUDES, inputs=[OutputNames.AA_MAGNITUDES],
        reductions=[Reductions.LAST],
    ))

@dataclass
class Outliers(LossFnCfg):
    # couple of metrics for seeing logits and wf output outlier
    wf_raw: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.IDENTITY, inputs=[OutputNames.WF_RAW, OutputNames.NO_MASK],
        reductions=[Reductions.MAX, Reductions.MIN],
    ))
    logits: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.IDENTITY, inputs=[OutputNames.SEQ_LOGITS, OutputNames.LOSS_MASK],
        reductions=[Reductions.MAX, Reductions.MIN],
    ))



@dataclass
class SeqCEL_AND_AAMags(SeqCEL, AAMags): pass
@dataclass
class SeqCELPerLabel_AND_AAMags(SeqCELPerLabel, AAMags): pass
@dataclass
class SeqCELPerLabel_AND_AAMags_AND_Outlier(SeqCELPerLabel, AAMags, Outliers): pass
@dataclass
class SeqLenBinned_AND_AAMags(SeqLenBinned, AAMags): pass
@dataclass
class SeqCELPerLabelAndSeqLenBinned_AND_AAMags(SeqCELPerLabelAndSeqLenBinned, AAMags): pass
@dataclass
class SeqCELMasked_AND_AAMags(SeqCELMasked, AAMags): pass

def register_losses():
    cs = ConfigStore.instance()
    cs.store(name="cel", node=SeqCEL, group="losses")
    cs.store(name="cel_aa", node=SeqCEL_AND_AAMags, group="losses")
    cs.store(name="cel_perlbl", node=SeqCELPerLabel, group="losses")
    cs.store(name="cel_perlbl_aa", node=SeqCELPerLabel_AND_AAMags, group="losses")
    cs.store(name="cel_perlbl_aa_outliers", node=SeqCELPerLabel_AND_AAMags_AND_Outlier, group="losses")
    cs.store(name="cel_aa_seqbinacc", node=SeqLenBinned_AND_AAMags, group="losses")
    cs.store(name="cel_perlbl_aa_seqbinacc", node=SeqCELPerLabelAndSeqLenBinned_AND_AAMags, group="losses")
    cs.store(name="cel_masked", node=SeqCELMasked, group="losses")
    cs.store(name="cel_masked_aa", node=SeqCELMasked_AND_AAMags, group="losses")
