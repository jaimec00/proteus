from proteus.losses.training_loss import LossFnCfg, LossTermCfg, Reductions, LossFnNames
from proteus.model import OutputNames

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class SeqCEL(LossFnCfg):
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

@dataclass
class SeqCELPerLabel(LossFnCfg):
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
class SeqCEL_AND_AAMags(SeqCEL, AAMags):
    pass

@dataclass
class SeqCEL_AND_SeqCELPerLabel(SeqCEL, SeqCELPerLabel):
    pass

@dataclass
class SeqCEL_AND_SeqCELPerLabel_AND_AAMags(SeqCEL, SeqCELPerLabel, AAMags):
    pass

@dataclass
class SeqLenBinnedAcc(LossFnCfg):
    seq_len_binned_acc: LossTermCfg = field(default_factory=lambda: LossTermCfg(
        fn=LossFnNames.BINNED_MATCHES,
        inputs=[OutputNames.SEQ_LOGITS, OutputNames.SEQ_LABELS, OutputNames.LOSS_MASK, OutputNames.CU_SEQLENS],
        kwargs={"bins": [[0, 128], [128, 256], [256, 512], [512, 768], [768, 1024]]},
        reductions=[Reductions.MEAN],
    ))

@dataclass
class SeqCEL_AND_SeqCELPerLabel_AND_AAMags_AND_Outlier(SeqCEL, SeqCELPerLabel, AAMags, Outliers):
    pass

@dataclass
class SeqCEL_AND_AAMags_AND_SeqLenBinnedAcc(SeqCEL, AAMags, SeqLenBinnedAcc):
    pass

def register_losses():
    cs = ConfigStore.instance()
    cs.store(name="cel", node=SeqCEL, group="losses")
    cs.store(name="cel_aa", node=SeqCEL_AND_AAMags, group="losses")
    cs.store(name="cel_perlbl", node=SeqCEL_AND_SeqCELPerLabel, group="losses")
    cs.store(name="cel_perlbl_aa", node=SeqCEL_AND_SeqCELPerLabel_AND_AAMags, group="losses")
    cs.store(name="cel_perlbl_aa_outliers", node=SeqCEL_AND_SeqCELPerLabel_AND_AAMags_AND_Outlier, group="losses")
    cs.store(name="cel_aa_seqbinacc", node=SeqCEL_AND_AAMags_AND_SeqLenBinnedAcc, group="losses")

