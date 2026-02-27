
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from proteus.model import OutputNames
from proteus.model.base import Base
from proteus.data.data_utils import DataBatch
from proteus.model.model_utils.mlp import SeqProjectionHead, SeqProjectionHeadCfg
from proteus.model.transformer.transformer import TransformerModel, TransformerModelCfg
from proteus.model.tokenizer.wave_func_tokenizer.wf_tokenizer import WaveFunctionTokenizerCfg, WaveFunctionTokenizer
from proteus.utils.noise_utils import NoiseSchedule, NoiseScheduleCfg
from proteus.types import Float, T
    
@dataclass
class proteusAICfg:
    d_model: int
    noise_schedule: NoiseScheduleCfg = field(default_factory=NoiseScheduleCfg)
    tokenizer: WaveFunctionTokenizerCfg = field(default_factory=WaveFunctionTokenizerCfg)
    transformer: TransformerModelCfg = field(default_factory=TransformerModelCfg)
    seq_proj_head: SeqProjectionHeadCfg = field(default_factory=SeqProjectionHeadCfg)
    model_cls: str = "proteus.model.composed.inverse_folding.proteusAI.proteusAI"


class proteusAI(Base):
    def __init__(self, cfg: proteusAICfg):
        super().__init__()

        self.noise_schedule = NoiseSchedule(cfg.noise_schedule)
        self.tokenizer = WaveFunctionTokenizer(cfg.tokenizer)
        self.transformer = TransformerModel(cfg.transformer)
        self.seq_proj_head = SeqProjectionHead(cfg.seq_proj_head)

    def forward(self, data_batch: DataBatch) -> Float[T, "BL AA"]:

        aas, is_masked, t = self.noise_schedule.corrupt_labels(
            labels=data_batch.labels,
            sample_idx=data_batch.sample_idx,
            cu_seqlens=data_batch.cu_seqlens,
        )

        wf, wf_raw = self.tokenizer(
            data_batch.coords_ca, 
            data_batch.coords_cb_unit,
            aas,
            data_batch.cu_seqlens,
            return_wf_raw=True,
        )

        latent = self.transformer(
            wf, wf,
            data_batch.cu_seqlens, data_batch.cu_seqlens,
            data_batch.max_seqlen, data_batch.max_seqlen,
        )

        seq_logits = self.seq_proj_head(latent)

        return {
            OutputNames.SEQ_LOGITS: seq_logits,
            OutputNames.SEQ_LABELS: data_batch.labels,
            OutputNames.LOSS_MASK: data_batch.loss_mask,
            OutputNames.IS_MASKED: is_masked,
            OutputNames.AA_MAGNITUDES: self.tokenizer.aa_magnitudes,
            OutputNames.WF_RAW: wf_raw,
            OutputNames.NO_MASK: torch.ones_like(data_batch.loss_mask),
            OutputNames.CU_SEQLENS: data_batch.cu_seqlens,
            OutputNames.TIMESTEP: t,
        }
