from proteus.data.data_loader import DataHolderCfg
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from cloudpathlib import S3Path

@dataclass
class DefaultData(DataHolderCfg):
    s3_bucket: S3Path = field(default_factory=S3Path("s3://proteus-data-bucket"))
    num_train: int = -1
    num_val: int = -1
    num_test: int = -1
    batch_tokens: int = 65536
    min_seq_size: int = 16
    max_seq_size: int = 512
    max_resolution: float = 3.5
    min_plddt: float = 0.70
    asymmetric_units_only: bool = False
    num_workers: int = 16
    prefetch_factor: int = 8
    rng_seed: int = 6000
    buffer_size: int = 64

@dataclass
class XSmallSeqData(DefaultData):
    max_seq_size: int = 128
@dataclass
class SmallSeqData(DefaultData):
    max_seq_size: int = 512
@dataclass
class MediumSeqData(DefaultData):
    max_seq_size: int = 1024
@dataclass
class LargeSeqData(DefaultData):
    max_seq_size: int = 8192
@dataclass
class XLargeSeqData(DefaultData):
    max_seq_size: int = 8192

def register_data():
    cs = ConfigStore.instance()
    cs.store(name="default", node=DefaultData, group="data")
    cs.store(name="extra_small_seq", node=XSmallSeqData, group="data")
    cs.store(name="small_seq", node=SmallSeqData, group="data")
    cs.store(name="medium_seq", node=MediumSeqData, group="data")
    cs.store(name="large_seq", node=LargeSeqData, group="data")
    cs.store(name="extra_large_seq", node=XLargeSeqData, group="data")
