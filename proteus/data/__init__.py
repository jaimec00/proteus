from enum import StrEnum

from .data_loader import DataHolder, DataHolderCfg
from .data_utils import DataBatch


class DataPath(StrEnum):
	"""relative paths shared between the data pipeline and the dataloader"""
	INDEX = "shards/index.parquet"
	SHARDS = "shards"


__all__ = [
    "DataHolder",
    "DataHolderCfg",
    "DataBatch",
    "DataPath",
]