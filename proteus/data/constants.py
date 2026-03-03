from enum import StrEnum, auto


class DataPath(StrEnum):
	"""relative paths shared between the data pipeline and the dataloader"""
	INDEX = "shards/index.parquet"
	SHARDS = "shards"


class DataIndexCols(StrEnum):
	"""column names in the index parquet"""
	PDB = auto()
	CHAIN = auto()
	SOURCE = auto()
	SHARD_ID = auto()
	OFFSET = auto()
	SIZE = auto()
	RESOLUTION = auto()
	METHOD = auto()
	DEPOSIT_DATE = auto()
	CLUSTER_ID = auto()
