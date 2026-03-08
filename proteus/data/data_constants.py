import enum


class IndexCol(enum.StrEnum):
	"""parquet index column names"""
	PDB = enum.auto()
	CHAIN = enum.auto()
	SOURCE = enum.auto()
	SHARD_ID = enum.auto()
	OFFSET = enum.auto()
	SIZE = enum.auto()
	RESOLUTION = enum.auto()
	METHOD = enum.auto()
	DEPOSIT_DATE = enum.auto()
	CLUSTER_ID = enum.auto()


class DataSource(enum.StrEnum):
	"""source labels for downloaded structures"""
	EXPERIMENTAL = enum.auto()
	PDB_REDO = enum.auto()
	RCSB = enum.auto()


class ChainKey(enum.StrEnum):
	"""per-chain data dict keys"""
	SEQUENCE = enum.auto()
	COORDS = enum.auto()
	ATOM_MASK = enum.auto()
	BFACTOR = enum.auto()
	CIF = enum.auto()


class DataPath(enum.StrEnum):
	"""relative paths shared between the data pipeline and the dataloader"""
	INDEX = "shards/index.parquet"
	SHARDS = "shards"


class ProteinKey(enum.StrEnum):
	"""top-level parsed protein data dict keys"""
	CHAINS = enum.auto()
	ASSEMBLIES = enum.auto()
	ASMB_XFORMS = enum.auto()
	RESOLUTION = enum.auto()
	METHOD = enum.auto()
	DEPOSIT_DATE = enum.auto()
	SOURCE = enum.auto()


class NpzKey(enum.StrEnum):
	"""special npz archive keys"""
	META = enum.auto()

class ExpMethods(enum.StrEnum):
	XRAY = "X-RAY DIFFRACTION"
	CRYO_EM = "ELECTRON MICROSCOPY"