from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class FoldSeekCfg:
	input_path: str = "${..local_path}/raw_cif" # where the raw mmCIF's get stored
	db_path: str = "${..local_path}/db" # where the foldseek dbs live

	# shared
	verbosity: int = 1

	# createdb
	distance_threshold: float = 8.0
	mask_bfactor_threshold: float = 0.0
	coord_store_mode: int = 2 # 1: CA as float, 2: CA as difference (uint16)
	chain_name_mode: int = 0 # 0: auto, 1: always add chain to name

	# clustering
	tmscore_thresholds: list[float] = field(default_factory=lambda: [0.7])
	tmscore_threshold_mode: int = 0 # 0: alignment length, 1: query length, 2: target length
	lddt_threshold: float = 0.0 # additional lddt filter [0.0-1.0], 0 = disabled
	coverage: float = 0.8
	cov_mode: int = 0 # 0: query+target, 1: target, 2: query
	cluster_mode: int = 0 # 0: set-cover, 1: connected component, 2/3: greedy by length
	sensitivity: float = 7.5
	e_value: float = 10.0
	max_seqs: int = 1000 # max prefilter results per query (affects sensitivity)
	min_aln_len: int = 0 # minimum alignment length
	max_seq_len: int = 65535
	split: int = 0 # 0 = auto split based on available memory
	split_memory_limit: str = "0" # max memory per split, e.g. "10G". 0 = all available
	cluster_steps: int = 3 # cascaded clustering steps
	cluster_reassign: bool = True

def register_foldseek():
	cs = ConfigStore.instance()
	cs.store("default", FoldSeekCfg, group="foldseek")
