from dataclasses import dataclass, field
from hydra.conf import RunDir, SweepDir, HydraConf
from hydra.core.config_store import ConfigStore

@dataclass
class ExperimentalDataDownloadCfg:
	# filtering
	methods: list = field(default_factory=lambda: ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"])
	max_resolution: float = 3.5
	max_entries: int = 32  # -1 = all, testing for now

	# pipeline
	semaphore_limit: int = 32
	chunk_size: int = 1000
	s3_path: str = "${..s3_path}"
	local_path: str = "${foldseek.input_path}"

@dataclass
class FoldSeekCfg:
	input_path: str = "${..local_path}/raw_cif" # where the raw mmCIF's get stored
	db_path: str = "${..local_path}/db" # where the foldseek dbs live

	# shared
	verbosity: int = 3

	# createdb
	distance_threshold: float = 8.0
	mask_bfactor_threshold: float = 0.0
	coord_store_mode: int = 2 # 1: CA as float, 2: CA as difference (uint16)
	chain_name_mode: int = 0 # 0: auto, 1: always add chain to name

	# clustering
	tmscore_threshold: float = 0.7
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
	remove_tmp_files: bool = False


@dataclass
class DataPipelineCfg:
	experimental_dl: ExperimentalDataDownloadCfg = field(default_factory=ExperimentalDataDownloadCfg)
	foldseek: FoldSeekCfg = field(default_factory=FoldSeekCfg)
	s3_path: str = "s3://proteus-data-bucket"
	local_path: str = "/home/ubuntu/proteus/data/tmp"

@dataclass
class DownloadHydra(HydraConf):
	run: RunDir = field(default_factory=lambda: RunDir("/home/ubuntu/proteus/data/download/logs/${now:%Y-%m-%d}/${now:%H-%M-%S}"))

def register_download_configs():
	cs = ConfigStore.instance()
	cs.store("default", DataPipelineCfg)
	cs.store("default", ExperimentalDataDownloadCfg, group="experimental_dl")
	cs.store("default", FoldSeekCfg, group="foldseek")
	cs.store("config", DownloadHydra, group="hydra")
