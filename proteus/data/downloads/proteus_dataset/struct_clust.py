import shutil
import subprocess

from pathlib import Path

from proteus.types import Dict
from proteus.data.downloads.proteus_dataset.conf.foldseek import FoldSeekCfg


class FoldSeek:
	def __init__(self, cfg: FoldSeekCfg):
		self.input_path = Path(cfg.input_path)
		self.db_path = Path(cfg.db_path)
		self.raw_db_path = self.db_path / "raw_db"
		self.cluster_db_path = self.db_path / "cluster_db"
		self.cluster_tsv_path = self.db_path / "clusters.tsv"
		self.tmp_dir = self.db_path / "tmp"

		# shared
		self.verbosity = str(cfg.verbosity)

		# createdb
		self.distance_threshold = str(cfg.distance_threshold)
		self.mask_bfactor_threshold = str(cfg.mask_bfactor_threshold)
		self.coord_store_mode = str(cfg.coord_store_mode)
		self.chain_name_mode = str(cfg.chain_name_mode)

		# clustering
		self.tmscore_threshold = str(cfg.tmscore_threshold)
		self.tmscore_threshold_mode = str(cfg.tmscore_threshold_mode)
		self.lddt_threshold = str(cfg.lddt_threshold)
		self.coverage = str(cfg.coverage)
		self.cov_mode = str(cfg.cov_mode)
		self.cluster_mode = str(cfg.cluster_mode)
		self.sensitivity = str(cfg.sensitivity)
		self.e_value = str(cfg.e_value)
		self.max_seqs = str(cfg.max_seqs)
		self.min_aln_len = str(cfg.min_aln_len)
		self.max_seq_len = str(cfg.max_seq_len)
		self.split = str(cfg.split)
		self.split_memory_limit = cfg.split_memory_limit
		self.cluster_steps = str(cfg.cluster_steps)
		self.cluster_reassign = str(int(cfg.cluster_reassign))

	def create_db(self):
		self.db_path.mkdir(parents=True, exist_ok=True)
		cmd = [
			"foldseek", "createdb",
			str(self.input_path),
			str(self.raw_db_path),
			"--db-extraction-mode", "0",              # chain extraction
			"--input-format", "2",                    # mmCIF
			"--distance-threshold", self.distance_threshold,
			"--mask-bfactor-threshold", self.mask_bfactor_threshold,
			"--coord-store-mode", self.coord_store_mode,
			"--chain-name-mode", self.chain_name_mode,
			"-v", self.verbosity,
		]

		db_output = subprocess.run(cmd)

		# delete the raw cifs after we create the db
		db_output.check_returncode()
		shutil.rmtree(self.input_path)

	def run_cluster(self):
		"""run foldseek cluster, clean up tmp dir. does not parse results."""
		raw_db = str(self.raw_db_path)
		cluster_db = str(self.cluster_db_path)
		tmp_dir = str(self.tmp_dir)

		cmd = [
			"foldseek", "cluster",
			raw_db, cluster_db, tmp_dir,
			"--alignment-type", "0",                    # 3Di only (structure, no sequence)
			"--min-seq-id", "0.0",                      # no sequence identity filter
			"--tmscore-threshold", self.tmscore_threshold,
			"--tmscore-threshold-mode", self.tmscore_threshold_mode,
			"--lddt-threshold", self.lddt_threshold,
			"-c", self.coverage,
			"--cov-mode", self.cov_mode,
			"--cluster-mode", self.cluster_mode,
			"-e", self.e_value,
			"-s", self.sensitivity,
			"--max-seqs", self.max_seqs,
			"--min-aln-len", self.min_aln_len,
			"--max-seq-len", self.max_seq_len,
			"--split", self.split,
			"--split-memory-limit", self.split_memory_limit,
			"--cluster-steps", self.cluster_steps,
			"--cluster-reassign", self.cluster_reassign,
			"--remove-tmp-files", "1",
			"-v", self.verbosity,
		]

		result = subprocess.run(cmd)
		result.check_returncode()
		if self.tmp_dir.exists():
			shutil.rmtree(self.tmp_dir)

	def parse_clusters(self) -> Dict[str, str]:
		"""run createtsv, clean up db files, return {chain_id: cluster_representative}.
		leaves the TSV on disk so it can serve as a resume signal."""

		raw_db, cluster_db, tsv_path = str(self.raw_db_path), str(self.cluster_db_path), str(self.cluster_tsv_path)

		result = subprocess.run([
			"foldseek", "createtsv",
			raw_db, raw_db, cluster_db, tsv_path,
			"-v", self.verbosity,
		])
		result.check_returncode()

		# clean up db files now that we have the tsv
		for f in self.db_path.glob(self.raw_db_path.name + "*"):
			f.unlink()
		for f in self.db_path.glob(self.cluster_db_path.name + "*"):
			f.unlink()

		return self.load_clusters()

	def load_clusters(self) -> Dict[str, str]:
		"""read existing clusters.tsv into {chain_id: cluster_representative}"""
		clusters = {}
		for line in self.cluster_tsv_path.read_text().splitlines():
			rep, member = line.split("\t")
			clusters[member] = rep
		return clusters

	def cleanup_tsv(self):
		"""remove the clusters TSV after index has been safely uploaded"""
		if self.cluster_tsv_path.exists():
			self.cluster_tsv_path.unlink()
