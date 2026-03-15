import shutil
import subprocess

from pathlib import Path

from proteus.types import Dict
from proteus.data.data_constants import ClusteringMethod, ClusterInputType
from proteus.data.downloads.proteus_dataset.conf.cluster import MMSeqsCfg
from proteus.data.downloads.proteus_dataset.cluster.base import ClusterMethodBase


class MMSeqs(ClusterMethodBase):
	method = ClusteringMethod.MMSEQS
	required_inputs = {ClusterInputType.FASTA}

	def __init__(self, cfg: MMSeqsCfg):
		self.input_path = Path(cfg.input_path)
		self.db_path = Path(cfg.db_path)
		self.raw_db_path = self.db_path / "raw_db"
		self.tmp_dir = self.db_path / "tmp"

		self.seq_id_thresholds = cfg.seq_id_thresholds
		assert all(round(t, 2) == t for t in self.seq_id_thresholds), \
			"seq_id thresholds must have at most 2 decimal places"

		# shared
		self.verbosity = str(cfg.verbosity)

		# clustering
		self.coverage = str(cfg.coverage)
		self.cov_mode = str(cfg.cov_mode)
		self.cluster_mode = str(cfg.cluster_mode)
		self.sensitivity = str(cfg.sensitivity)
		self.e_value = str(cfg.e_value)
		self.max_seqs = str(cfg.max_seqs)
		self.seq_id_mode = str(cfg.seq_id_mode)
		self.min_aln_len = str(cfg.min_aln_len)
		self.split = str(cfg.split)
		self.split_memory_limit = cfg.split_memory_limit

	@property
	def thresholds(self) -> list[float]:
		return self.seq_id_thresholds

	def cluster_db_path(self, threshold: float) -> Path:
		return self.db_path / f"cluster_db_{threshold}"

	def cluster_tsv_path(self, threshold: float) -> Path:
		return self.db_path / f"clusters_{threshold}.tsv"

	def has_raw_db(self) -> bool:
		return any(self.db_path.glob(self.raw_db_path.name + "*"))

	def has_cluster_db(self, threshold: float) -> bool:
		return any(self.db_path.glob(self.cluster_db_path(threshold).name + "*"))

	def create_db(self):
		self.db_path.mkdir(parents=True, exist_ok=True)

		# collect all fasta files from input directory
		fasta_files = sorted(self.input_path.glob("*.fasta"))
		assert len(fasta_files) > 0, f"no .fasta files found in {self.input_path}"

		# concatenate all per-chain fastas into a single file for createdb
		combined_fasta = self.db_path / "combined.fasta"
		with open(combined_fasta, "w") as out:
			for fasta in fasta_files:
				out.write(fasta.read_text())

		cmd = [
			"mmseqs", "createdb",
			str(combined_fasta),
			str(self.raw_db_path),
			"-v", self.verbosity,
		]

		result = subprocess.run(cmd)
		result.check_returncode()
		combined_fasta.unlink()

	def run_cluster(self, threshold: float):
		"""run mmseqs cluster for a single threshold. does not parse results."""
		self.tmp_dir.mkdir(parents=True, exist_ok=True)
		raw_db = str(self.raw_db_path)
		cluster_db = str(self.cluster_db_path(threshold))
		tmp_dir = str(self.tmp_dir)

		cmd = [
			"mmseqs", "cluster",
			raw_db, cluster_db, tmp_dir,
			"--min-seq-id", str(threshold),
			"-c", self.coverage,
			"--cov-mode", self.cov_mode,
			"--cluster-mode", self.cluster_mode,
			"-s", self.sensitivity,
			"-e", self.e_value,
			"--max-seqs", self.max_seqs,
			"--seq-id-mode", self.seq_id_mode,
			"--min-aln-len", self.min_aln_len,
			"--split", self.split,
			"--split-memory-limit", self.split_memory_limit,
			"-v", self.verbosity,
		]

		result = subprocess.run(cmd)
		result.check_returncode()
		if self.tmp_dir.exists():
			shutil.rmtree(self.tmp_dir)

	def parse_clusters(self, threshold: float) -> Dict[str, str]:
		"""run createtsv for a single threshold, clean up that threshold's cluster db,
		return {chain_id: cluster_representative}. leaves the TSV on disk as a resume signal."""

		raw_db = str(self.raw_db_path)
		cluster_db = str(self.cluster_db_path(threshold))
		tsv_path = str(self.cluster_tsv_path(threshold))

		result = subprocess.run([
			"mmseqs", "createtsv",
			raw_db, raw_db, cluster_db, tsv_path,
			"-v", self.verbosity,
		])
		result.check_returncode()

		# clean up this threshold's cluster db files (not the raw db)
		for f in self.db_path.glob(self.cluster_db_path(threshold).name + "*"):
			f.unlink()

		return self.load_clusters(threshold)

	def load_clusters(self, threshold: float) -> Dict[str, str]:
		"""read existing clusters TSV into {chain_id: cluster_representative}"""
		clusters = {}
		for line in self.cluster_tsv_path(threshold).read_text().splitlines():
			rep, member = line.split("\t")
			clusters[member] = rep
		return clusters

	def cleanup_raw_db(self):
		"""remove raw mmseqs db files after all thresholds are done"""
		for f in self.db_path.glob(self.raw_db_path.name + "*"):
			f.unlink()

	def cleanup_tsvs(self):
		"""remove all cluster TSVs after index has been safely uploaded"""
		for threshold in self.seq_id_thresholds:
			tsv = self.cluster_tsv_path(threshold)
			if tsv.exists():
				tsv.unlink()
