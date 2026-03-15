import pytest

from proteus.data.data_constants import ClusteringMethod, ClusterInputType
from proteus.data.downloads.proteus_dataset.cluster import MMSeqs


pytestmark = pytest.mark.cpu


class TestMMSeqsClassVars:
	def test_method(self):
		assert MMSeqs.method == ClusteringMethod.MMSEQS

	def test_required_inputs(self):
		assert MMSeqs.required_inputs == {ClusterInputType.FASTA}


class TestMMSeqsProperties:
	def test_thresholds(self, mmseqs_instance):
		assert mmseqs_instance.thresholds == [0.3]

	def test_cluster_tsv_path(self, mmseqs_instance):
		path = mmseqs_instance.cluster_tsv_path(0.3)
		assert path == mmseqs_instance.db_path / "clusters_0.3.tsv"


class TestMMSeqsCreateDb:
	def test_has_raw_db(self, mmseqs_with_db):
		assert mmseqs_with_db.has_raw_db()

	def test_combined_fasta_cleaned_up(self, mmseqs_with_db):
		combined = mmseqs_with_db.db_path / "combined.fasta"
		assert not combined.exists()

	def test_input_not_deleted(self, mmseqs_with_db):
		assert mmseqs_with_db.input_path.exists()


class TestMMSeqsCluster:
	def test_has_cluster_db(self, mmseqs_clustered):
		assert mmseqs_clustered.has_cluster_db(0.3)


class TestMMSeqsParse:
	def test_returns_dict(self, mmseqs_parsed):
		_, clusters = mmseqs_parsed
		assert isinstance(clusters, dict)
		assert len(clusters) > 0

	def test_all_chains_present(self, mmseqs_parsed):
		_, clusters = mmseqs_parsed
		expected = {"pdb1_A", "pdb1_B", "pdb2_A", "pdb3_A"}
		assert set(clusters.keys()) == expected

	def test_values_are_valid_chain_ids(self, mmseqs_parsed):
		_, clusters = mmseqs_parsed
		all_ids = set(clusters.keys())
		for rep in clusters.values():
			assert rep in all_ids

	def test_identical_seqs_cluster_together(self, mmseqs_parsed):
		_, clusters = mmseqs_parsed
		# pdb1_A and pdb1_B have identical sequences
		assert clusters["pdb1_A"] == clusters["pdb1_B"]

	def test_cluster_db_cleaned_up(self, mmseqs_parsed):
		ms, _ = mmseqs_parsed
		assert not ms.has_cluster_db(0.3)

	def test_tsv_exists(self, mmseqs_parsed):
		ms, _ = mmseqs_parsed
		assert ms.cluster_tsv_path(0.3).exists()


class TestMMSeqsLoadClusters:
	def test_matches_parse_result(self, mmseqs_parsed):
		ms, parsed_clusters = mmseqs_parsed
		loaded = ms.load_clusters(0.3)
		assert loaded == parsed_clusters


class TestMMSeqsCleanup:
	@pytest.fixture
	def cleanup_mmseqs(self, mmseqs_parsed):
		ms, _ = mmseqs_parsed
		return ms

	def test_cleanup_raw_db(self, cleanup_mmseqs):
		cleanup_mmseqs.cleanup_raw_db()
		assert not cleanup_mmseqs.has_raw_db()

	def test_cleanup_tsvs(self, cleanup_mmseqs):
		cleanup_mmseqs.cleanup_tsvs()
		assert not cleanup_mmseqs.cluster_tsv_path(0.3).exists()
