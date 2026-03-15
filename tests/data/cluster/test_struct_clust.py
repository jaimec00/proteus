import pytest

from proteus.data.data_constants import ClusteringMethod, ClusterInputType
from proteus.data.downloads.proteus_dataset.cluster import FoldSeek


pytestmark = pytest.mark.cpu


class TestFoldSeekClassVars:
	def test_method(self):
		assert FoldSeek.method == ClusteringMethod.FOLDSEEK

	def test_required_inputs(self):
		assert FoldSeek.required_inputs == {ClusterInputType.MMCIF}


class TestFoldSeekProperties:
	def test_thresholds(self, foldseek_instance):
		assert foldseek_instance.thresholds == [0.7]

	def test_cluster_tsv_path(self, foldseek_instance):
		path = foldseek_instance.cluster_tsv_path(0.7)
		assert path == foldseek_instance.db_path / "clusters_0.7.tsv"


class TestFoldSeekCreateDb:
	def test_has_raw_db(self, foldseek_with_db):
		assert foldseek_with_db.has_raw_db()

	def test_input_not_deleted(self, foldseek_with_db):
		assert foldseek_with_db.input_path.exists()


class TestFoldSeekCluster:
	def test_has_cluster_db(self, foldseek_clustered):
		assert foldseek_clustered.has_cluster_db(0.7)


class TestFoldSeekParse:
	def test_returns_dict(self, foldseek_parsed):
		_, clusters = foldseek_parsed
		assert isinstance(clusters, dict)
		assert len(clusters) > 0

	def test_all_chains_present(self, foldseek_parsed):
		_, clusters = foldseek_parsed
		# foldseek uses filenames as chain ids (without .cif.gz)
		expected = {"pdb1_A", "pdb1_B", "pdb2_A"}
		assert set(clusters.keys()) == expected

	def test_values_are_valid_chain_ids(self, foldseek_parsed):
		_, clusters = foldseek_parsed
		all_ids = set(clusters.keys())
		for rep in clusters.values():
			assert rep in all_ids

	def test_identical_structures_cluster_together(self, foldseek_parsed):
		_, clusters = foldseek_parsed
		# pdb1_A and pdb1_B have identical coordinates, should share a representative
		assert clusters["pdb1_A"] == clusters["pdb1_B"]

	def test_cluster_db_cleaned_up(self, foldseek_parsed):
		fs, _ = foldseek_parsed
		assert not fs.has_cluster_db(0.7)

	def test_tsv_exists(self, foldseek_parsed):
		fs, _ = foldseek_parsed
		assert fs.cluster_tsv_path(0.7).exists()


class TestFoldSeekLoadClusters:
	def test_matches_parse_result(self, foldseek_parsed):
		fs, parsed_clusters = foldseek_parsed
		loaded = fs.load_clusters(0.7)
		assert loaded == parsed_clusters


class TestFoldSeekCleanup:
	"""cleanup tests run last since they destroy state.
	use a separate instance to avoid polluting session fixtures."""

	@pytest.fixture
	def cleanup_foldseek(self, foldseek_parsed):
		"""return instance after full lifecycle, ready for cleanup tests"""
		fs, _ = foldseek_parsed
		return fs

	def test_cleanup_raw_db(self, cleanup_foldseek):
		cleanup_foldseek.cleanup_raw_db()
		assert not cleanup_foldseek.has_raw_db()

	def test_cleanup_tsvs(self, cleanup_foldseek):
		cleanup_foldseek.cleanup_tsvs()
		assert not cleanup_foldseek.cluster_tsv_path(0.7).exists()
