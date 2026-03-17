import pytest
import numpy as np

from proteus.data.data_constants import ChainKey, ProteinKey, ExpMethods
from proteus.data.downloads.proteus_dataset.data_parsing import _parse_mmcif, _best_atom, _best_residue, compute_chain_similarities


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _FakeAtom:
	def __init__(self, name: str, occ: float = 1.0):
		self.name = name
		self.occ = occ


class _FakeResidue(list):
	"""list of _FakeAtom with a .name attribute"""
	def __init__(self, name: str, atoms: list[_FakeAtom]):
		super().__init__(atoms)
		self.name = name


# ---------------------------------------------------------------------------
# TestParseMMCIF
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestParseMMCIF:

	def test_wrong_method_returns_none(self, mmcif_builder):
		cif = mmcif_builder(method="NEUTRON DIFFRACTION")
		result = _parse_mmcif(cif, {ExpMethods.XRAY}, max_resolution=3.5)
		assert result is None

	def test_resolution_too_high_returns_none(self, mmcif_builder):
		cif = mmcif_builder(resolution=5.0)
		result = _parse_mmcif(cif, {ExpMethods.XRAY}, max_resolution=3.5)
		assert result is None

	def test_resolution_zero_returns_none(self, mmcif_builder):
		# omitting resolution makes gemmi report 0.0
		cif = mmcif_builder(resolution=None)
		result = _parse_mmcif(cif, {ExpMethods.XRAY}, max_resolution=3.5)
		assert result is None

	def test_short_chains_skipped(self, mmcif_builder):
		cif = mmcif_builder(chains=[("A", "ACD")])  # 3 residues
		result = _parse_mmcif(cif, {ExpMethods.XRAY}, max_resolution=3.5, min_chain_length=4)
		assert result is None

	def test_successful_parse(self, mmcif_builder):
		seq = "ACDEFG"
		cif = mmcif_builder(method=ExpMethods.XRAY, resolution=2.0, chains=[("A", seq)])
		result = _parse_mmcif(cif, {ExpMethods.XRAY}, max_resolution=3.5)

		assert result is not None
		assert "A" in result[ProteinKey.CHAINS]
		chain = result[ProteinKey.CHAINS]["A"]
		assert chain[ChainKey.SEQUENCE] == seq
		L = len(seq)
		assert chain[ChainKey.COORDS].shape == (L, 14, 3)
		assert chain[ChainKey.ATOM_MASK].shape == (L, 14)
		assert result[ProteinKey.RESOLUTION] == 2.0
		assert result[ProteinKey.METHOD] == ExpMethods.XRAY

	def test_multiple_chains(self, mmcif_builder):
		cif = mmcif_builder(chains=[("A", "ACDEF"), ("B", "GGGGG")])
		result = _parse_mmcif(cif, {ExpMethods.XRAY}, max_resolution=3.5)

		assert result is not None
		assert "A" in result[ProteinKey.CHAINS]
		assert "B" in result[ProteinKey.CHAINS]
		assert result[ProteinKey.CHAINS]["A"][ChainKey.SEQUENCE] == "ACDEF"
		assert result[ProteinKey.CHAINS]["B"][ChainKey.SEQUENCE] == "GGGGG"

	def test_override_method(self, mmcif_builder):
		cif = mmcif_builder(method="NEUTRON DIFFRACTION")
		# without override, this would be None
		result = _parse_mmcif(
			cif, {ExpMethods.XRAY}, max_resolution=3.5,
			override_method=ExpMethods.XRAY,
		)
		assert result is not None
		assert result[ProteinKey.METHOD] == ExpMethods.XRAY

	def test_single_generator_assembly(self, mmcif_builder):
		"""1 assembly, 1 generator, 2 chains, 1 operator"""
		cif = mmcif_builder(
			chains=[("A", "ACDEFG"), ("B", "GGGGG")],
			assemblies=[[
				{"chains": ["A", "B"], "operators": [np.eye(4, dtype=np.float32)]},
			]],
		)
		result = _parse_mmcif(cif, {ExpMethods.XRAY}, max_resolution=3.5)
		assert result is not None
		asmbs = result[ProteinKey.ASSEMBLIES]
		assert len(asmbs) == 1
		assert len(asmbs[0]) == 1
		gen = asmbs[0][0]
		assert set(gen[ProteinKey.CHAINS]) == {"A", "B"}
		assert gen[ProteinKey.ASMB_XFORMS].shape == (1, 4, 4)

	def test_multi_operator_assembly(self, mmcif_builder):
		"""1 assembly, 1 generator, 2 chains, 3 operators"""
		op1 = np.eye(4, dtype=np.float32)
		op2 = np.eye(4, dtype=np.float32)
		op2[:3, 3] = [10.0, 0.0, 0.0]
		op3 = np.eye(4, dtype=np.float32)
		op3[:3, 3] = [0.0, 10.0, 0.0]
		cif = mmcif_builder(
			chains=[("A", "ACDEFG"), ("B", "GGGGG")],
			assemblies=[[
				{"chains": ["A", "B"], "operators": [op1, op2, op3]},
			]],
		)
		result = _parse_mmcif(cif, {ExpMethods.XRAY}, max_resolution=3.5)
		assert result is not None
		asmbs = result[ProteinKey.ASSEMBLIES]
		assert len(asmbs) == 1
		assert len(asmbs[0]) == 1
		gen = asmbs[0][0]
		assert set(gen[ProteinKey.CHAINS]) == {"A", "B"}
		assert gen[ProteinKey.ASMB_XFORMS].shape == (3, 4, 4)
		np.testing.assert_allclose(gen[ProteinKey.ASMB_XFORMS][0], op1, atol=1e-5)
		np.testing.assert_allclose(gen[ProteinKey.ASMB_XFORMS][1], op2, atol=1e-5)
		np.testing.assert_allclose(gen[ProteinKey.ASMB_XFORMS][2], op3, atol=1e-5)

	def test_multi_generator_assembly(self, mmcif_builder):
		"""1 assembly, 2 generators with different chains and operators"""
		op1 = np.eye(4, dtype=np.float32)
		op2 = np.eye(4, dtype=np.float32)
		op2[:3, 3] = [10.0, 0.0, 0.0]
		cif = mmcif_builder(
			chains=[("A", "ACDEFG"), ("B", "GGGGG")],
			assemblies=[[
				{"chains": ["A"], "operators": [op1]},
				{"chains": ["B"], "operators": [op1, op2]},
			]],
		)
		result = _parse_mmcif(cif, {ExpMethods.XRAY}, max_resolution=3.5)
		assert result is not None
		asmbs = result[ProteinKey.ASSEMBLIES]
		assert len(asmbs) == 1
		assert len(asmbs[0]) == 2
		gen0 = asmbs[0][0]
		gen1 = asmbs[0][1]
		assert gen0[ProteinKey.CHAINS] == ["A"]
		assert gen0[ProteinKey.ASMB_XFORMS].shape == (1, 4, 4)
		assert gen1[ProteinKey.CHAINS] == ["B"]
		assert gen1[ProteinKey.ASMB_XFORMS].shape == (2, 4, 4)


# ---------------------------------------------------------------------------
# TestBestAtom
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestBestAtom:

	def test_highest_occupancy_wins(self):
		res = _FakeResidue("ALA", [
			_FakeAtom("CA", occ=0.5),
			_FakeAtom("CA", occ=0.9),
			_FakeAtom("N", occ=1.0),
		])
		atom = _best_atom(res, "CA")
		assert atom.occ == 0.9

	def test_missing_atom_returns_none(self):
		res = _FakeResidue("ALA", [_FakeAtom("CA", occ=1.0)])
		assert _best_atom(res, "CB") is None


# ---------------------------------------------------------------------------
# TestBestResidue
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestBestResidue:

	def test_single_returns_itself(self):
		res = _FakeResidue("ALA", [_FakeAtom("CA", occ=1.0)])
		assert _best_residue([res]) is res

	def test_highest_avg_occupancy_wins(self):
		low = _FakeResidue("ALA", [_FakeAtom("CA", occ=0.3), _FakeAtom("N", occ=0.3)])
		high = _FakeResidue("ALA", [_FakeAtom("CA", occ=0.9), _FakeAtom("N", occ=0.8)])
		assert _best_residue([low, high]) is high


# ---------------------------------------------------------------------------
# helpers for chain similarity tests
# ---------------------------------------------------------------------------

def _make_chain(sequence: str, coords: np.ndarray) -> dict:
	"""build a minimal chain dict with coords and full atom mask for CA."""
	L = len(sequence)
	assert coords.shape == (L, 3), "coords must be (L, 3) CA positions"
	full_coords = np.zeros((L, 14, 3), dtype=np.float32)
	full_coords[:, 1, :] = coords  # atom index 1 = CA
	mask = np.zeros((L, 14), dtype=bool)
	mask[:, 1] = True
	return {
		ChainKey.SEQUENCE: sequence,
		ChainKey.COORDS: full_coords,
		ChainKey.ATOM_MASK: mask,
	}


# ---------------------------------------------------------------------------
# TestComputeChainSimilarities
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestComputeChainSimilarities:

	def test_single_chain(self):
		"""single chain returns (1,1) arrays of [[1.0]]"""
		coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
		chains = {"A": _make_chain("ACDE", coords)}
		tm, si = compute_chain_similarities(chains, ["A"])
		assert tm.shape == (1, 1)
		assert si.shape == (1, 1)
		np.testing.assert_allclose(tm, [[1.0]])
		np.testing.assert_allclose(si, [[1.0]])

	def test_identical_chains(self):
		"""two identical chains should have TM=1.0 and identity=1.0"""
		coords = np.array([
			[0, 0, 0], [3.8, 0, 0], [7.6, 0, 0], [11.4, 0, 0],
			[15.2, 0, 0], [19.0, 0, 0], [22.8, 0, 0], [26.6, 0, 0],
		], dtype=np.float32)
		seq = "ACDEFGHI"
		chains = {
			"A": _make_chain(seq, coords),
			"B": _make_chain(seq, coords.copy()),
		}
		tm, si = compute_chain_similarities(chains, ["A", "B"])
		assert tm.shape == (2, 2)
		np.testing.assert_allclose(tm[0, 1], 1.0, atol=1e-3)
		np.testing.assert_allclose(tm[1, 0], 1.0, atol=1e-3)
		np.testing.assert_allclose(si[0, 1], 1.0, atol=1e-3)

	def test_different_chains(self):
		"""two distinct chains should have scores in (0, 1)"""
		rng = np.random.default_rng(123)
		coords_a = np.cumsum(rng.standard_normal((10, 3)).astype(np.float32) * 3.8, axis=0)
		coords_b = np.cumsum(rng.standard_normal((8, 3)).astype(np.float32) * 3.8, axis=0)
		chains = {
			"A": _make_chain("ACDEFGHIKL", coords_a),
			"B": _make_chain("WWWWWWWW", coords_b),
		}
		tm, si = compute_chain_similarities(chains, ["A", "B"])
		# off-diagonal should be in (0, 1)
		assert 0 < tm[0, 1] < 1
		assert 0 < tm[1, 0] < 1
		assert 0 < si[0, 1] < 1
		# diagonal should be 1
		np.testing.assert_allclose(np.diag(tm), 1.0)
		np.testing.assert_allclose(np.diag(si), 1.0)

	def test_skip_tm_for_similar_chains(self):
		"""chains with high sequence similarity should skip TM-align"""
		from unittest.mock import patch

		coords_a = np.array([
			[0, 0, 0], [3.8, 0, 0], [7.6, 0, 0], [11.4, 0, 0],
			[15.2, 0, 0], [19.0, 0, 0], [22.8, 0, 0], [26.6, 0, 0],
		], dtype=np.float32)
		coords_b = np.cumsum(np.ones((8, 3), dtype=np.float32) * 3.8, axis=0)
		seq_ab = "ACDEFGHI"
		seq_cd = "WWWWWWWW"

		chains = {
			"A": _make_chain(seq_ab, coords_a),
			"B": _make_chain(seq_ab, coords_a.copy()),
			"C": _make_chain(seq_cd, coords_b),
			"D": _make_chain(seq_cd, coords_b.copy()),
		}

		import tmtools as _tmtools
		original_tm_align = _tmtools.tm_align

		call_count = [0]
		def counting_tm_align(*args, **kwargs):
			call_count[0] += 1
			return original_tm_align(*args, **kwargs)

		with patch("proteus.data.downloads.proteus_dataset.data_parsing.tmtools.tm_align", side_effect=counting_tm_align):
			tm, si = compute_chain_similarities(chains, ["A", "B", "C", "D"])

		# identical-sequence pairs should be 1.0 (skipped TM via seq similarity)
		np.testing.assert_allclose(tm[0, 1], 1.0)
		np.testing.assert_allclose(tm[1, 0], 1.0)
		np.testing.assert_allclose(tm[2, 3], 1.0)
		np.testing.assert_allclose(tm[3, 2], 1.0)
		np.testing.assert_allclose(si[0, 1], 1.0)
		np.testing.assert_allclose(si[2, 3], 1.0)

		# cross-group pairs should be computed (not 1.0 for different sequences)
		assert 0 < tm[0, 2] < 1

		# TM-align only called for dissimilar pairs (4 cross-group pairs: A-C, A-D, B-C, B-D)
		assert call_count[0] == 4, f"expected 4 tm_align calls, got {call_count[0]}"

	def test_index_order_matches_chain_keys(self):
		"""array indices must follow the provided chain_ids order"""
		coords_a = np.array([[0, 0, 0], [3.8, 0, 0], [7.6, 0, 0], [11.4, 0, 0]], dtype=np.float32)
		coords_b = np.array([[0, 5, 0], [3.8, 5, 0], [7.6, 5, 0], [11.4, 5, 0]], dtype=np.float32)
		chains = {
			"X": _make_chain("GGGG", coords_a),
			"Y": _make_chain("GGGG", coords_b),
		}
		tm_xy, _ = compute_chain_similarities(chains, ["X", "Y"])
		tm_yx, _ = compute_chain_similarities(chains, ["Y", "X"])
		# swapping order should transpose the matrix
		np.testing.assert_allclose(tm_xy, tm_yx.T, atol=1e-5)
