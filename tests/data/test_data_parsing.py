import pytest
import numpy as np

from proteus.data.data_constants import ChainKey, ProteinKey, ExpMethods
from proteus.data.downloads.proteus_dataset.data_parsing import _parse_mmcif, _best_atom, _best_residue


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
