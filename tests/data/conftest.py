import numpy as np
import pytest

from proteus.static.constants import atoms as atom14_order, three_2_one
from proteus.data.data_constants import ChainKey, ProteinKey, DataSource, ExpMethods


def _build_mmcif(
	pdb_id: str = "1abc",
	method: str = ExpMethods.XRAY,
	resolution: float | None = 2.0,
	chains: list[tuple[str, str]] | None = None,
	deposit_date: str = "2020-01-01",
) -> str:
	"""build a minimal but valid mmCIF string that gemmi can parse.

	chains: list of (chain_id, one_letter_sequence). defaults to [("A", "ACDEFG")].
	resolution: None omits the tag entirely (gemmi returns 0.0).
	"""
	if chains is None:
		chains = [("A", "ACDEFG")]

	one_2_three = {v: k for k, v in three_2_one.items()}

	lines = [
		f"data_{pdb_id.upper()}",
		f"_entry.id {pdb_id.upper()}",
		f"_exptl.method '{method}'",
		f"_pdbx_database_status.recvd_initial_deposition_date {deposit_date}",
	]

	# gemmi needs _refine in loop format to populate structure.resolution
	if resolution is not None:
		lines += [
			"loop_",
			"_refine.pdbx_refine_id",
			"_refine.ls_d_res_high",
			f"'{method}' {resolution:.2f}",
		]

	# gemmi needs entity records to recognize chains as polymers
	entity_lines = ["loop_", "_entity.id", "_entity.type"]
	poly_lines = ["loop_", "_entity_poly.entity_id", "_entity_poly.type", "_entity_poly.pdbx_strand_id"]
	for i, (chain_id, _) in enumerate(chains, start=1):
		entity_lines.append(f"{i} polymer")
		poly_lines.append(f"{i} polypeptide(L) {chain_id}")
	lines += entity_lines + poly_lines

	# atom site loop
	lines += [
		"loop_",
		"_atom_site.group_PDB",
		"_atom_site.id",
		"_atom_site.type_symbol",
		"_atom_site.label_atom_id",
		"_atom_site.label_alt_id",
		"_atom_site.label_comp_id",
		"_atom_site.label_asym_id",
		"_atom_site.label_entity_id",
		"_atom_site.label_seq_id",
		"_atom_site.Cartn_x",
		"_atom_site.Cartn_y",
		"_atom_site.Cartn_z",
		"_atom_site.occupancy",
		"_atom_site.B_iso_or_equiv",
		"_atom_site.auth_seq_id",
		"_atom_site.auth_asym_id",
		"_atom_site.pdbx_PDB_model_num",
	]

	atom_id = 0
	for chain_id, seq in chains:
		for res_idx, aa in enumerate(seq, start=1):
			resname = one_2_three[aa]
			atom_names = atom14_order[resname]
			for atom_name in atom_names:
				atom_id += 1
				elem = atom_name[0]
				x, y, z = float(atom_id), float(atom_id) * 0.5, float(atom_id) * 0.25
				lines.append(
					f"ATOM {atom_id} {elem} {atom_name} . {resname} "
					f"{chain_id} 1 {res_idx} "
					f"{x:.3f} {y:.3f} {z:.3f} "
					f"1.00 30.00 {res_idx} {chain_id} 1"
				)

	return "\n".join(lines) + "\n"


def _build_protein_dict(
	chains: list[tuple[str, str]] | None = None,
	resolution: float = 2.0,
	method: str = ExpMethods.XRAY,
	source: str = DataSource.RCSB,
	deposit_date: str = "2020-01-01",
) -> dict:
	"""build a dict matching _parse_mmcif output with synthetic numpy arrays."""
	if chains is None:
		chains = [("A", "ACDEFG")]

	rng = np.random.default_rng(42)
	chains_data = {}
	for chain_id, seq in chains:
		L = len(seq)
		chains_data[chain_id] = {
			ChainKey.SEQUENCE: seq,
			ChainKey.COORDS: rng.standard_normal((L, 14, 3)).astype(np.float32),
			ChainKey.ATOM_MASK: np.ones((L, 14), dtype=bool),
			ChainKey.BFACTOR: rng.uniform(10, 50, (L, 14)).astype(np.float32),
			ChainKey.PLDDT: np.full(L, float("nan"), dtype=np.float32),
			ChainKey.OCCUPANCY: np.ones((L, 14), dtype=np.float32),
			ChainKey.CIF: f"data_{chain_id}\nloop_\n",
		}

	# identity assembly
	assemblies = [{
		ProteinKey.CHAINS: [cid for cid, _ in chains],
		ProteinKey.ASMB_XFORMS: np.eye(4, dtype=np.float32)[np.newaxis],
	}]

	return {
		ProteinKey.CHAINS: chains_data,
		ProteinKey.ASSEMBLIES: assemblies,
		ProteinKey.RESOLUTION: resolution,
		ProteinKey.METHOD: method,
		ProteinKey.DEPOSIT_DATE: deposit_date,
		ProteinKey.SOURCE: source,
		ProteinKey.MEAN_PLDDT: float("nan"),
		ProteinKey.PTM: float("nan"),
	}


@pytest.fixture
def mmcif_builder():
	return _build_mmcif


@pytest.fixture
def protein_dict_builder():
	return _build_protein_dict
