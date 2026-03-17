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
	assemblies: list[list[dict]] | None = None,
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

	# assembly records for gemmi to populate structure.assemblies
	if assemblies is not None:
		all_ops = []
		gen_records = []
		for asmb_idx, asmb in enumerate(assemblies):
			for gen in asmb:
				op_ids = []
				for op in gen["operators"]:
					all_ops.append(op)
					op_ids.append(str(len(all_ops)))
				gen_records.append((str(asmb_idx + 1), ",".join(op_ids), ",".join(gen["chains"])))

		# gemmi needs all 5 _pdbx_struct_assembly columns to parse assemblies
		lines += [
			"loop_",
			"_pdbx_struct_assembly.id",
			"_pdbx_struct_assembly.details",
			"_pdbx_struct_assembly.method_details",
			"_pdbx_struct_assembly.oligomeric_details",
			"_pdbx_struct_assembly.oligomeric_count",
		]
		for asmb_idx in range(len(assemblies)):
			lines.append(f"{asmb_idx + 1} author_defined_assembly ? ? ?")

		lines += [
			"loop_",
			"_pdbx_struct_assembly_gen.assembly_id",
			"_pdbx_struct_assembly_gen.oper_expression",
			"_pdbx_struct_assembly_gen.asym_id_list",
		]
		for asmb_id, oper_expr, chain_list in gen_records:
			lines.append(f"{asmb_id} {oper_expr} {chain_list}")

		lines += [
			"loop_",
			"_pdbx_struct_oper_list.id",
			"_pdbx_struct_oper_list.type",
			"_pdbx_struct_oper_list.matrix[1][1]",
			"_pdbx_struct_oper_list.matrix[1][2]",
			"_pdbx_struct_oper_list.matrix[1][3]",
			"_pdbx_struct_oper_list.vector[1]",
			"_pdbx_struct_oper_list.matrix[2][1]",
			"_pdbx_struct_oper_list.matrix[2][2]",
			"_pdbx_struct_oper_list.matrix[2][3]",
			"_pdbx_struct_oper_list.vector[2]",
			"_pdbx_struct_oper_list.matrix[3][1]",
			"_pdbx_struct_oper_list.matrix[3][2]",
			"_pdbx_struct_oper_list.matrix[3][3]",
			"_pdbx_struct_oper_list.vector[3]",
		]
		for i, op in enumerate(all_ops):
			mat = op[:3, :3]
			vec = op[:3, 3]
			lines.append(
				f"{i + 1} 'crystal symmetry operation' "
				f"{mat[0, 0]:.6f} {mat[0, 1]:.6f} {mat[0, 2]:.6f} {vec[0]:.6f} "
				f"{mat[1, 0]:.6f} {mat[1, 1]:.6f} {mat[1, 2]:.6f} {vec[1]:.6f} "
				f"{mat[2, 0]:.6f} {mat[2, 1]:.6f} {mat[2, 2]:.6f} {vec[2]:.6f}"
			)

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

	# identity assembly (list of assemblies, each a list of generators)
	assemblies = [[{
		ProteinKey.CHAINS: [cid for cid, _ in chains],
		ProteinKey.ASMB_XFORMS: np.eye(4, dtype=np.float32)[np.newaxis],
	}]]

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
