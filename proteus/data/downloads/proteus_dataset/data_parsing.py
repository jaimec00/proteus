import logging
import numpy as np
import gemmi

from proteus.types import Dict, List
from proteus.static.constants import resname_2_one, noncanonical_parent, atoms as atom14_order
from proteus.data.data_constants import ChainKey, ProteinKey

logger = logging.getLogger(__name__)

def _parse_mmcif(content: str, methods: List[str], max_resolution: float, min_chain_length: int = 4) -> Dict | None:
	structure = gemmi.read_structure_string(content)
	pdb_id = structure.name.lower()

	# filter by experimental method
	method = structure.info['_exptl.method'] if '_exptl.method' in structure.info else ''
	method = method.strip("'\" ")
	if method not in methods:
		logger.info(f"{pdb_id}: skipped, method '{method}' not in {methods}")
		return None

	# filter by resolution
	if structure.resolution > max_resolution:
		logger.info(f"{pdb_id}: skipped, resolution {structure.resolution:.2f} > {max_resolution}")
		return None

	model = structure[0]

	# per-chain data
	chains_data = {}
	for chain in model:
		polymer = chain.get_polymer()
		if not polymer:
			continue

		# collect residues, resolving microheterogeneity
		resolved = []
		for res_or_group in polymer:
			if hasattr(res_or_group, '__iter__') and not hasattr(res_or_group, 'name'):
				candidates = [r for r in res_or_group if r.name in resname_2_one]
				if candidates:
					resolved.append(_best_residue(candidates))
			elif res_or_group.name in resname_2_one:
				resolved.append(res_or_group)

		if len(resolved) < min_chain_length:
			continue

		L = len(resolved)
		coords = np.zeros((L, 14, 3), dtype=np.float32)
		mask = np.zeros((L, 14), dtype=bool)
		bfactors = np.zeros(L, dtype=np.float32)
		seq = []
		# backbone atoms for foldseek (needs at least N, CA, C to compute 3Di)
		backbone_names = ["N", "CA", "C", "O", "CB"]
		cif_lines = [
			f"data_{chain.name}",
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
		cif_atom_id = 0

		for i, res in enumerate(resolved):
			resname = res.name
			seq.append(resname_2_one[resname])
			parent = noncanonical_parent.get(resname, resname)
			expected_atoms = atom14_order[parent]

			for j, atom_name in enumerate(expected_atoms):
				atom = _best_atom(res, atom_name)
				if atom is not None:
					coords[i, j] = [atom.pos.x, atom.pos.y, atom.pos.z]
					mask[i, j] = True
					if atom_name == "CA":
						bfactors[i] = atom.b_iso

					if atom_name in backbone_names:
						cif_atom_id += 1
						elem = atom_name[0]
						cif_lines.append(
							f"ATOM {cif_atom_id} {elem} {atom_name} . {resname} "
							f"{chain.name} 1 {i + 1} "
							f"{atom.pos.x:.3f} {atom.pos.y:.3f} {atom.pos.z:.3f} "
							f"1.00 {atom.b_iso:.2f} {i + 1} {chain.name} 1"
						)

		chains_data[chain.name] = {
			ChainKey.SEQUENCE: "".join(seq),
			ChainKey.COORDS: coords,
			ChainKey.ATOM_MASK: mask,
			ChainKey.BFACTOR: bfactors,
			ChainKey.CIF: "\n".join(cif_lines) + "\n",
		}

	# assembly / biounit info with Nx4x4 homogeneous transforms
	assemblies = []
	for assembly in structure.assemblies:
		biounit_chains = []
		transforms = []
		for gen in assembly.generators:
			biounit_chains.extend(list(gen.subchains) or list(gen.chains))
			for oper in gen.operators:
				mat4 = np.eye(4, dtype=np.float32)
				mat4[:3, :3] = np.array(oper.transform.mat.tolist(), dtype=np.float32)
				mat4[:3, 3] = np.array(oper.transform.vec.tolist(), dtype=np.float32)
				transforms.append(mat4)
		# only keep chains that passed filtering
		biounit_chains = [c for c in biounit_chains if c in chains_data]
		if not biounit_chains:
			continue
		assemblies.append({
			ProteinKey.CHAINS: biounit_chains,
			ProteinKey.ASMB_XFORMS: np.stack(transforms) if transforms else np.empty((0, 4, 4), dtype=np.float32),
		})

	if not chains_data:
		logger.info(f"{pdb_id}: skipped, no chains with >= {min_chain_length} residues")
		return None

	date_key = '_pdbx_database_status.recvd_initial_deposition_date'
	deposit_date = structure.info[date_key] if date_key in structure.info else ''

	return {
		ProteinKey.CHAINS: chains_data,
		ProteinKey.ASSEMBLIES: assemblies,
		ProteinKey.RESOLUTION: structure.resolution,
		ProteinKey.METHOD: method,
		ProteinKey.DEPOSIT_DATE: deposit_date,
	}

def _best_atom(res, atom_name: str):
	'''find the atom with the highest occupancy'''
	best = None
	for atom in res:
		if atom.name != atom_name:
			continue
		if best is None or atom.occ > best.occ:
			best = atom
	return best

def _best_residue(residues):
	'''for microheterogeneity, pick the residue with highest average occupancy'''
	if len(residues) == 1:
		return residues[0]
	best_res, best_occ = None, -1.0
	for res in residues:
		atoms = [a for a in res]
		if not atoms:
			continue
		avg_occ = sum(a.occ for a in atoms) / len(atoms)
		if avg_occ > best_occ:
			best_occ = avg_occ
			best_res = res
	return best_res