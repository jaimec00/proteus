import logging

import numpy as np
import gemmi
import tmtools
import parasail

from proteus.types import Dict, List
from proteus.static.constants import resname_2_one, noncanonical_parent, atoms as atom14_order
from proteus.data.data_constants import ChainKey, ProteinKey

logger = logging.getLogger(__name__)

def _parse_mmcif(
	content: str, methods: List[str], max_resolution: float,
	min_chain_length: int = 4, override_method: str | None = None,
) -> Dict | None:
	structure = gemmi.read_structure_string(content)
	pdb_id = structure.name.lower()

	# pdb-redo refined cifs don't carry _exptl.method metadata, but all pdb-redo
	# entries are x-ray. override_method lets the caller set it explicitly so we
	# skip the filter and still record the correct method in the index.
	if override_method is not None:
		method = override_method
	else:
		raw_method = structure.info['_exptl.method'] if '_exptl.method' in structure.info else ''
		raw_method = raw_method.strip("'\" ")
		# accept multi-method entries if any listed method matches
		matched = [m for m in methods if m in raw_method]
		if not matched:
			logger.info(f"{pdb_id}: skipped, method '{raw_method}' not in {methods}")
			return None
		method = matched[0]

	# filter by resolution (0.0 means unset)
	if structure.resolution == 0.0 or structure.resolution > max_resolution:
		logger.info(f"{pdb_id}: skipped, resolution {structure.resolution:.2f} (max {max_resolution})")
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
		bfactors = np.zeros((L, 14), dtype=np.float32)
		occupancies = np.zeros((L, 14), dtype=np.float32)
		plddts = np.full(L, float('nan'), dtype=np.float32)
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
					bfactors[i, j] = atom.b_iso
					occupancies[i, j] = atom.occ

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
			ChainKey.PLDDT: plddts,
			ChainKey.OCCUPANCY: occupancies,
			ChainKey.CIF: "\n".join(cif_lines) + "\n",
		}

	# assembly / biounit info with Nx4x4 homogeneous transforms
	# each assembly is a list of generators, each generator has its own chains and operators
	assemblies = []
	for assembly in structure.assemblies:
		generators = []
		for gen in assembly.generators:
			gen_chains = list(gen.subchains) or list(gen.chains)
			biounit_chains = list(dict.fromkeys(c for c in gen_chains if c in chains_data))
			if not biounit_chains:
				continue
			transforms = []
			for oper in gen.operators:
				mat4 = np.eye(4, dtype=np.float32)
				mat4[:3, :3] = np.array(oper.transform.mat.tolist(), dtype=np.float32)
				mat4[:3, 3] = np.array(oper.transform.vec.tolist(), dtype=np.float32)
				transforms.append(mat4)
			generators.append({
				ProteinKey.CHAINS: biounit_chains,
				ProteinKey.ASMB_XFORMS: np.stack(transforms) if transforms else np.empty((0, 4, 4), dtype=np.float32),
			})
		if generators:
			assemblies.append(generators)

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
		ProteinKey.MEAN_PLDDT: float('nan'),
		ProteinKey.PTM: float('nan'),
	}

def compute_chain_similarities(
	chains_data: dict, chain_ids: list[str],
	gap_open: int = 10, gap_extend: int = 1,
	skip_tm_at_seqsim: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
	"""compute pairwise chain TM-scores and sequence identity.

	sequence alignment (parasail) is computed for all unique pairs first.
	TM-align is only run for pairs below skip_tm_at_seqsim — pairs above
	are treated as homo chains (TM=1.0).

	returns (tm_scores, seq_identity) both (C, C) float32 arrays
	where C = len(chain_ids). array index order matches chain_ids.
	"""
	C = len(chain_ids)
	tm_scores = np.eye(C, dtype=np.float32)
	seq_identity = np.eye(C, dtype=np.float32)

	if C == 1:
		return tm_scores, seq_identity

	# extract CA coords and sequences per chain
	# full sequences for parasail, CA-filtered sequences for tmtools
	ca_coords = []
	ca_sequences = []
	full_sequences = []
	for cid in chain_ids:
		chain = chains_data[cid]
		coords = chain[ChainKey.COORDS]      # (L, 14, 3)
		mask = chain[ChainKey.ATOM_MASK]      # (L, 14)
		seq = chain[ChainKey.SEQUENCE]
		# CA is atom index 1
		ca_mask = mask[:, 1]
		ca_xyz = coords[ca_mask, 1, :]
		ca_coords.append(ca_xyz)
		ca_sequences.append("".join(aa for aa, has_ca in zip(seq, ca_mask) if has_ca))
		full_sequences.append(seq)

	# compute sequence identity for all unique pairs, only run TM-align
	# for pairs below the similarity threshold
	matrix = parasail.blosum62
	for i in range(C):
		for j in range(i + 1, C):
			res = parasail.nw_stats_striped_16(full_sequences[i], full_sequences[j], gap_open, gap_extend, matrix)
			max_len = max(len(full_sequences[i]), len(full_sequences[j]))
			si_val = res.similar / max_len

			seq_identity[i, j] = si_val
			seq_identity[j, i] = si_val

			if si_val >= skip_tm_at_seqsim:
				tm_scores[i, j] = 1.0
				tm_scores[j, i] = 1.0
			else:
				result = tmtools.tm_align(ca_coords[i], ca_coords[j], ca_sequences[i], ca_sequences[j])
				tm_scores[i, j] = result.tm_norm_chain1
				tm_scores[j, i] = result.tm_norm_chain2

	return tm_scores, seq_identity


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
