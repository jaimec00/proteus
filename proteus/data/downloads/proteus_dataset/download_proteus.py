'''
entry point for the experimental structure data collection and cleaning pipeline
'''

import gzip
import io
import json
import logging
import aiohttp
import aioboto3
import asyncio
import requests
import hydra
from pathlib import Path
from cloudpathlib import S3Path
from tqdm import tqdm

import numpy as np
import gemmi

from proteus.types import Dict, List
from proteus.static.constants import resname_2_one, noncanonical_parent, atoms as atom14_order
from proteus.utils.s3_utils import upload_bytes_to_s3, REGION
from proteus.data.downloads.proteus_dataset.conf.download import (
	ExperimentalDataDownloadCfg,
	register_download_configs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

register_download_configs()

class ExperimentalDataDownload:
	def __init__(self, cfg: ExperimentalDataDownloadCfg):
		self.methods = set(cfg.methods)
		self.max_resolution = cfg.max_resolution
		self.max_entries = cfg.max_entries
		self.semaphore_limit = cfg.semaphore_limit
		self.s3_path = cfg.s3_path
		self.local_path = Path(cfg.local_path)

	def download(self):
		experimental_ids = self._get_experimental_ids()
		asyncio.run(self._download_async(experimental_ids))

	async def _download_async(self, experimental_ids: List[str]):
		semaphore = asyncio.Semaphore(self.semaphore_limit)
		s3 = S3Path(self.s3_path)
		self.local_path.mkdir(parents=True, exist_ok=True)
		pbar = tqdm(total=len(experimental_ids), desc="downloading")

		async def _task(pdb_id, s3_client):
			result = await self._maybe_pdbredo_else_rcsb(pdb_id, session, semaphore, s3_client, s3.bucket, s3.key)
			pbar.update(1)
			return result

		s3_session = aioboto3.Session()
		async with aiohttp.ClientSession() as session, \
			s3_session.client("s3", region_name=REGION) as s3_client:
			tasks = [_task(pdb_id, s3_client) for pdb_id in experimental_ids]
			results = await asyncio.gather(*tasks, return_exceptions=True)
		pbar.close()

		succeeded, failed = 0, 0
		for pdb_id, result in zip(experimental_ids, results):
			if isinstance(result, Exception):
				logger.error(f"{pdb_id}: {result}")
				failed += 1
			elif result is None:
				failed += 1
			else:
				succeeded += 1

		logger.info(f"done: {succeeded} succeeded, {failed} skipped out of {len(experimental_ids)}")

	async def _maybe_pdbredo_else_rcsb(
		self, pdb_id: str, session: aiohttp.ClientSession,
		semaphore: asyncio.Semaphore, s3_client, bucket: str, prefix: str,
	):
		# semaphore only covers HTTP downloads to rate-limit external requests
		async with semaphore:
			data = await self._download_pdbredo(pdb_id, session)
			if data is None:
				data = await self._download_rcsb(pdb_id, session)
		if data is None:
			return None

		pid = pdb_id.lower()
		key_base = f"{prefix}/{pid}" if prefix else pid

		# upload per-chain npz, write ca cif locally for foldseek
		for chain_id, chain_data in data["chains"].items():
			buf = io.BytesIO()
			np.savez_compressed(
				buf,
				coords=chain_data["coords"],
				atom_mask=chain_data["atom_mask"],
				bfactor=chain_data["bfactor"],
				sequence=np.array(chain_data["sequence"]),
			)
			await upload_bytes_to_s3(buf.getvalue(), bucket, f"{key_base}/{pid}_{chain_id}.npz", s3_client)

			cif_path = self.local_path / f"{pid}_{chain_id}_ca.cif.gz"
			cif_path.write_bytes(gzip.compress(chain_data["ca_cif"].encode()))

		# upload pdb-level metadata as gzipped json
		meta = {
			"resolution": data["resolution"],
			"method": data["method"],
			"deposit_date": data["deposit_date"],
			"source": data["source"],
			"assemblies": [
				{
					"chains": a["chains"],
					"transforms": a["transforms"].tolist(),
				}
				for a in data["assemblies"]
			],
		}
		meta_gz = gzip.compress(json.dumps(meta).encode())
		await upload_bytes_to_s3(meta_gz, bucket, f"{key_base}/{pid}_meta.json.gz", s3_client)

		return pdb_id

	async def _download_pdbredo(self, pdb_id: str, session: aiohttp.ClientSession):
		pid = pdb_id.lower()
		url = f"https://pdb-redo.eu/db/{pid}/{pid}_final.cif"
		async with session.get(url) as resp:
			if resp.status != 200:
				return None
			content = (await resp.read()).decode("utf-8")
		data = _parse_mmcif(content, self.methods, self.max_resolution)
		if data is None:
			return None
		data |= {"source": "pdb-redo"}
		return data

	async def _download_rcsb(self, pdb_id: str, session: aiohttp.ClientSession):
		pid = pdb_id.lower()
		url = f"https://files.wwpdb.org/pub/pdb/data/structures/divided/mmCIF/{pid[1:3]}/{pid}.cif.gz"
		async with session.get(url) as resp:
			if resp.status != 200:
				return None
			raw = await resp.read()
		with gzip.open(io.BytesIO(raw), "rt") as f:
			content = f.read()

		data = _parse_mmcif(content, self.methods, self.max_resolution)
		if data is None:
			return None
		data |= {"source": "rcsb"}
		return data

	def _get_experimental_ids(self) -> List[str]:
		url = "https://files.wwpdb.org/pub/pdb/holdings/current_file_holdings.json.gz"
		logger.info(f"retrieving rcsb metadata at {url} ...")
		resp = requests.get(url)
		resp.raise_for_status()

		with gzip.open(io.BytesIO(resp.content), "rt") as f:
			holdings = json.load(f)
		pdbids = list(holdings.keys())

		if self.max_entries > 0:
			pdbids = pdbids[:self.max_entries]

		return pdbids

def _parse_mmcif(content: str, methods: List[str], max_resolution: float) -> Dict | None:
	structure = gemmi.read_structure_string(content)

	# filter by experimental method
	method = structure.info['_exptl.method'] if '_exptl.method' in structure.info else ''
	method = method.strip("'\" ")
	if method not in methods:
		return None

	# filter by resolution
	if structure.resolution > max_resolution:
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

		if not resolved:
			continue

		L = len(resolved)
		coords = np.zeros((L, 14, 3), dtype=np.float32)
		mask = np.zeros((L, 14), dtype=bool)
		bfactors = np.zeros(L, dtype=np.float32)
		seq = []
		ca_cif_lines = [
			f"data_{chain.name}",
			"loop_",
			"_atom_site.group_PDB",
			"_atom_site.id",
			"_atom_site.label_atom_id",
			"_atom_site.label_comp_id",
			"_atom_site.label_asym_id",
			"_atom_site.label_seq_id",
			"_atom_site.Cartn_x",
			"_atom_site.Cartn_y",
			"_atom_site.Cartn_z",
		]
		ca_atom_id = 0

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
						ca_atom_id += 1
						ca_cif_lines.append(
							f"ATOM {ca_atom_id} CA {resname} {chain.name} {i + 1} "
							f"{atom.pos.x:.3f} {atom.pos.y:.3f} {atom.pos.z:.3f}"
						)

		chains_data[chain.name] = {
			"sequence": "".join(seq),
			"coords": coords,
			"atom_mask": mask,
			"bfactor": bfactors,
			"ca_cif": "\n".join(ca_cif_lines) + "\n",
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
		assemblies.append({
			"chains": biounit_chains,
			"transforms": np.stack(transforms) if transforms else np.empty((0, 4, 4), dtype=np.float32),
		})

	if not chains_data:
		return None

	date_key = '_pdbx_database_status.recvd_initial_deposition_date'
	deposit_date = structure.info[date_key] if date_key in structure.info else ''

	return {
		"chains": chains_data,
		"assemblies": assemblies,
		"resolution": structure.resolution,
		"method": method,
		"deposit_date": deposit_date,
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


@hydra.main(version_base=None, config_name="default")
def main(cfg: ExperimentalDataDownloadCfg):
	downloader = ExperimentalDataDownload(cfg)
	downloader.download()

if __name__ == "__main__":
	main()
