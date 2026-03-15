import gzip
import textwrap

import pytest

from proteus.data.downloads.proteus_dataset.conf.cluster import FoldSeekCfg, MMSeqsCfg
from proteus.data.downloads.proteus_dataset.cluster import FoldSeek, MMSeqs


# -- test sequences for FASTA fixtures --

SEQUENCES = {
	"pdb1_A": "MKFLILFNILVSSSAYA",
	"pdb1_B": "MKFLILFNILVSSSAYA",  # identical to pdb1_A
	"pdb2_A": "GGGGGGGGGGGGGGGGG",  # distinct
	"pdb3_A": "MKFLILFNILVSSSAYG",  # similar to pdb1
}


def _make_fasta(name: str, seq: str) -> str:
	return f">{name}\n{seq}\n"


# -- minimal mmCIF for foldseek (N, CA, C per residue, 5 residues of alanine) --

_CIF_HEADER = textwrap.dedent("""\
	data_{name}
	loop_
	_atom_site.group_PDB
	_atom_site.id
	_atom_site.type_symbol
	_atom_site.label_atom_id
	_atom_site.label_alt_id
	_atom_site.label_comp_id
	_atom_site.label_asym_id
	_atom_site.label_entity_id
	_atom_site.label_seq_id
	_atom_site.Cartn_x
	_atom_site.Cartn_y
	_atom_site.Cartn_z
	_atom_site.occupancy
	_atom_site.B_iso_or_equiv
	_atom_site.auth_seq_id
	_atom_site.auth_asym_id
	_atom_site.pdbx_PDB_model_num
""")

# ideal alanine backbone geometry (N, CA, C) for 5 residues
# each residue offset ~3.8 angstroms along x
_BACKBONE_TEMPLATE = [
	# (atom_id, symbol, label_atom_id, dx, dy, dz) relative to residue origin
	("N",  "N",  0.000,  0.000,  0.000),
	("CA", "C",  1.458,  0.000,  0.000),
	("C",  "C",  2.009,  1.390,  0.000),
]


def _make_cif(name: str, z_offset: float = 0.0) -> bytes:
	"""generate minimal CIF content. z_offset shifts all z-coords to make structures distinct."""
	lines = [_CIF_HEADER.format(name=name)]
	atom_id = 1
	for res_idx in range(5):
		x_base = res_idx * 3.8
		for atom_label, symbol, dx, dy, dz in _BACKBONE_TEMPLATE:
			x = f"{x_base + dx:.3f}"
			y = f"{dy:.3f}"
			z = f"{dz + z_offset:.3f}"
			lines.append(
				f"ATOM {atom_id} {symbol} {atom_label} . ALA A 1 {res_idx + 1} "
				f"{x} {y} {z} 1.00 0.00 {res_idx + 1} A 1"
			)
			atom_id += 1
	return "\n".join(lines).encode()


# ---- FASTA fixtures ----

@pytest.fixture(scope="session")
def fasta_input_dir(tmp_path_factory):
	d = tmp_path_factory.mktemp("fasta_input")
	for name, seq in SEQUENCES.items():
		(d / f"{name}.fasta").write_text(_make_fasta(name, seq))
	return d


# ---- mmCIF fixtures ----

@pytest.fixture(scope="session")
def mmcif_input_dir(tmp_path_factory):
	d = tmp_path_factory.mktemp("mmcif_input")
	# pdb1_A and pdb1_B: identical coordinates
	for name in ("pdb1_A", "pdb1_B"):
		(d / f"{name}.cif.gz").write_bytes(gzip.compress(_make_cif(name, z_offset=0.0)))
	# pdb2_A: very different coordinates
	(d / "pdb2_A.cif.gz").write_bytes(gzip.compress(_make_cif("pdb2_A", z_offset=50.0)))
	return d


# ---- FoldSeek session fixtures ----

@pytest.fixture(scope="session")
def foldseek_cfg(mmcif_input_dir, tmp_path_factory):
	db = tmp_path_factory.mktemp("foldseek_db")
	return FoldSeekCfg(
		input_path=str(mmcif_input_dir),
		db_path=str(db),
		tmscore_thresholds=[0.7],
		verbosity=1,
	)


@pytest.fixture(scope="session")
def foldseek_instance(foldseek_cfg):
	return FoldSeek(foldseek_cfg)


@pytest.fixture(scope="session")
def foldseek_with_db(foldseek_instance):
	foldseek_instance.create_db()
	return foldseek_instance


@pytest.fixture(scope="session")
def foldseek_clustered(foldseek_with_db):
	foldseek_with_db.run_cluster(0.7)
	return foldseek_with_db


@pytest.fixture(scope="session")
def foldseek_parsed(foldseek_clustered):
	clusters = foldseek_clustered.parse_clusters(0.7)
	return foldseek_clustered, clusters


# ---- MMSeqs session fixtures ----

@pytest.fixture(scope="session")
def mmseqs_cfg(fasta_input_dir, tmp_path_factory):
	db = tmp_path_factory.mktemp("mmseqs_db")
	return MMSeqsCfg(
		input_path=str(fasta_input_dir),
		db_path=str(db),
		seq_id_thresholds=[0.3],
		verbosity=1,
	)


@pytest.fixture(scope="session")
def mmseqs_instance(mmseqs_cfg):
	return MMSeqs(mmseqs_cfg)


@pytest.fixture(scope="session")
def mmseqs_with_db(mmseqs_instance):
	mmseqs_instance.create_db()
	return mmseqs_instance


@pytest.fixture(scope="session")
def mmseqs_clustered(mmseqs_with_db):
	mmseqs_with_db.run_cluster(0.3)
	return mmseqs_with_db


@pytest.fixture(scope="session")
def mmseqs_parsed(mmseqs_clustered):
	clusters = mmseqs_clustered.parse_clusters(0.3)
	return mmseqs_clustered, clusters
