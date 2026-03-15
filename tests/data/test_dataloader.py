"""tests for the S3-shard-based dataloader: S3Orchestrator, Sampler, PDBData, and Data iterator"""

from __future__ import annotations

from unittest.mock import patch, MagicMock
from dataclasses import dataclass

import numpy as np
import polars as pl
import pytest

from proteus.data.data_utils import (
	S3Orchestrator, BlobEntry, ReadGroup, Sampler, PDBData, BatchBuilder, Assembly,
)
from proteus.data.data_constants import IndexCol, ProteinKey, ChainKey


# ── S3Orchestrator ──

class TestS3Orchestrator:

	@pytest.mark.cpu
	def test_single_entry(self):
		orch = S3Orchestrator(gap_threshold=64)
		entries = [("shard/000000", 512, 1000)]
		chain_ids = [("1abc", "A")]
		groups = orch.plan_reads(entries, chain_ids)

		assert len(groups) == 1
		g = groups[0]
		assert g.shard_id == "shard/000000"
		assert g.byte_start == 512
		assert g.byte_end == 1512
		assert len(g.entries) == 1
		assert g.entries[0].pdb == "1abc"
		assert g.entries[0].chain == "A"

	@pytest.mark.cpu
	def test_coalesces_nearby_entries(self):
		"""entries within gap_threshold should be merged into one ReadGroup"""
		orch = S3Orchestrator(gap_threshold=100)
		entries = [
			("shard/000000", 100, 50),
			("shard/000000", 200, 50),  # gap of 50 < 100
		]
		chain_ids = [("1abc", "A"), ("1abc", "B")]
		groups = orch.plan_reads(entries, chain_ids)

		assert len(groups) == 1
		assert groups[0].byte_start == 100
		assert groups[0].byte_end == 250
		assert len(groups[0].entries) == 2

	@pytest.mark.cpu
	def test_splits_distant_entries(self):
		"""entries with large gaps should become separate ReadGroups"""
		orch = S3Orchestrator(gap_threshold=10)
		entries = [
			("shard/000000", 100, 50),
			("shard/000000", 1000, 50),  # gap of 850 > 10
		]
		chain_ids = [("1abc", "A"), ("2def", "A")]
		groups = orch.plan_reads(entries, chain_ids)

		assert len(groups) == 2

	@pytest.mark.cpu
	def test_groups_by_shard(self):
		"""entries in different shards are never coalesced"""
		orch = S3Orchestrator(gap_threshold=10000)
		entries = [
			("shard/000000", 100, 50),
			("shard/000001", 100, 50),
		]
		chain_ids = [("1abc", "A"), ("2def", "A")]
		groups = orch.plan_reads(entries, chain_ids)

		assert len(groups) == 2
		shard_ids = {g.shard_id for g in groups}
		assert shard_ids == {"shard/000000", "shard/000001"}

	@pytest.mark.cpu
	def test_sorts_by_offset(self):
		"""entries should be sorted by offset before coalescing"""
		orch = S3Orchestrator(gap_threshold=200)
		# feed entries out of order
		entries = [
			("shard/000000", 500, 50),
			("shard/000000", 100, 50),
			("shard/000000", 300, 50),
		]
		chain_ids = [("1abc", "A"), ("2def", "A"), ("3ghi", "A")]
		groups = orch.plan_reads(entries, chain_ids)

		assert len(groups) == 1
		assert groups[0].byte_start == 100
		assert groups[0].byte_end == 550
		# entries should be in offset order
		offsets = [e.offset for e in groups[0].entries]
		assert offsets == [100, 300, 500]


# ── Sampler ──

class TestSampler:

	def _make_index(self, n_clusters: int = 5, samples_per_cluster: int = 3) -> pl.DataFrame:
		rows = []
		for c in range(n_clusters):
			for s in range(samples_per_cluster):
				rows.append({
					IndexCol.PDB: f"pdb{c:03d}",
					IndexCol.CHAIN: chr(65 + s),
					IndexCol.SHARD_ID: f"shard/{c // 2:06d}",
					IndexCol.OFFSET: 512 + s * 2048,
					IndexCol.SIZE: 1024,
					"foldseek_70": f"cluster_{c}",
				})
		return pl.DataFrame(rows)

	@pytest.mark.cpu
	def test_deterministic(self):
		"""same seed + epoch produces identical read plans"""
		index = self._make_index()
		s1 = Sampler(index, "foldseek_70", base_seed=42)
		s2 = Sampler(index, "foldseek_70", base_seed=42)

		groups1 = s1.sample(num_workers=1, worker_id=0)
		groups2 = s2.sample(num_workers=1, worker_id=0)

		assert len(groups1) == len(groups2)
		for g1, g2 in zip(groups1, groups2):
			assert g1.shard_id == g2.shard_id
			assert g1.byte_start == g2.byte_start

	@pytest.mark.cpu
	def test_epoch_changes_output(self):
		"""different epochs produce different orderings"""
		index = self._make_index(n_clusters=20)
		s = Sampler(index, "foldseek_70", base_seed=42)

		groups_e0 = s.sample(num_workers=1, worker_id=0)
		s.epoch = 1
		groups_e1 = s.sample(num_workers=1, worker_id=0)

		# at least some groups should be in different order
		shard_ids_e0 = [(g.shard_id, g.byte_start) for g in groups_e0]
		shard_ids_e1 = [(g.shard_id, g.byte_start) for g in groups_e1]
		assert shard_ids_e0 != shard_ids_e1

	@pytest.mark.cpu
	def test_val_test_ignores_epoch(self):
		"""val/test samplers produce the same plan regardless of epoch"""
		index = self._make_index()
		s = Sampler(index, "foldseek_70", base_seed=42, is_val_or_test=True)

		groups_e0 = s.sample(num_workers=1, worker_id=0)
		s.epoch = 5
		groups_e5 = s.sample(num_workers=1, worker_id=0)

		shard_ids_e0 = [(g.shard_id, g.byte_start) for g in groups_e0]
		shard_ids_e5 = [(g.shard_id, g.byte_start) for g in groups_e5]
		assert shard_ids_e0 == shard_ids_e5

	@pytest.mark.cpu
	def test_worker_partitioning(self):
		"""groups are non-overlapping across workers"""
		index = self._make_index(n_clusters=10)
		s = Sampler(index, "foldseek_70", base_seed=42)

		groups_w0 = s.sample(num_workers=2, worker_id=0)
		groups_w1 = s.sample(num_workers=2, worker_id=1)

		# no overlap
		keys_w0 = {(g.shard_id, g.byte_start) for g in groups_w0}
		keys_w1 = {(g.shard_id, g.byte_start) for g in groups_w1}
		assert keys_w0.isdisjoint(keys_w1)

	@pytest.mark.cpu
	def test_len(self):
		index = self._make_index(n_clusters=7)
		s = Sampler(index, "foldseek_70", base_seed=42)
		assert len(s) == 7


# ── PDBData ──

class TestPDBData:

	@pytest.mark.cpu
	def test_sample_asmb_basic(self, protein_dict_builder):
		"""basic assembly sampling with a single chain"""
		pdb_dict = protein_dict_builder(chains=[("A", "ACDEFG")])
		pdb_data = PDBData(
			pdb_dict,
			min_seq_size=1,
			max_seq_size=1024,
			homo_thresh=0.7,
			asymmetric_units_only=False,
		)

		asmb = pdb_data.sample_asmb("A")
		assert asmb is not None
		assert isinstance(asmb, Assembly)
		assert len(asmb) > 0

	@pytest.mark.cpu
	def test_sample_asmb_multi_chain(self, protein_dict_builder):
		"""assembly with multiple chains includes all chains"""
		pdb_dict = protein_dict_builder(chains=[("A", "ACDEFG"), ("B", "HIKLMN")])
		pdb_data = PDBData(
			pdb_dict,
			min_seq_size=1,
			max_seq_size=1024,
			homo_thresh=0.7,
			asymmetric_units_only=False,
		)

		asmb = pdb_data.sample_asmb("A")
		assert asmb is not None
		# both chains should contribute (6+6 = 12 residues total before any xform)
		assert asmb.labels.shape[0] == 12

	@pytest.mark.cpu
	def test_sample_asmb_missing_chain(self, protein_dict_builder):
		"""sampling a chain that doesn't exist returns None"""
		pdb_dict = protein_dict_builder(chains=[("A", "ACDEFG")])
		pdb_data = PDBData(
			pdb_dict,
			min_seq_size=1,
			max_seq_size=1024,
			homo_thresh=0.7,
			asymmetric_units_only=False,
		)

		asmb = pdb_data.sample_asmb("Z")
		assert asmb is None

	@pytest.mark.cpu
	def test_sample_asmb_with_tm_scores(self, protein_dict_builder):
		"""homo_chains are identified from CHAIN_TM_SCORES"""
		pdb_dict = protein_dict_builder(chains=[("A", "ACDEFG"), ("B", "HIKLMN")])
		n_chains = 2
		# create tm scores (N x N): A is homo to B (score 0.9), B is homo to A
		tm = np.zeros((n_chains, n_chains), dtype=np.float32)
		tm[0, 0] = 1.0  # self
		tm[0, 1] = 0.9  # A->B high
		tm[1, 0] = 0.9  # B->A high
		tm[1, 1] = 1.0  # self
		pdb_dict[ProteinKey.CHAIN_TM_SCORES] = tm

		pdb_data = PDBData(
			pdb_dict,
			min_seq_size=1,
			max_seq_size=1024,
			homo_thresh=0.7,
			asymmetric_units_only=False,
		)

		asmb = pdb_data.sample_asmb("A")
		assert asmb is not None

	@pytest.mark.cpu
	def test_asymmetric_units_only(self, protein_dict_builder):
		"""asymmetric_units_only forces identity xform"""
		pdb_dict = protein_dict_builder(chains=[("A", "ACDEFG")])
		# add a non-identity xform to the assembly
		xform = np.stack([np.eye(4, dtype=np.float32)] * 3, axis=0)
		xform[1, :3, 3] = [10, 0, 0]
		xform[2, :3, 3] = [0, 10, 0]
		pdb_dict[ProteinKey.ASSEMBLIES][0][ProteinKey.ASMB_XFORMS] = xform

		pdb_data = PDBData(
			pdb_dict,
			min_seq_size=1,
			max_seq_size=1024,
			homo_thresh=0.7,
			asymmetric_units_only=True,
		)

		asmb = pdb_data.sample_asmb("A")
		assert asmb is not None
		# should only have identity xform (1 copy)
		assert asmb.asmb_xform.shape[0] == 1
		np.testing.assert_array_equal(asmb.asmb_xform[0], np.eye(4))


# ── Data iterator (integration) ──

class TestDataIterator:

	def _make_pdb_dict(self):
		"""create a minimal pdb_dict for testing"""
		rng = np.random.default_rng(42)
		seq = "ACDEFG"
		L = len(seq)
		chains_data = {
			"A": {
				ChainKey.SEQUENCE: seq,
				ChainKey.COORDS: rng.standard_normal((L, 14, 3)).astype(np.float32),
				ChainKey.ATOM_MASK: np.ones((L, 14), dtype=bool),
				ChainKey.BFACTOR: rng.uniform(10, 50, (L, 14)).astype(np.float32),
				ChainKey.PLDDT: np.full(L, 0.9, dtype=np.float32),
				ChainKey.OCCUPANCY: np.ones((L, 14), dtype=np.float32),
			},
		}
		return {
			ProteinKey.CHAINS: chains_data,
			ProteinKey.ASSEMBLIES: [{
				ProteinKey.CHAINS: ["A"],
				ProteinKey.ASMB_XFORMS: np.eye(4, dtype=np.float32)[np.newaxis],
			}],
			ProteinKey.RESOLUTION: 2.0,
			ProteinKey.METHOD: "X-RAY DIFFRACTION",
			ProteinKey.DEPOSIT_DATE: "2020-01-01",
			ProteinKey.SOURCE: "rcsb",
			ProteinKey.MEAN_PLDDT: 0.9,
			ProteinKey.PTM: 0.9,
		}

	def _make_index(self, n: int = 10) -> pl.DataFrame:
		rows = []
		for i in range(n):
			rows.append({
				IndexCol.PDB: f"pdb{i:03d}",
				IndexCol.CHAIN: "A",
				IndexCol.SHARD_ID: f"shard/{i // 5:06d}",
				IndexCol.OFFSET: 512 + i * 2048,
				IndexCol.SIZE: 1024,
				"foldseek_70": f"cluster_{i}",
			})
		return pl.DataFrame(rows)

	@pytest.mark.cpu
	def test_data_iter_produces_batches(self):
		"""mock S3Reader and verify Data produces DataBatch objects"""
		from proteus.data.data_utils import S3Reader, DataBatch
		from proteus.data.data_loader import Data
		from proteus.data.construct_registry import ConstructRegistry, ConstructFunctionNames

		pdb_dict = self._make_pdb_dict()
		index = self._make_index(n=10)
		sampler = Sampler(index, "foldseek_70", base_seed=42)

		# mock S3Reader to return our pdb_dict for any group
		mock_reader = MagicMock(spec=S3Reader)
		def mock_read_group(group):
			return [(entry, pdb_dict) for entry in group.entries]
		mock_reader.read_group.side_effect = mock_read_group

		# need construct registry set
		ConstructRegistry.set_construct_function(ConstructFunctionNames.PROTEUS)

		try:
			data = Data(
				sampler=sampler,
				s3_reader=mock_reader,
				batch_tokens=128,
				buffer_size=4,
				min_seq_size=1,
				max_seq_size=64,
				homo_thresh=0.7,
				asymmetric_units_only=False,
			)

			# mock get_worker_info to return None (single process)
			with patch("proteus.data.data_loader.get_worker_info", return_value=None):
				batches = list(data)

			assert len(batches) > 0
			for batch in batches:
				assert isinstance(batch, DataBatch)
				assert not batch.is_empty
		finally:
			ConstructRegistry.unset_construct_function()

	@pytest.mark.cpu
	def test_data_resume_skips_steps(self):
		"""setting resume_step skips earlier groups"""
		from proteus.data.data_utils import S3Reader, DataBatch
		from proteus.data.data_loader import Data
		from proteus.data.construct_registry import ConstructRegistry, ConstructFunctionNames

		pdb_dict = self._make_pdb_dict()
		index = self._make_index(n=10)
		sampler = Sampler(index, "foldseek_70", base_seed=42)

		mock_reader = MagicMock(spec=S3Reader)
		def mock_read_group(group):
			return [(entry, pdb_dict) for entry in group.entries]
		mock_reader.read_group.side_effect = mock_read_group

		ConstructRegistry.set_construct_function(ConstructFunctionNames.PROTEUS)

		try:
			# full run
			data_full = Data(
				sampler=sampler,
				s3_reader=mock_reader,
				batch_tokens=128,
				buffer_size=4,
				min_seq_size=1,
				max_seq_size=64,
				homo_thresh=0.7,
				asymmetric_units_only=False,
			)
			with patch("proteus.data.data_loader.get_worker_info", return_value=None):
				batches_full = list(data_full)

			# resumed run — skip first few steps
			data_resume = Data(
				sampler=sampler,
				s3_reader=mock_reader,
				batch_tokens=128,
				buffer_size=4,
				min_seq_size=1,
				max_seq_size=64,
				homo_thresh=0.7,
				asymmetric_units_only=False,
			)
			data_resume.set_resume_step(5)
			with patch("proteus.data.data_loader.get_worker_info", return_value=None):
				batches_resume = list(data_resume)

			# resumed run should produce fewer or equal batches
			assert len(batches_resume) <= len(batches_full)
		finally:
			ConstructRegistry.unset_construct_function()
