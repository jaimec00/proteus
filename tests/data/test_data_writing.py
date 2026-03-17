import asyncio
import io
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import AsyncMock

from proteus.data.data_constants import ChainKey, ProteinKey, DataSource, IndexCol, DataPath
from proteus.data.downloads.proteus_dataset.data_writing import (
	ShardWriter, _serialize_pdb_blob, _deserialize_pdb_blob,
)


# ---------------------------------------------------------------------------
# TestSerializeDeserialize
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestSerializeDeserialize:

	def test_roundtrip(self, protein_dict_builder):
		data = protein_dict_builder(chains=[("A", "ACDEFG"), ("B", "GGGG")])

		# add chain similarity arrays
		C = len(data[ProteinKey.CHAINS])
		data[ProteinKey.CHAIN_TM_SCORES] = np.eye(C, dtype=np.float32)
		data[ProteinKey.CHAIN_SEQ_IDENTITY] = np.eye(C, dtype=np.float32)

		blob = _serialize_pdb_blob("1abc", data)
		restored = _deserialize_pdb_blob(blob)

		# string fields
		assert restored[ProteinKey.METHOD] == data[ProteinKey.METHOD]
		assert restored[ProteinKey.DEPOSIT_DATE] == data[ProteinKey.DEPOSIT_DATE]
		assert restored[ProteinKey.SOURCE] == data[ProteinKey.SOURCE]
		assert restored[ProteinKey.RESOLUTION] == data[ProteinKey.RESOLUTION]

		# nan fields
		assert np.isnan(restored[ProteinKey.MEAN_PLDDT])
		assert np.isnan(restored[ProteinKey.PTM])

		# per-chain arrays
		for cid in ("A", "B"):
			orig = data[ProteinKey.CHAINS][cid]
			rest = restored[ProteinKey.CHAINS][cid]
			assert rest[ChainKey.SEQUENCE] == orig[ChainKey.SEQUENCE]
			np.testing.assert_allclose(rest[ChainKey.COORDS], orig[ChainKey.COORDS], atol=1e-5)
			np.testing.assert_array_equal(rest[ChainKey.ATOM_MASK], orig[ChainKey.ATOM_MASK])

		# chain similarity arrays roundtrip
		np.testing.assert_allclose(restored[ProteinKey.CHAIN_TM_SCORES], data[ProteinKey.CHAIN_TM_SCORES])
		np.testing.assert_allclose(restored[ProteinKey.CHAIN_SEQ_IDENTITY], data[ProteinKey.CHAIN_SEQ_IDENTITY])

	def test_roundtrip_with_assemblies(self, protein_dict_builder):
		data = protein_dict_builder(chains=[("A", "ACDEF"), ("B", "GGGGG")])
		# add a non-identity transform
		xform = np.array([
			[[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]],
			[[0, 1, 0, 5], [1, 0, 0, 5], [0, 0, 1, 5], [0, 0, 0, 1]],
		], dtype=np.float32)
		data[ProteinKey.ASSEMBLIES] = [[{
			ProteinKey.CHAINS: ["A", "B"],
			ProteinKey.ASMB_XFORMS: xform,
		}]]

		blob = _serialize_pdb_blob("2xyz", data)
		restored = _deserialize_pdb_blob(blob)

		assert len(restored[ProteinKey.ASSEMBLIES]) == 1
		assert len(restored[ProteinKey.ASSEMBLIES][0]) == 1
		np.testing.assert_allclose(
			restored[ProteinKey.ASSEMBLIES][0][0][ProteinKey.ASMB_XFORMS], xform, atol=1e-5,
		)


# ---------------------------------------------------------------------------
# TestShardWriter
# ---------------------------------------------------------------------------

@pytest.mark.cpu
@pytest.mark.asyncio
class TestShardWriter:

	@staticmethod
	def _make_writer(s3_client, checkpoint_path, shard_size_bytes=10 * 1024 * 1024, resume_shard_id=0):
		from cloudpathlib import S3Path
		return ShardWriter(
			s3_prefix=S3Path("s3://test-bucket"),
			shard_size_bytes=shard_size_bytes,
			source=DataSource.EXPERIMENTAL,
			s3_client=s3_client,
			checkpoint_path=checkpoint_path,
			resume_shard_id=resume_shard_id,
		)

	@staticmethod
	def _make_blob(pdb_id: str, protein_dict_builder) -> tuple[bytes, list[str], dict]:
		data = protein_dict_builder(chains=[("A", "ACDEF")])
		blob = _serialize_pdb_blob(pdb_id, data)
		chain_ids = list(data[ProteinKey.CHAINS].keys())
		meta = {
			ProteinKey.RESOLUTION: data[ProteinKey.RESOLUTION],
			ProteinKey.METHOD: data[ProteinKey.METHOD],
			ProteinKey.DEPOSIT_DATE: data[ProteinKey.DEPOSIT_DATE],
			ProteinKey.SOURCE: data[ProteinKey.SOURCE],
			ProteinKey.MEAN_PLDDT: data[ProteinKey.MEAN_PLDDT],
			ProteinKey.PTM: data[ProteinKey.PTM],
		}
		return blob, chain_ids, meta

	async def test_add_and_finalize(self, tmp_path, protein_dict_builder):
		"""3 entries with large shard size -> 1 tar in S3"""
		s3_client = AsyncMock()
		checkpoint = tmp_path / "checkpoint.jsonl"
		writer = self._make_writer(s3_client, checkpoint)

		for i in range(3):
			blob, chain_ids, meta = self._make_blob(f"pdb{i}", protein_dict_builder)
			writer.add(f"pdb{i}", blob, chain_ids, meta)

		rows = await writer.finalize()

		assert len(rows) == 3
		assert s3_client.put_object.call_count == 1
		assert checkpoint.exists()
		ckpt_lines = checkpoint.read_text().strip().splitlines()
		assert len(ckpt_lines) == 3

	async def test_shard_rotation(self, tmp_path, protein_dict_builder):
		"""shard_size=1 -> each entry triggers a new shard"""
		s3_client = AsyncMock()
		checkpoint = tmp_path / "checkpoint.jsonl"
		writer = self._make_writer(s3_client, checkpoint, shard_size_bytes=1)

		for i in range(3):
			blob, chain_ids, meta = self._make_blob(f"pdb{i}", protein_dict_builder)
			writer.add(f"pdb{i}", blob, chain_ids, meta)

		rows = await writer.finalize()

		assert len(rows) == 3
		# each entry triggers rotation plus finalize flushes empty (or not)
		# with shard_size=1, after each add the shard flushes
		assert s3_client.put_object.call_count == 3
		shard_ids = {row[IndexCol.SHARD_ID] for row in rows}
		assert len(shard_ids) == 3

	async def test_resume(self, tmp_path, protein_dict_builder):
		"""resume_shard_id=5 -> new shard starts at 5"""
		s3_client = AsyncMock()
		checkpoint = tmp_path / "checkpoint.jsonl"
		writer = self._make_writer(s3_client, checkpoint, resume_shard_id=5)

		blob, chain_ids, meta = self._make_blob("pdb0", protein_dict_builder)
		writer.add("pdb0", blob, chain_ids, meta)
		rows = await writer.finalize()

		assert rows[0][IndexCol.SHARD_ID] == f"{DataSource.EXPERIMENTAL}/000005"
