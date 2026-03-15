from __future__ import annotations

from torch.utils.data import get_worker_info
import torch

from bisect import insort, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import random
import logging

import polars as pl

from proteus.static.constants import seq_2_lbls, aa_2_lbl
from proteus.types import A, T, Float, Int, Bool, List, Dict, Tuple, Generator, Optional, Any
from proteus.data.construct_registry import ConstructRegistry, InputNames
from proteus.data.data_constants import DataPath, IndexCol, ProteinKey, ChainKey
from proteus.data.downloads.proteus_dataset.data_writing import _deserialize_pdb_blob
from proteus.utils.s3_utils import read_byte_range

logger = logging.getLogger(__name__)

BIG_PRIME = 6_364_136_223_846_793_005


# --- S3 orchestrator: groups and coalesces byte ranges by shard ---

@dataclass
class BlobEntry:
	pdb: str
	chain: str
	offset: int
	size: int

@dataclass
class ReadGroup:
	shard_id: str
	byte_start: int
	byte_end: int
	entries: list[BlobEntry]


class S3Orchestrator:
	"""pure computation — groups entries by shard and coalesces nearby byte ranges"""

	def __init__(self, gap_threshold: int = 64 * 1024):
		self._gap_threshold = gap_threshold

	def plan_reads(
		self,
		entries: list[tuple[str, int, int]],
		chain_ids: list[tuple[str, str]],
	) -> list[ReadGroup]:
		# group by shard_id
		by_shard: dict[str, list[tuple[BlobEntry, int]]] = defaultdict(list)
		for (shard_id, offset, size), (pdb, chain) in zip(entries, chain_ids):
			entry = BlobEntry(pdb=pdb, chain=chain, offset=offset, size=size)
			by_shard[shard_id].append((entry, offset))

		groups: list[ReadGroup] = []
		for shard_id, items in by_shard.items():
			# sort by offset
			items.sort(key=lambda x: x[1])

			# coalesce entries whose gap is below threshold
			cur_entries: list[BlobEntry] = [items[0][0]]
			cur_start = items[0][0].offset
			cur_end = items[0][0].offset + items[0][0].size

			for entry, _ in items[1:]:
				entry_end = entry.offset + entry.size
				if entry.offset - cur_end <= self._gap_threshold:
					cur_entries.append(entry)
					cur_end = max(cur_end, entry_end)
				else:
					groups.append(ReadGroup(
						shard_id=shard_id,
						byte_start=cur_start,
						byte_end=cur_end,
						entries=cur_entries,
					))
					cur_entries = [entry]
					cur_start = entry.offset
					cur_end = entry_end

			groups.append(ReadGroup(
				shard_id=shard_id,
				byte_start=cur_start,
				byte_end=cur_end,
				entries=cur_entries,
			))

		return groups


# --- S3 reader: fetches byte ranges, deserializes blobs ---

class S3Reader:
	"""stateless utility that fetches byte ranges from S3 and deserializes protein blobs"""

	def __init__(self, s3_bucket: str):
		self._bucket = s3_bucket
		self._client = None

	def _get_client(self):
		# lazy init per worker (boto3 clients are not fork-safe)
		if self._client is None:
			import boto3
			from proteus.utils.s3_utils import REGION
			self._client = boto3.client("s3", region_name=REGION)
		return self._client

	def read_group(self, group: ReadGroup) -> list[tuple[BlobEntry, dict]]:
		client = self._get_client()
		key = f"{DataPath.SHARDS}/{group.shard_id}.tar"
		raw = read_byte_range(self._bucket, key, group.byte_start, group.byte_end, client=client)

		# deduplicate by (pdb, offset) — same blob shared by multiple chains
		seen: dict[tuple[str, int], dict] = {}
		results: list[tuple[BlobEntry, dict]] = []

		for entry in group.entries:
			dedup_key = (entry.pdb, entry.offset)
			if dedup_key not in seen:
				local_start = entry.offset - group.byte_start
				blob_bytes = raw[local_start:local_start + entry.size]
				seen[dedup_key] = _deserialize_pdb_blob(blob_bytes)
			results.append((entry, seen[dedup_key]))

		return results


# --- PDBData: thin wrapper around a deserialized protein dict ---

class PDBData:
	"""wraps a deserialized protein dict, provides assembly sampling"""

	def __init__(
		self,
		pdb_dict: dict,
		min_seq_size: int,
		max_seq_size: int,
		homo_thresh: float,
		asymmetric_units_only: bool,
	):
		self._pdb_dict = pdb_dict
		self._min_seq_size = min_seq_size
		self._max_seq_size = max_seq_size
		self._homo_thresh = homo_thresh
		self._asymmetric_units_only = asymmetric_units_only

		# build chain name -> index mapping from meta chain list
		chain_list = list(pdb_dict[ProteinKey.CHAINS].keys())
		self._chain_to_idx = {c: i for i, c in enumerate(chain_list)}

	def sample_asmb(self, chain: str) -> Optional[Assembly]:
		chains_data = self._pdb_dict[ProteinKey.CHAINS]

		if chain not in chains_data:
			return None

		# find assemblies containing this chain
		assemblies = self._pdb_dict[ProteinKey.ASSEMBLIES]
		valid_asmb_ids = [
			i for i, a in enumerate(assemblies)
			if chain in a[ProteinKey.CHAINS]
		]

		# pick an assembly (or use chain alone with identity xform)
		if valid_asmb_ids:
			asmb_id = random.choice(valid_asmb_ids)
			asmb = assemblies[asmb_id]
			asmb_chains = list(asmb[ProteinKey.CHAINS])
		else:
			asmb_id = -1
			asmb_chains = [chain]

		# shuffle but keep target chain first
		asmb_chains = [chain] + random.sample(
			[c for c in asmb_chains if c != chain],
			k=len(asmb_chains) - 1,
		)

		labels_list = []
		coords_list = []
		atom_mask_list = []
		chain_info = []
		trgt_chain_idx = self._chain_to_idx[chain]

		for asmb_chain in asmb_chains:
			if asmb_chain not in chains_data:
				continue
			cd = chains_data[asmb_chain]

			seq_labels = seq_2_lbls(cd[ChainKey.SEQUENCE])
			coords = cd[ChainKey.COORDS].astype(np.float32)
			mask = cd[ChainKey.ATOM_MASK].astype(bool)

			# replace nans
			coords[np.isnan(coords)] = 0.0

			labels_list.append(seq_labels)
			coords_list.append(coords)
			atom_mask_list.append(mask)
			chain_info.append((self._chain_to_idx[asmb_chain], mask.shape[0]))

		if not labels_list:
			return None

		labels = np.concatenate(labels_list, axis=0)
		coords = np.concatenate(coords_list, axis=0)
		atom_mask = np.concatenate(atom_mask_list, axis=0)

		# homo chains from tm score matrix (shape: N x N)
		tm_scores = self._pdb_dict.get(ProteinKey.CHAIN_TM_SCORES)
		if tm_scores is not None and trgt_chain_idx < tm_scores.shape[0]:
			homo_chains = np.arange(len(self._chain_to_idx))[
				tm_scores[trgt_chain_idx, :] >= self._homo_thresh
			]
		else:
			homo_chains = np.array([trgt_chain_idx], dtype=np.intp)

		# assembly transform
		if self._asymmetric_units_only or asmb_id == -1:
			asmb_xform = np.eye(4, dtype=np.float32)[np.newaxis]
		else:
			asmb_xform = asmb[ProteinKey.ASMB_XFORMS]

		return Assembly(
			coords, labels, atom_mask,
			chain_info, trgt_chain_idx, homo_chains,
			asmb_xform, self._max_seq_size, self._min_seq_size,
		)


# --- Sampler: deterministic, epoch-aware, worker-partitioned ---

class Sampler:
	"""owns the index and S3Orchestrator — generates deterministic read plans per epoch"""

	def __init__(
		self,
		index: pl.DataFrame,
		cluster_col: str,
		base_seed: int,
		is_val_or_test: bool = False,
	):
		self._index = index
		self._cluster_col = cluster_col
		self._base_seed = base_seed
		self._is_val_or_test = is_val_or_test
		self._epoch = 0
		self._orchestrator = S3Orchestrator()

	@property
	def epoch(self) -> int:
		return self._epoch

	@epoch.setter
	def epoch(self, value: int) -> None:
		self._epoch = value

	def sample(self, num_workers: int, worker_id: int) -> list[ReadGroup]:
		# deterministic seed
		if self._is_val_or_test:
			seed = self._base_seed
		else:
			seed = (self._base_seed + self._epoch * BIG_PRIME) % (2**64)

		rng = np.random.default_rng(seed)

		# sample one row per cluster (sort first for determinism across runs)
		sample_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
		sampled = self._index.sort(self._cluster_col, IndexCol.PDB, IndexCol.CHAIN).group_by(
			self._cluster_col, maintain_order=True,
		).agg(
			pl.all().sample(n=1, seed=sample_seed)
		).explode(pl.all().exclude(self._cluster_col))

		# extract columns for orchestrator
		entries = list(zip(
			sampled[IndexCol.SHARD_ID].to_list(),
			sampled[IndexCol.OFFSET].to_list(),
			sampled[IndexCol.SIZE].to_list(),
		))
		chain_ids = list(zip(
			sampled[IndexCol.PDB].to_list(),
			sampled[IndexCol.CHAIN].to_list(),
		))

		# plan coalesced reads
		groups = self._orchestrator.plan_reads(entries, chain_ids)

		# deterministic shuffle
		shuffle_order = rng.permutation(len(groups)).tolist()
		groups = [groups[i] for i in shuffle_order]

		# partition across workers
		return groups[worker_id::num_workers]

	def __len__(self) -> int:
		return len(self._index[self._cluster_col].unique())


# --- BatchBuilder ---

class BatchBuilder:
	def __init__(self, batch_tokens: int, buffer_size: int) -> None:
		self._buffer: List[Assembly] = []
		self._cur_batch: List[Assembly] = []
		self._cur_tokens: int = 0
		self._buffer_size: int = buffer_size
		self._batch_tokens: int = batch_tokens

	def add_sample(self, sample: Assembly) -> Generator[DataBatch, None, None]:
		self._add_buffer(sample)

		if self._buffer_full():
			if self._batch_full():
				yield from self._yield_batch()
				self._clear_batch()
			self._add_batch()

	def drain_buffer(self) -> Generator[DataBatch, None, None]:
		while self._buffer:
			if self._batch_full():
				yield from self._yield_batch()
				self._clear_batch()
			self._add_batch()

		if self._cur_batch:
			yield from self._yield_batch()

	def _yield_batch(self):
		data_batch = DataBatch(self._cur_batch)
		if not data_batch.is_empty:
			yield data_batch

	def _add_buffer(self, asmb: Assembly) -> None:
		insort(self._buffer, asmb, key=len)

	def _add_batch(self) -> None:
		remaining = self._batch_tokens - self._cur_tokens
		idx = bisect_right(self._buffer, remaining, key=len) - 1
		assert idx >= 0, f"_add_batch called but nothing fits: {remaining=}, smallest={len(self._buffer[0])}"
		sampled_asmb = self._buffer.pop(idx)
		self._cur_batch.append(sampled_asmb)
		self._cur_tokens += len(sampled_asmb)

	def _clear_batch(self) -> None:
		self._cur_batch.clear()
		self._cur_tokens = 0

	def _buffer_full(self) -> bool:
		return len(self._buffer) >= self._buffer_size

	def _batch_full(self) -> bool:
		return (self._cur_tokens + (len(self._buffer[0]) if self._buffer else 0) > self._batch_tokens) and self._cur_tokens > 0


# --- DataBatch ---

class DataBatch:

	@torch.no_grad()
	def __init__(self, batch_list: List[Assembly]) -> None:

		if not batch_list:
			raise ValueError()

		batch_dict = defaultdict(list)
		seq_lens = []
		tot_tokens = 0

		for idx, asmb in enumerate(batch_list):
			constructed = asmb.construct()
			if constructed is None:
				continue
			for key, value in constructed.items():
				tokens = value.size(0)
				if not tokens:
					break
				if key == "labels":
					seq_lens.append(tokens)
					tot_tokens += tokens
				batch_dict[key].append(value)

		self.is_empty = not seq_lens
		if self.is_empty:
			return

		self.sample_idx = torch.cat([torch.full((i,), idx) for idx, i in enumerate(seq_lens)], dim=0)
		seq_lens = torch.tensor(seq_lens, dtype=torch.int, device="cpu")
		self.max_seqlen = seq_lens.max().item()
		self.samples = seq_lens.size(0)

		self.cu_seqlens = torch.nn.functional.pad(seq_lens.cumsum(dim=0), pad=(1, 0), mode="constant", value=0).int()

		assert InputNames.LABELS in batch_dict
		assert InputNames.LOSS_MASK in batch_dict
		self._tensor_names = list(batch_dict.keys()) + [InputNames.CU_SEQLENS, InputNames.SAMPLE_IDX]
		for tensor_name, tensor_list in batch_dict.items():
			tensor = torch.cat(tensor_list, dim=0)
			setattr(self, tensor_name, tensor)

	@property
	def loss_tokens(self):
		return self.loss_mask.sum()

	@property
	def tokens(self):
		return self.loss_mask.size(0)

	def move_to(self, device: torch.device) -> None:
		if not hasattr(self, "_tensor_names"):
			raise ValueError(f"no tensors!! the data batch is probably empty: {self.is_empty=}")
		for tensor_name in self._tensor_names:
			tensor = getattr(self, tensor_name)
			if isinstance(tensor, T):
				setattr(self, tensor_name, tensor.to(device))

	def __len__(self) -> int:
		return self.tokens


# --- Assembly ---

class Assembly:
	def __init__(
		self,
		coords: Float[A, "L 14 3"], labels: Int[A, "L"], atom_mask: Bool[A, "L 14"],
		chain_info: List[Tuple[int, int]], trgt_chain: int, homo_chains: Int[A, "H"],
		asmb_xform: Float[A, "N 4 4"], max_seq_size: int, min_seq_size: int = 0,
	) -> None:

		self.coords = coords
		self.labels = labels
		self.atom_mask = atom_mask

		self._chain_info = chain_info
		self._trgt_chain = trgt_chain
		self._homo_chains = homo_chains

		self.asmb_xform = asmb_xform
		self._min_seq_size = min_seq_size

		self._crop(max_seq_size)

	@torch.no_grad()
	def construct(self) -> Optional[Dict[str, T]]:

		coords = torch.from_numpy(self.coords).float()
		labels = torch.from_numpy(self.labels).long()

		# compute seq idxs and chain idxs, note that need to crop in case labels was cropped earlier
		seq_idx = torch.cat([torch.arange(size) for _, size in self._chain_info], dim=0)[:labels.size(0)].long()
		chain_idx = torch.cat([torch.full((size,), idx) for idx, size in self._chain_info], dim=0)[:labels.size(0)].long()

		# make the masks to remove invalid (no coords)
		atom_mask = torch.from_numpy(self.atom_mask).bool()
		valid_mask = atom_mask[..., :3].all(dim=-1)

		coords = coords[valid_mask]
		labels = labels[valid_mask]
		seq_idx = seq_idx[valid_mask]
		chain_idx = chain_idx[valid_mask]

		# perform the xform on coords and adjust the other tensors accordingly
		asmb_xform = torch.from_numpy(self.asmb_xform).float()

		N, A, S = coords.shape
		num_copies = asmb_xform.size(0)

		R = asmb_xform[:, :3, :3]
		T = asmb_xform[:, :3, 3]

		coords = (torch.einsum("bij,raj->brai", R, coords) + T.view(num_copies, 1, 1, 3)).reshape(N * num_copies, A, S)
		labels = labels.repeat(num_copies)
		seq_idx = seq_idx.repeat(num_copies)
		chain_idx = chain_idx.repeat(num_copies)

		trgt_mask = chain_idx == self._trgt_chain
		homo_mask = torch.isin(chain_idx, torch.from_numpy(self._homo_chains))
		caa_mask = labels != aa_2_lbl("X")

		result = ConstructRegistry.construct(
			coords, labels, seq_idx, chain_idx, trgt_mask, homo_mask, caa_mask, atom_mask
		)

		if result["labels"].size(0) < self._min_seq_size:
			return None

		return result

	@torch.no_grad()
	def _crop(self, max_seq_size: int) -> None:

		N, A, S = self.coords.shape
		num_copies = min(max_seq_size // N, self.asmb_xform.shape[0])

		if num_copies == 0:
			self.coords = self.coords[:max_seq_size, :, :]
			self.labels = self.labels[:max_seq_size]
			self.atom_mask = self.atom_mask[:max_seq_size, :]
			self.asmb_xform = np.expand_dims(np.eye(4), 0)
		else:
			self.asmb_xform = self.asmb_xform[:num_copies, :, :]

	def __len__(self) -> int:
		return self.labels.shape[0] * self.asmb_xform.shape[0]
