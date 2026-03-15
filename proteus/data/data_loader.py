

from __future__ import annotations

from torch.utils.data import IterableDataset, DataLoader
import torch

from typing import Generator, Tuple, Optional, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from functools import partial
from cloudpathlib import S3Path
import pyarrow as pa
import polars as pl
import logging
import re

from proteus.data.data_utils import Assembly, PDBCache, BatchBuilder, DataBatch, Sampler, PDBCacheCfg, SamplerCfg, BatchBuilderCfg
from proteus.data.data_constants import DataPath, IndexCol, ClusteringMethod, TEST_SEED, VAL_SEED, UINT64
from proteus.utils.s3_utils import REGION
from polars_xxhash import stable_hash

logger = logging.getLogger(__name__)

def _validate_cluster_col(cluster_col: str) -> None:
	"""validate that cluster_col is '{method}_{two_digit_number}' with a known method"""
	m = re.fullmatch(r"(.+)_(\d{2,3})", cluster_col)
	if m is None:
		err = f"cluster_col must be '{{method}}_{{threshold}}', got '{cluster_col}'"
		logger.error(err)
		raise RuntimeError(err)
	method_str, _threshold = m.group(1), m.group(2)
	ClusteringMethod(method_str) # raises ValueError if not a valid member

@dataclass
class DataSplitInfoCfg:
	num_clusters: int = -1
	max_samples_per_cluster: int = -1

@dataclass
class DataSplitCfg:
	train: DataSplitInfoCfg = field(default_factory=DataSplitInfoCfg)
	val: DataSplitInfoCfg = field(default_factory=DataSplitInfoCfg)
	test: DataSplitInfoCfg = field(default_factory=DataSplitInfoCfg)

@dataclass
class DataFilterCfg:
	max_resolution: float = 3.5
	plddt_thresh: float = 0.70
	ptm_thresh: float = 0.70

@dataclass
class DataHolderCfg:
	s3_bucket: S3Path

	cluster_col: str = "foldseek_70"
	cluster_split_cols: List[str] = field(default_factory=lambda: ["foldseek_70", "mmseqs_30"])
	
	train_val_test_split: List = field(default_factory=lambda: [0.9, 0.05, 0.05])
	split_limit: DataSplitCfg = field(default_factory=DataSplitCfg)

	min_seq_size: int = 16
	max_seq_size: int = 16384

	filters: DataFilterCfg = field(default_factory=DataFilterCfg)
	homo_thresh: float = 0.70

	batch_tokens: int = 16384
	asymmetric_units_only: bool = False
	num_workers: int = 8
	prefetch_factor: int = 2
	rng_seed: int = 42
	buffer_size: int = 32


class DataHolder:

	'''
	hold DataLoader Objects, one each for train, test and val
	multi-chain (experimental structures): https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz
	'''

	def __init__(self, cfg: DataHolderCfg) -> None:

		_validate_cluster_col(cfg.cluster_col)
		for col in cfg.cluster_split_cols:
			_validate_cluster_col(col)

		if cfg.cluster_col not in cfg.cluster_split_cols:
			err = f"cluster_col '{cfg.cluster_col}' must be in cluster_split_cols {cfg.cluster_split_cols}"
			logger.error(err)
			raise RuntimeError(err)

		# define data path and path to pdbs
		s3_bucket = S3Path(cfg.s3_bucket)
		index_path = s3_bucket / DataPath.INDEX
		raw_index = pl.read_parquet(str(index_path), storage_options={"aws_region": REGION})
		logger.info(f"raw index contains {len(raw_index)} samples and {len(raw_index[cfg.cluster_col].unique())} clusters")

		if cfg.cluster_col not in raw_index.columns:
			err = f"cluster column '{cfg.cluster_col}' not found in index (columns: {raw_index.columns})"
			logger.error(err)
			raise RuntimeError(err)
		for col in cfg.cluster_split_cols:
			if col not in raw_index.columns:
				err = f"cluster_split_col '{col}' not found in index (columns: {raw_index.columns})"
				logger.error(err)
				raise RuntimeError(err)

		index = self._apply_filter(raw_index, cfg.filters)
		logger.info(f"filtered index contains {len(index)} samples and {len(index[cfg.cluster_col].unique())} clusters")
		train, val, test = self._get_splits(
			index,
			cfg.train_val_test_split,
			cfg.cluster_split_cols,
		)
		for mode, mode_idx in [
			("train", train), 
			("val", val), 
			("test", test),
		]:
			logger.info(f"raw {mode} index contains {len(mode_idx)} samples and {len(mode_idx[cfg.cluster_col].unique())} clusters")
	
		train_index = self._limit_index(train, cfg.split_limit.train, cfg.cluster_col)
		val_index = self._limit_index(val, cfg.split_limit.val, cfg.cluster_col)
		test_index = self._limit_index(test, cfg.split_limit.test, cfg.cluster_col)

		train_samples, train_clusters = len(train_index), len(train_index[cfg.cluster_col].unique())
		val_samples, val_clusters = len(val_index), len(val_index[cfg.cluster_col].unique())
		test_samples, test_clusters = len(test_index), len(test_index[cfg.cluster_col].unique())

		logger.info(f"final train index contains {train_samples} samples and {train_clusters} clusters")
		logger.info(f"final val index contains {val_samples} samples and {val_clusters} clusters")
		logger.info(f"final test index contains {test_samples} samples and {test_clusters} clusters")

		tot_samples, tot_clusters = (
			train_samples + val_samples + test_samples,
			train_clusters + val_clusters + test_clusters,
		)

		for idx, (mode, mode_samples, mode_clusters) in enumerate([
			("train", train_samples, train_clusters), 
			("val", val_samples, val_clusters), 
			("test", test_samples, test_clusters),
		]):
			logger.info(
				f"\n{mode}:\n"
				f"  target percentage: {cfg.train_val_test_split[idx]*100:.2f}%\n"
				f"  samples percentage: {(mode_samples/tot_samples)*100:.2f}%\n"
				f"  clusters percentage: {(mode_clusters/tot_clusters)*100:.2f}%\n"
			)

		# init_data = partial(
		#     Data,
		#     data_path=pdb_path,
		#     clusters_df=train_info,
		#     num_clusters=samples,
		#     batch_tokens=cfg.batch_tokens,
		#     min_seq_size=cfg.min_seq_size,
		#     max_seq_size=cfg.max_seq_size,
		#     homo_thresh=cfg.homo_thresh,
		#     asymmetric_units_only=cfg.asymmetric_units_only,
		#     buffer_size=cfg.buffer_size,
		#     seed=cfg.rng_seed,
		# )  

		# init_loader = partial(
		#     DataLoader,
		#     batch_size=None, 
		#     num_workers=cfg.num_workers, 
		#     collate_fn=lambda x: x,
		#     prefetch_factor=cfg.prefetch_factor if cfg.num_workers else None, 
		#     persistent_workers=cfg.num_workers>0
		# )

		# # initialize the loaders
		# self.train = init_loader(init_data(train_info, cfg.num_train))
		# self.val = init_loader(init_data(val_info, cfg.num_val))
		# self.test = init_loader(init_data(test_info, cfg.num_test))

	def _limit_index(
		self,
		index: pl.DataFrame,
		split_info: DataSplitInfoCfg,
		cluster_col: str,
	) -> pl.DataFrame:
		lf = index.lazy()
		if split_info.num_clusters != -1:
			top_clusters = (
				lf.select(cluster_col)
				.unique()
				.sort(cluster_col)
				.head(split_info.num_clusters)
			)
			lf = lf.join(top_clusters, on=cluster_col, how="semi")
		if split_info.max_samples_per_cluster != -1:
			lf = lf.group_by(cluster_col, maintain_order=True).head(split_info.max_samples_per_cluster)
		return lf.collect()

	def _apply_filter(
		self,
		index: pl.DataFrame,
		filters: DataFilterCfg,
	) -> pl.DataFrame:
		return (
			index.lazy()
			.filter(pl.col(IndexCol.RESOLUTION) <= filters.max_resolution)
			.filter(pl.col(IndexCol.MEAN_PLDDT) >= filters.plddt_thresh)
			.filter(pl.col(IndexCol.PTM) >= filters.ptm_thresh)
			.collect()
		)

	def _get_splits(
		self,
		index: pl.DataFrame,
		train_val_test_split: List[float],
		cluster_split_cols: List[str],
	) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:

		# compute per-column thresholds adjusted for OR-union across N columns
		# formula: per_col_pct = 1 - (1 - target_pct)^(1/N) so the union ≈ target_pct
		assert sum(train_val_test_split) == 1.0
		_, val_pct, test_pct = train_val_test_split
		val_pct = val_pct / (1 - test_pct)
		n_cols = len(cluster_split_cols)
		adj_test_pct = 1 - (1 - test_pct) ** (1 / n_cols)
		adj_val_pct = 1 - (1 - val_pct) ** (1 / n_cols)
		test_thresh = int(adj_test_pct * UINT64)
		val_thresh = int(adj_val_pct * UINT64)

		_IS_TEST, _SPLIT = "_is_test", "_split"
		_TEST, _VAL, _TRAIN = 0, 1, 2

		# build OR expressions across all cluster columns
		is_test_expr = pl.lit(False)
		for col in cluster_split_cols:
			is_test_expr = is_test_expr | (stable_hash(pl.col(col), seed=TEST_SEED) < test_thresh)

		is_val_expr = pl.lit(False)
		for col in cluster_split_cols:
			is_val_expr = is_val_expr | (stable_hash(pl.col(col), seed=VAL_SEED) < val_thresh)

		# single lazy chain: assign test → filter pdb leakage → assign split → filter pdb leakage → collect
		split_index = (
			index.lazy()
			.with_columns(is_test_expr.alias(_IS_TEST))
			# remove any chain not in test whose pdb id has a chain in test
			.filter(~(~pl.col(_IS_TEST) & pl.col(_IS_TEST).any().over(IndexCol.PDB)))
			.with_columns(
				pl.when(pl.col(_IS_TEST)).then(_TEST)
				.when(is_val_expr).then(_VAL)
				.otherwise(_TRAIN)
				.alias(_SPLIT)
			)
			# remove any chain in train whose pdb id is in val
			.filter(~((pl.col(_SPLIT) == _TRAIN) & (pl.col(_SPLIT) == _VAL).any().over(IndexCol.PDB)))
			.drop(_IS_TEST)
			.collect()
		)

		# split
		train_index = (
			split_index.lazy()
			.filter(pl.col(_SPLIT) == _TRAIN)
			.drop(_SPLIT)
			.collect()
		)                                                                                                                                        
		val_index = (
			split_index.lazy()
			.filter(pl.col(_SPLIT) == _VAL)
			.drop(_SPLIT)
			.collect()
		)           
		test_index = (
			split_index.lazy()
			.filter(pl.col(_SPLIT) == _TEST)
			.drop(_SPLIT)
			.collect()
		)         
		return train_index, val_index, test_index

class Data(IterableDataset):
	def __init__(
		self,
		data_path: Path,
		clusters_df: pd.DataFrame,
		num_clusters: int = -1,
		batch_tokens: int = 16384,
		min_seq_size: int = 16,
		max_seq_size: int = 16384,
		homo_thresh: float = 0.70,
		asymmetric_units_only: bool = False,
		buffer_size: int = 32,
		seed: int = 42,
		epoch: Any = None,
	) -> None:

		super().__init__()

		assert max_seq_size <= batch_tokens, f"max_seq_size ({max_seq_size}) must be <= batch_tokens ({batch_tokens})"

		# keep a cache of pdbs
		self._pdb_cache = PDBCache(PDBCacheCfg(
			pdb_path=data_path,
			min_seq_size=min_seq_size,
			max_seq_size=max_seq_size,
			homo_thresh=homo_thresh,
			asymmetric_units_only=asymmetric_units_only
		))

		# for deterministic and UNIQUE sampling
		self._sampler = Sampler(clusters_df, SamplerCfg(
			num_clusters=num_clusters,
			seed=seed
		))

		# for batch building
		self._batch_builder_config = BatchBuilderCfg(
			batch_tokens=batch_tokens,
			buffer_size=buffer_size
		)

	def _get_asmb(self, row: pd.Series) -> Optional[Assembly]:

		# get pdb and chain name
		pdb, chain = row.CHAINID.split("_")

		# get the data corresponding to this pdb
		pdb_data = self._pdb_cache.get_pdb(pdb)

		# sample an assembly containing this chain
		asmb = pdb_data.sample_asmb(chain)

		return asmb

	def __iter__(self) -> Generator[DataBatch]:

		# sample rows from the df
		sampled_rows = self._sampler.sample_rows()

		# init the batch builder
		batch_builder = BatchBuilder(self._batch_builder_config)

		# iterate through the sampled chains
		for _, row in sampled_rows.iterrows():

			# add the sample, only yields if batch is ready
			sample = self._get_asmb(row)
			if sample is not None:
				yield from batch_builder.add_sample(sample)

		# drain the buffer and yield last batches
		yield from batch_builder.drain_buffer()


	def __len__(self) -> int:
		return len(self._sampler)