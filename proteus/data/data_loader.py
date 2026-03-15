

from __future__ import annotations

from torch.utils.data import IterableDataset, DataLoader
import torch

from typing import Generator, Tuple, Optional, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from functools import partial
from cloudpathlib import S3Path
import pyarrow.parquet as pq
import pyarrow as pa
import polars as pl

from proteus.data.data_utils import Assembly, PDBCache, BatchBuilder, DataBatch, Sampler, PDBCacheCfg, SamplerCfg, BatchBuilderCfg
from proteus.data.data_constants import DataPath, IndexCol, ClusteringMethod, TEST_SEED, VAL_SEED, UINT64
from polars_xxhash import stable_hash

import re

def _validate_cluster_col(cluster_col: str) -> None:
	"""validate that cluster_col is '{method}_{two_digit_number}' with a known method"""
	m = re.fullmatch(r"(.+)_(\d{2,3})", cluster_col)
	assert m is not None, f"cluster_col must be '{{method}}_{{threshold}}', got '{cluster_col}'"
	method_str, _threshold = m.group(1), m.group(2)
	ClusteringMethod(method_str) # raises ValueError if not a valid member

@dataclass
class DataHolderCfg:
    s3_bucket: S3Path
    cluster_col: str = "foldseek_70"
    cluster_split_cols: List[str] = field(default_factory=lambda: ["foldseek_70", "mmseqs_30"])
    train_val_test_split: List = field(default_factory=lambda: [0.9, 0.05, 0.05])

    batch_tokens: int = 16384

    min_seq_size: int = 16
    max_seq_size: int = 16384

    max_resolution: float = 3.5
    homo_thresh: float = 0.70
    plddt_thresh: float = 0.70

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

        assert cfg.cluster_col in cfg.cluster_split_cols, \
            f"cluster_col '{cfg.cluster_col}' must be in cluster_split_cols {cfg.cluster_split_cols}"

        # define data path and path to pdbs
        s3_bucket = S3Path(cfg.s3_bucket)
        index_path = s3_bucket / DataPath.INDEX
        self.raw_index = pl.from_arrow(pq.read_table(index_path))

        assert cfg.cluster_col in self.raw_index.columns, \
            f"cluster column '{cfg.cluster_col}' not found in index (columns: {self.raw_index.columns})"
        for col in cfg.cluster_split_cols:
            assert col in self.raw_index.columns, \
                f"cluster_split_col '{col}' not found in index (columns: {self.raw_index.columns})"

        self.cluster_col = cfg.cluster_col
        self.cluster_split_cols = cfg.cluster_split_cols
        self.index = self._apply_filter(
            self.raw_index,
            cfg.max_resolution,
            cfg.plddt_thresh,
        )
        self.train_index, self.val_index, self.test_index = self._get_splits(
            self.index,
            cfg.train_val_test_split,
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

    def _apply_filter(
        self,
        index: pl.DataFrame,
        max_resolution: float,
        plddt_thresh: float,
    ) -> pl.DataFrame:
        return (
            index.lazy()
            .filter(pl.col(IndexCol.RESOLUTION) <= max_resolution)
            # TODO: add this column / data in the dataset
            # .filter(pl.col(IndexCol.MEAN_PLDDT) >= plddt_thresh)
            .collect()
        )

    def _get_splits(
        self,
        index: pl.DataFrame,
        train_val_test_split: List[float],
    ) -> Tuple[pa.Table, pa.Table, pa.Table]:

        # compute per-column thresholds adjusted for OR-union across N columns
        # formula: per_col_pct = 1 - (1 - target_pct)^(1/N) so the union ≈ target_pct
        assert sum(train_val_test_split) == 1.0
        _, val_pct, test_pct = train_val_test_split
        val_pct = val_pct / (1 - test_pct)
        n_cols = len(self.cluster_split_cols)
        adj_test_pct = 1 - (1 - test_pct) ** (1 / n_cols)
        adj_val_pct = 1 - (1 - val_pct) ** (1 / n_cols)
        test_thresh = int(adj_test_pct * UINT64)
        val_thresh = int(adj_val_pct * UINT64)

        _IS_TEST, _SPLIT = "_is_test", "_split"
        _TEST, _VAL, _TRAIN = 0, 1, 2

        # build OR expressions across all cluster columns
        is_test_expr = pl.lit(False)
        for col in self.cluster_split_cols:
            is_test_expr = is_test_expr | (stable_hash(pl.col(col), seed=TEST_SEED) < test_thresh)

        is_val_expr = pl.lit(False)
        for col in self.cluster_split_cols:
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