

from __future__ import annotations

from torch.utils.data import IterableDataset, DataLoader
import torch

from typing import Generator, Tuple, Optional, Any
from dataclasses import dataclass, field
from cloudpathlib import S3Path
import pyarrow.parquet as pq
import pandas as pd

from proteus.data.constants import DataPath, DataIndexCols
from proteus.data.data_utils import Assembly, PDBCache, BatchBuilder, DataBatch, Sampler, PDBCacheCfg, SamplerCfg, BatchBuilderCfg

@dataclass
class DataHolderCfg:
    s3_bucket: S3Path
    num_train: int = -1
    num_val: int = -1
    num_test: int = -1
    batch_tokens: int = 16384
    min_seq_size: int = 16
    max_seq_size: int = 16384
    max_resolution: float = 3.5
    min_plddt: float = 0.70
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

        # define data path and path to pdbs
        s3_bucket = cfg.s3_bucket
        index_path = s3_bucket / DataPath.INDEX

        # load the index
        index = pq.read_table(index_path)

        # filter
        index = self._apply_filters(cfg.max_resolution, cfg.min_plddt)

        # dummy split, need to add splits in index
        train_index, val_index, test_index = self._get_splits(index, cfg.num_train, cfg.num_val, cfg.num_test)

        # initialize Data objects
        init_data = partial(
            Data, 
            s3_bucket=cfg.s3_bucket, 
            batch_tokens=config.batch_tokens,
            min_seq_size=config.min_seq_size,
            max_seq_size=config.max_seq_size,
            asymmetric_units_only=config.asymmetric_units_only,
            buffer_size=config.buffer_size,
            seed=config.rng_seed,
        )
        train_data = init_data(index=train_index),
        val_data = init_data(index=val_index)
        test_data = init_data(index=test_index)

        # initialize dataloaders
        init_loader = partial(
            DataLoader, 
            batch_size=None, 
            num_workers=config.num_workers, 
            collate_fn=lambda x: x,
            prefetch_factor=config.prefetch_factor if config.num_workers else None, 
            persistent_workers=config.num_workers>0
        )

        # initialize the loaders
        self.train = init_loader(train_data)
        self.val = init_loader(val_data)
        self.test = init_loader(test_data)

    def _apply_filters(
        self, 
        index: pa.Table, 
        max_resolution: int = 3.5, 
        min_plddt: float = 0.70
    ) -> pa.Table:
        raise NotImplementedError

    def _get_splits(
        self, 
        index: pa.Table, 
        num_train: int, 
        num_val: int, 
        num_test: int
    ) -> Tuple[pa.Table, pa.Table, pa.Table]:
        raise NotImplementedError


class Data(IterableDataset):
    def __init__(
        self, 
        s3_bucket: S3Path, 
        index: pa.Table,
        batch_tokens: int = 16384,
        min_seq_size: int = 16,
        max_seq_size: int = 512,
        asymmetric_units_only: bool = False,
        buffer_size: int = 32,
        seed: int = 0,
    ) -> None:

        super().__init__()

        assert max_seq_size <= batch_tokens, f"max_seq_size ({max_seq_size}) must be <= batch_tokens ({batch_tokens})"
        raise NotImplementedError # not ready yet, get back to this once dataset creation is more consistent

        # keep a cache of pdbs
        self._pdb_cache = PDBCache(PDBCacheCfg(
            pdb_path=config.data_path,
            min_seq_size=config.min_seq_size,
            max_seq_size=config.max_seq_size,
            homo_thresh=config.homo_thresh,
            asymmetric_units_only=config.asymmetric_units_only
        ))

        # for deterministic and UNIQUE sampling
        self._sampler = Sampler(config.clusters_df, SamplerCfg(
            num_clusters=config.num_clusters,
            seed=config.seed
        ))

        # for batch building
        self._batch_builder_config = BatchBuilderCfg(
            batch_tokens=config.batch_tokens,
            buffer_size=config.buffer_size
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