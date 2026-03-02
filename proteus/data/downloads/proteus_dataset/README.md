# proteus dataset

## sources

- **PDB**: experimental structures, xray and cryo-em
- **PDB-REDO**: xray experimental structures that are refined
- **BFVD**: AF predicted viral structures (monomers only)
- **Viro3d**: AF predicted viral structures (monomers only)
- **AFDB**: structures predicted from AF2 from uniprot
- **ESMAtlas**: ESMFold predicted structures from the metagenomic atlas

## per-sample data

each pdb is stored as a unit. all its chains live together in a single
zstd-compressed npz blob (`{pdb_id}.npz.zst`).

per chain (keyed by `{chain_id}/` prefix in npz):
- coords: L,14,3 — atom14 3d coords
- sequence: string of one letter AA identifiers
- atom_mask: L,14 boolean mask of valid atom coords
- bfactor: L, (for experimental structures)

per pdb (in `_meta` json entry):
- chains: list of chain IDs
- resolution, method, deposit_date, source
- assemblies: list of {chains, transforms (Nx4x4 homogeneous)}

CA-only mmCIF files are written locally during download for foldseek input,
then deleted after clustering.

## index

parquet file, small enough to fit in memory on every process. one row per chain:
- pdb, chain, source, shard_id, offset, size (bytes)
- cluster_id, split (train/val/test)
- resolution, method, mean_plddt, ptm

built incrementally during shard creation, updated in bulk after clustering
with cluster_id and split columns. uploaded to s3 once finalized.

## dataset creation

### download + shard (parallel across sources)

for each source, in parallel:
1. stream download the compressed archive (or API fetch)
2. for each pdb in the stream:
    - serialize all chains + metadata into a single zstd-compressed npz blob
    - append the blob to the current shard buffer
    - when the shard buffer hits the size target (~128-256MB), finalize and
      upload to `s3://bucket/shards/{source}/{shard_id}.tar`
    - append per-chain rows to the local index
    - write CA-only mmCIF per chain to local disk (for foldseek)
3. source archive is never stored locally, just streamed

shards are built on-the-fly during download. no staging step, no re-sharding.
each pdb blob is written to s3 exactly once.

shards are source-homogeneous — each shard contains pdbs from a single source.
this allows independent re-downloading and re-sharding per source without
touching other sources' data.

peak local disk: CA CIFs for foldseek (~200GB). everything else streams
directly to s3.

#### crash-resume via checkpoint

the download pipeline is resumable. after each shard is uploaded to s3, its
index rows are appended to a local `checkpoint.jsonl` file. the checkpoint
write happens only after the s3 upload succeeds, so if a row is in the
checkpoint, its shard is guaranteed to be in s3.

on restart, the checkpoint is read to recover: which pdbs are already done,
the pre-built index rows, and the next shard id. only remaining pdbs are
downloaded. a crash mid-upload loses at most the current shard (re-downloaded
on resume). the checkpoint file is a few MB even at hundreds of thousands of
rows.

after the full pipeline completes successfully (index written, clustering
done, final index uploaded), the checkpoint is deleted. absence of a
checkpoint on the next run means start from scratch.

### foldseek clustering (global, sequential)

1. build foldseek db from local CA CIFs
2. cluster with foldseek at structural similarity threshold
3. parse cluster assignments → update index with cluster_id column
4. delete local CA CIFs

clustering is used only for train/val/test splitting, not for shard layout
or training-time sampling.

### train/val/test split

split is done at the cluster level on experimental structures only, so no
structural leakage between splits.

- ~95% of experimental clusters → train
- ~2.5% → val
- ~2.5% → test
- all predicted sources (AFDB, ESMAtlas, BFVD, Viro3d) → train only

val and test are experimental-only because evaluation should be against
ground truth coordinates, not predicted structures.

val/test pdbs remain in their original source shards. the index labels each
chain with its split. no separate val/test shards are created. during
evaluation, the dataloader uses byte-range requests to read only the
specific val/test pdbs from their source shards. this is efficient because
val/test is small and evaluated infrequently. train workers always read
whole shards for throughput.

after splitting, upload the finalized index to s3.

## shard format

- unit of storage is the pdb (all its chains together, zstd-compressed npz)
- shards are uncompressed tar archives of pdb blobs
- source-homogeneous: each shard contains pdbs from one source only
- target ~128-256MB per shard
    - small enough to download quickly (~1-2s) and hold in worker memory
    - large enough to contain hundreds/thousands of pdbs for good
      within-shard diversity
    - 8TB / 128MB = ~62K shards, 8TB / 256MB = ~31K shards
    - many more shards than max world_size * num_workers so each process
      gets many shards to shuffle over

## dataloader

### core idea

a sample is a chain, presented in its biounit context when available.

every process — whether a torch DataLoader worker or a DDP rank — operates
the same way. each gets assigned a unique random subset of shards, streams
them from s3, and yields samples. no master, no dispatch, no special
coordination. the same code works for single-GPU with num_workers and
multi-GPU with DDP.

### shard assignment

each process is assigned a unique random subset of all shards. assignment
is computed independently per process from the index (filtered to train
split), seeded deterministically so no communication is needed.

### per process, per epoch

1. load the index (fits in memory). filter to train split.

2. shuffle owned shard order for this epoch (seeded by epoch number
   for reproducibility).

3. for each shard:
    a. using only the index, randomly select one chain per pdb in the shard.
    b. stream the shard from s3. as each pdb blob arrives in the tar stream:
        - decompress the blob
        - look up the pre-selected chain
        - randomly sample an assembly (biounit) containing that chain.
          if no assembly contains the chain, yield the chain as-is.
        - build the biounit (apply transforms), tensorize, yield.
    c. prefetch the next shard while processing the current one.

each pdb in the shard produces exactly one sample. the number of samples
per shard is known before streaming begins (= number of train pdbs in shard).

### source ratios

all sources are sampled at their natural frequency. source ratios are
respected via loss weighting — experimental structures receive higher
loss weight relative to predicted structures. this avoids complexity
in the data pipeline and ensures every sample is seen.

### batching with packed sequences

- process maintains a token budget per batch (eg max_tokens = 4096)
- a buffer of yielded samples is accumulated; once the budget is met,
  the batch is returned
- packing is greedy, first-come-first-served from the shard stream

### val/test evaluation

val/test pdbs are scattered across source shards. on first evaluation,
the dataloader reads them via byte-range requests using offset/size from
the index, and caches the results locally. subsequent evaluations read
from cache, avoiding repeated s3 reads.

## why this works

- crash-resumable — checkpoint after each shard upload, lose at most one shard
- no staging, no re-sharding — each pdb written to s3 exactly once
- source-homogeneous shards allow independent updates per source
- streaming shard reads — low memory footprint, process blobs as they arrive
- two-level randomness (shard order + chain selection) gives good
  shuffling independent of source download order
- zero inter-process communication during training
- no data duplication, each pdb stored once
- same code for single-GPU num_workers and multi-GPU DDP
- biounits constructed from single pdb blobs, no cross-shard dependencies
- val/test cached locally after first eval, no separate shard infrastructure
- source ratios via loss weighting, no data pipeline complexity
