# proteus dataset

## sources

- **PDB**: experimental structures, xray and cryo-em (per-structure HTTP download)
- **PDB-REDO**: refined xray experimental structures (per-structure HTTP download)
- **BFVD**: AF predicted viral structures (streamed archive, TODO)
- **Viro3d**: AF predicted viral structures (streamed archive, TODO)
- **AFDB**: AF2 predicted structures from uniprot (streamed archive, TODO)
- **ESMAtlas**: ESMFold predicted structures from the metagenomic atlas (streamed archive, TODO)

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

full-backbone mmCIF files are written locally during download (gzip compressed,
as foldseek can work with this directly), then deleted after clustering.

## index

parquet file, small enough to fit in memory on every process. one row per chain:
- pdb, chain, source, shard_id, offset, size (bytes)
- cluster_id
- resolution, method, deposit_date
- mean_plddt, ptm (for predicted sources, TODO)

built incrementally during shard creation, updated in bulk after clustering
with the cluster_id column. uploaded to s3 once finalized.

splitting is not recorded in the index — it is determined at training time
to allow flexibility across experiments (see train/val/test split section).

## dataset creation

### download + shard (parallel across sources)

for each source, in parallel:
1. download structures (per-structure HTTP for experimental, streamed archive
   for predicted sources)
2. for each pdb:
    - serialize all chains + metadata into a single zstd-compressed npz blob
    - append the blob to the current shard buffer
    - when the shard buffer hits the configured size target, finalize and
      upload to s3
    - append per-chain rows to the local index
    - write full-backbone gzip-compressed mmCIF per chain to local disk
      (for foldseek)
3. predicted source archives are streamed, never stored locally

shards are built on-the-fly during download. no staging step, no re-sharding.
each pdb blob is written to s3 exactly once.

shards are source-homogeneous — each shard contains pdbs from a single source.
rcsb and pdb-redo are grouped together as "experimental" since pdb-redo is
refined rcsb data. this allows independent re-downloading and re-sharding
per source without touching other sources' data, and makes data ratio
experiments straightforward.

peak local disk: backbone CIFs for foldseek. everything else streams
directly to s3.

#### crash-resume via checkpoint

the download pipeline is resumable. after each shard is uploaded to s3, its
index rows are appended to a local checkpoint file. the checkpoint write
happens only after the s3 upload succeeds, so if a row is in the checkpoint,
its shard is guaranteed to be in s3.

on restart, the checkpoint is read to recover: which pdbs are already done,
the pre-built index rows, and the next shard id. only remaining pdbs are
downloaded. a crash mid-upload loses at most the current shard (re-downloaded
on resume).

after the full pipeline completes successfully (index uploaded to s3), the
checkpoint and cluster TSV are deleted. absence of both on the next run
means start from scratch.

### foldseek clustering (global, sequential)

the foldseek stage is broken into granular steps, each leaving artifacts
on disk so the pipeline can resume from whichever step completed last:

1. `create_db` — build foldseek db from local backbone CIFs, then delete CIFs
2. `run_cluster` — run foldseek cluster, clean up tmp dir
3. `parse_clusters` — run createtsv, delete db files, leave `clusters.tsv`
4. index is updated with cluster_id column and uploaded to s3
5. `cleanup_tsv` — delete clusters.tsv only after s3 upload succeeds

on resume, the pipeline checks state in priority order:
- `clusters.tsv` exists → skip all foldseek, just read the TSV
- cluster db files exist → skip createdb and cluster, run createtsv
- raw db files exist → skip createdb, run cluster + createtsv
- nothing exists → full pipeline from createdb

clustering is used only for train/val/test splitting, not for shard layout
or training-time sampling.

### train/val/test split

splitting is done at training time, not baked into the index. this allows
flexibility to experiment with different clustering definitions (e.g. 0.5
structural clustering, 0.3 sequence clustering) and compare them fairly.

at training startup:
- clusters are randomly sampled for val and test sets using a seed
- any chains in the train set that appear in either val or test clusters
  are removed, ensuring no leakage
- this is done independently for each clustering definition so results
  are comparable across methods

this approach wastes some training data (chains removed to prevent leakage
across multiple clustering definitions), but is necessary for fair comparison.
once a clustering method is chosen, the split can be redefined to be less
strict and recover that data for training, while still ensuring no leakage
within the chosen definition.

split ratios and which sources are eligible for val/test are configurable.
evaluation should use experimental structures for ground truth comparison.

## shard format

- unit of storage is the pdb (all its chains together, zstd-compressed npz)
- shards are uncompressed tar archives of pdb blobs
- source-homogeneous: each shard contains pdbs from one source only.
  rcsb and pdb-redo are grouped into "experimental" shards, each predicted
  source gets its own shards. this makes data ratio experiments easy —
  include or exclude entire source shard sets
- target shard size is configurable (default ~256MB)
    - small enough to download quickly and hold in worker memory
    - large enough to contain many pdbs for good within-shard diversity
    - many more shards than max world_size * num_workers so each process
      gets many shards to shuffle over

## dataloader

### core idea

a sample is a chain, presented in its biounit context when available.

all workers start with the same seed and sample one chain per cluster,
so they agree on which chains form the training set for the epoch. the
seed changes per epoch to rotate which chain represents each cluster.
effective dataset size per epoch = number of clusters.

### byte-range reads from s3

once the per-epoch chain set is determined, each worker uses the index
to compute the byte ranges needed from each shard. depending on how
well-coalesced the needed ranges are within a shard, reads are either
issued as individual byte-range requests or as a single contiguous read
spanning from the first needed pdb to the last.

work is partitioned across workers at the pdb level — no pdb is split
across workers, and each pdb is read by exactly one worker. partitioning
is deterministic from the seed so no inter-worker communication is needed.

### batching with packed sequences

- each worker maintains a token budget per batch (e.g. max_tokens = 4096)
- a buffer of yielded samples is accumulated; once the budget is met,
  the batch is returned
- packing is greedy, first-come-first-served from the read stream

### val/test evaluation

val/test chains are determined at training startup by cluster sampling
(see train/val/test split section). workers read only the needed byte
ranges from their source shards using the index.

## why this works

- crash-resumable — checkpoint after each shard upload, foldseek resumes from last completed step
- no staging, no re-sharding — each pdb written to s3 exactly once
- source-homogeneous shards for independent updates and data ratio experiments
- cluster-level chain sampling gives good diversity per epoch
- zero inter-process communication — all workers derive the same state from seed + index
- no data duplication, each pdb stored once
- same code for single-GPU num_workers and multi-GPU DDP
- biounits constructed from single pdb blobs, no cross-shard dependencies
- splitting at training time allows flexible experimentation with clustering methods
