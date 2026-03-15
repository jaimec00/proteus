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
- bfactor: L,14 (for experimental structures)
- plddt: L (for predicted structures)
- occupancy: L,14

per pdb (in `_meta` json entry):
- chains: ordered list of chain IDs
- resolution, method, deposit_date, source
- mean_plddt, ptm (quality scores, nan for experimental)
- assemblies: list of {chains, asmb_xforms (Nx4x4 homogeneous)}

per pdb (as numpy arrays in the npz, optional):
- chain_tm_scores: NxN float32 matrix of pairwise chain TM-scores
- chain_seq_identity: NxN float32 matrix of pairwise chain sequence identity

full-backbone mmCIF files are written locally during download (gzip compressed,
as foldseek can work with this directly), then deleted after clustering.

## index

parquet file, small enough to fit in memory on every process. one row per chain:
- pdb, chain, source, shard_id, offset, size (bytes)
- resolution, method, deposit_date
- mean_plddt, ptm
- dynamic cluster columns (e.g. `foldseek_70`, `mmseqs_30`) — one column per
  clustering method and threshold combination, named `{method}_{threshold}`

built incrementally during shard creation, updated in bulk after clustering
with the cluster columns. uploaded to s3 once finalized.

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

### clustering (global, sequential)

supports both foldseek (structural) and mmseqs (sequence) clustering at
configurable thresholds. multiple methods can be run, each producing a
separate column in the index (e.g. `foldseek_70`, `mmseqs_30`).

the foldseek stage is broken into granular steps, each leaving artifacts
on disk so the pipeline can resume from whichever step completed last:

1. `create_db` — build foldseek db from local backbone CIFs, then delete CIFs
2. `run_cluster` — run foldseek cluster, clean up tmp dir
3. `parse_clusters` — run createtsv, delete db files, leave `clusters.tsv`
4. index is updated with the cluster column and uploaded to s3
5. `cleanup_tsv` — delete clusters.tsv only after s3 upload succeeds

on resume, the pipeline checks state in priority order:
- `clusters.tsv` exists → skip all clustering, just read the TSV
- cluster db files exist → skip createdb and cluster, run createtsv
- raw db files exist → skip createdb, run cluster + createtsv
- nothing exists → full pipeline from createdb

clustering is used for train/val/test splitting and for per-epoch sampling
(one chain per cluster per epoch).

### train/val/test split

splitting is done at training time, not baked into the index. this allows
flexibility to experiment with different clustering definitions (e.g. 0.7
structural clustering, 0.3 sequence clustering) and compare them fairly.

at training startup:
- each cluster column is hashed with xxhash using a fixed seed. clusters
  whose hash falls below a threshold are assigned to test, then val, then
  train. thresholds are adjusted per-column so the OR-union across all
  columns approximates the target split ratio
- any pdb that has chains in both test and non-test is moved entirely to
  test to prevent leakage. same for val vs train
- this is deterministic from the seed — no randomness, same split every run

this approach wastes some training data (chains removed to prevent leakage
across multiple clustering definitions), but is necessary for fair comparison.
once a clustering method is chosen, the split can be redefined to be less
strict and recover that data for training, while still ensuring no leakage
within the chosen definition.

split ratios and which clustering columns to split on are configurable.

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

each epoch, one chain is sampled per cluster (deterministic from seed +
epoch number). the seed rotates per epoch for training; val/test use a
fixed seed for reproducibility. effective dataset size per epoch = number
of clusters.

### read planning (S3Orchestrator)

once the per-epoch chain set is determined, the sampled rows are grouped
by shard_id and sorted by byte offset. entries whose gaps are smaller
than a configurable threshold (default 64KB) are coalesced into a single
ReadGroup, minimizing the number of S3 range requests. the resulting
ReadGroups are deterministically shuffled and round-robin partitioned
across workers.

### s3 streaming (S3Reader)

each worker processes its assigned ReadGroups sequentially. for each
group, a single S3 byte-range GET fetches the coalesced range, then
individual blob slices are extracted and deserialized (zstd decompress +
numpy load). blobs shared by multiple chains (same pdb, same offset) are
deduplicated so each blob is deserialized at most once per group.

boto3 clients are lazily initialized per worker to avoid fork-safety
issues with DataLoader's multiprocessing.

### assembly sampling (PDBData)

for each sampled chain, PDBData picks a random assembly containing that
chain (or uses the chain alone with an identity transform if no assembly
includes it). chains within the assembly are shuffled with the target
chain first (so it survives cropping). per-chain coords, atom_mask, and
sequence labels are concatenated. homologous chains are identified from
the chain_tm_scores matrix. the assembly transform is applied during
construct() to produce symmetry copies.

### batching with packed sequences (BatchBuilder)

each worker maintains a sorted buffer of assemblies. when the buffer is
full (configurable buffer_size), the largest assembly that fits the
remaining token budget is popped and added to the current batch. when no
more assemblies fit, the batch is yielded and a new one starts. at epoch
end, the buffer is drained to yield any remaining batches.

### checkpoint / resume

the training loop can save and restore dataloader state:
- `DataHolder.checkpoint_state()` returns `{split: (epoch, step)}` for
  each split (train/val/test)
- `DataHolder.resume_from(state)` sets the sampler epoch and tells Data
  to skip the first N read groups, resuming from where it left off

the read plan is regenerated deterministically from (seed, epoch), so
only the epoch and step index are needed to reconstruct full state.

### val/test evaluation

val/test use a fixed seed so the same chains are evaluated every time.
workers read only the needed byte ranges from their assigned shards.

## why this works

- crash-resumable — checkpoint after each shard upload, clustering resumes from last completed step
- no staging, no re-sharding — each pdb written to s3 exactly once
- source-homogeneous shards for independent updates and data ratio experiments
- cluster-level chain sampling gives good diversity per epoch
- zero inter-process communication — all workers derive the same state from seed + index
- no data duplication, each pdb stored once
- same code for single-GPU num_workers and multi-GPU DDP
- biounits constructed from single pdb blobs, no cross-shard dependencies
- splitting at training time allows flexible experimentation with clustering methods
- deterministic from seed — full reproducibility, checkpoint/resume with just (epoch, step)
