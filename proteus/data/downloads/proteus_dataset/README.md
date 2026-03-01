# proteus dataset

## sources

- **PDB**: experimental structures, xray and cryo-em
- **PDB-REDO**: xray experimental structures that are refined
- **BFVD**: AF predicted viral structures
- **Viro3d**: AF predicted viral structures
- **AFDB**: structures predicted from AF2 from uniprot
- **ESMAtlas**: ESMFold predicted structures from the metagenomic atlas

## per-sample data

each pdb is stored as a unit. all its chains live together, zstd-compressed.

per chain:
- coords: L,14,3 — atom14 3d coords
- sequence: string of one letter AA identifiers
- atom_mask: L,14 boolean mask of valid atom coords
- ca_cif: CA-only mmCIF string (for foldseek, deleted after clustering)
- per resi plddt: L, (for predicted structures)
- per resi bfactor: L, (for experimental structures)

per pdb:
- chains: list of chain IDs
- biounits: list of lists (inner = chains in biounit, outer = each biounit)
- asmb_xforms: list of Nx4x4 homogeneous transforms, one per biounit

## index

small enough to fit in memory on every process. contains per chain:
- pdb, chain, shard, offset, size (bytes), clusterid, resolution, mean plddt,
  ptm, method (xray/cryo-em/refined/predicted), source (pdb/pdb-redo/afdb/esmatlas/viro3d/bfvd)

## dataset creation

### pass 1 — extraction + staging (parallel across all 6 sources)

for each source, in parallel:
1. stream download the compressed archive
2. for each pdb in the stream, extract everything at once:
    - per chain: `{pdb}_{chain}.npz` (coords, atom_mask, bfactor, sequence)
      -> upload to `s3://bucket/staging/{pdb}/{pdb}_{chain}.npz`
    - per chain: CA-only mmCIF -> gzip -> upload to
      `s3://bucket/staging/{pdb}/{pdb}_{chain}_ca.cif.gz`
    - per pdb: metadata (resolution, method, date, source, assemblies)
      -> gzip json -> upload to `s3://bucket/staging/{pdb}/{pdb}_meta.json.gz`
3. source archive is never stored locally, just streamed
4. chains are kept as separate files during staging so CA CIFs can be
   deleted after foldseek without unpacking blobs

peak local disk: negligible (no local staging, everything streamed to s3).
s3 staging grows to ~8TB of individual per-chain objects.
each source is streamed exactly once.

### foldseek (global, sequential)

1. download CA CIFs from s3 staging, build foldseek db
2. dedup with foldseek easy-linclust at high identity threshold (fast, linear time).
   assign dups the cluster of their representative. delete dup CA coords, keep dedup mapping.
3. cluster representatives with foldseek easy-cluster at structural similarity threshold.
   propagate cluster ids to all chains via dedup mapping.
4. delete CA CIFs from s3 staging. only npz + meta files remain.
5. build catalog (sqlite or parquet) with per-chain cluster assignments.
6. delete staging objects for deduped pdbs from s3

### shard planning (in memory, fast)

1. read the catalog: for each pdb, know its chains and their cluster ids
2. for each pdb, compute its cluster set (the set of clusters its chains belong to)
3. sort pdbs using MinHash LSH on their cluster sets (see [heuristic selection](#heuristic-selection))
4. pack sorted pdbs into shards sequentially, target ~2-4GB each
5. write out a shard plan: ordered list of pdb_ids per shard

### shard assembly from staging (parallel across shards)

for each shard in the plan:
1. read the shard's ordered list of pdb_ids
2. download each pdb blob from s3 staging in the planned order
3. pack into an uncompressed tar, compute offsets for the index
4. stream upload to `s3://bucket/shards/{shard_id}.tar`
5. delete the staging objects for pdbs in this shard

many shards assembled in parallel. peak local disk: one shard buffer per
parallel worker (~2-4GB each).

index is built during assembly — each pdb's offset and size within its shard
is known at pack time. uploaded to s3 at the end.

as shards are finalized, staging objects are deleted. peak total s3 = ~1x
dataset size (staging + partially built shards), not 2x.

### summary

- each source streamed exactly once
- local disk never exceeds ~200GB (CA coords during foldseek)
- s3 staging is temporary, ~$6 for 8TB for a few days
- full control over shard layout
- shard assembly reads from s3 in any order

## shard format

- unit of storage is the pdb (all its chains together, zstd-compressed)
- shards are uncompressed tar archives of pdb blobs
- target ~2-4GB per shard
    - larger shards = fewer shard boundaries = fewer split clusters
    - 8TB / 2GB = ~4,000 shards, 8TB / 4GB = ~2,000 shards
    - byte-range requests mean large shard size doesn't penalize — only fetch
      the pdbs you need
    - want many more shards than max world_size * num_workers so each process
      gets multiple

## heuristic selection

the goal is to sort pdbs so that pdbs with overlapping cluster sets end up in
the same shard, minimizing clusters split across shard boundaries.

**selected: MinHash LSH** — best tradeoff of quality vs complexity for this data.

represent each pdb's cluster set as a sparse binary vector over cluster id space.
use MinHash to compute a signature (64-256 hash functions), then sort pdbs by
their MinHash signature. pdbs with high Jaccard similarity in their cluster sets
get similar signatures and end up adjacent.

why this works well here:
- MinHash is designed for sparse set similarity, which is exactly what cluster
  sets are (~1-10 clusters per pdb out of potentially millions)
- O(N * k) where k = number of hash functions. fast at 50M pdbs.
- captures multi-cluster overlap that simpler heuristics miss — if pdb A has
  clusters {42, 7000} and pdb B has clusters {42, 7001}, MinHash recognizes
  the overlap and places them nearby

tradeoffs:
- projection from high-dimensional similarity to 1D ordering is lossy. two pdbs
  can hash nearby but share nothing, or hash far apart but share clusters. the
  guarantee is statistical, not per-element.
- need to tune number of hash functions. more = better quality, slower compute.
  64-256 is typical, measure split rate on real data.

### other candidates (for reference)

**sort by primary cluster id** (cluster of longest chain):
- O(N log N), trivial to implement
- optimal for single-chain pdbs (majority of AFDB/ESMAtlas = most of the dataset)
- ignores multi-cluster pdbs entirely. a pdb with clusters {42, 7000, 15000}
  only gets placed near cluster 42's neighbors
- good fallback if MinHash is overkill

**sort by min cluster id**:
- nearly identical to primary cluster sort, marginal difference
- only better if cluster id ordering has structural meaning (fragile assumption)

**greedy nearest-neighbor**:
- iteratively place each pdb next to the one with highest cluster set overlap
- highest local quality but O(N^2) naive — infeasible at 50M pdbs
- paints itself into corners (locally optimal, globally suboptimal)
- could work on subproblems (greedy within primary-cluster groups)

**graph-based spectral ordering**:
- build overlap graph, find Fiedler vector for optimal linear ordering
- theoretically best global solution
- infeasible at 50M nodes. could work on a coarsened graph (group by primary
  cluster first, build graph at group level, ~millions of nodes)
- overkill unless split rate from simpler heuristics is unacceptable

## dataloader

### core idea

every process — whether a torch DataLoader worker or a DDP rank — operates
the same way. each gets assigned contiguous shards, plans its own reads,
fetches, and processes. no master, no dispatch, no special coordination.
the same code works for single-GPU with num_workers and multi-GPU with DDP.

because shard planning sorted pdbs by cluster set similarity, clusters are
concentrated on few shards. most clusters are fully contained within a single
process's shard range. when a cluster is split across process boundaries,
both processes sample from it — redundant but not duplicate (different pdbs,
different chains). the MinHash heuristic minimizes this.

### shard assignment

```
total_processes = world_size * num_workers
process i owns shards [i * shards_per_process, (i+1) * shards_per_process)
```

each process computes this independently, no communication.

### per process, per epoch

1. load the index (fits in memory). for each owned shard, the process knows
   which pdbs are in it and which clusters those pdbs' chains belong to.

2. for each owned shard, sample one chain per cluster present in that shard
   (that hasn't been sampled yet this epoch by this process). look up which
   pdbs contain those chains. this gives the set of (pdb, offset, size)
   needed from this shard.

3. plan s3 reads for this shard:
    - few pdbs needed: parallel byte-range requests for just those pdbs
    - many pdbs or >~15% of shard bytes: download the whole shard, discard unneeded
    - coalesce nearby byte ranges within ~64KB gap

4. fetch, decompress, parse, build biounits, tensorize, yield samples.
   prefetch next shard while processing current one.

### pdb cache (per process)

- lru cache keyed by pdb_id
- check cache before adding a pdb to the s3 read plan
- useful across epochs: same pdb, different chain sampled
- useful within epoch: pdb has chains in multiple uncovered clusters

### node-level shared cache (optional)

- shared memory on /dev/shm keyed by pdb_id
- processes on the same node check shared cache before s3
- writes are idempotent, no locking for reads

### batching with packed sequences

- process maintains a token budget per batch (eg max_tokens = 4096)
- packs samples greedily, first-come-first-served
- no sorting needed

## why this works

- zero inter-process communication
- no data duplication, each pdb stored once
- shard locality from MinHash sorting means most clusters are covered by one process
- byte-range requests or full shard reads depending on what's needed
- same code for single-GPU num_workers and multi-GPU DDP
- packed sequences eliminate the need for length-aware shard layout
- redundant cross-process cluster sampling is bounded by heuristic quality
