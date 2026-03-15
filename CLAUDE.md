# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System & Environment

Uses **pixi** (modern conda alternative) for environment management:

```bash
pixi shell -e gpu             # GPU environment with CUDA 12 + flash-attn
pixi shell -e cpu             # CPU env for local dev
```

typically use `cpu` env unless `gpu` is necessary

### Key Types
`proteus/types/__init__.py` defines jaxtyping annotations used throughout:
- `Float`, `Int`, `Bool` tensor types with shape annotations
- Tensor shapes use conventions: `BL` = total residues in batch, `B` = number of samples, `L` = sequence length
- use these for typing. if you need a new type in your code, add it to the types

## GPU/CUDA Notes

- CUDA 12.0+ required for GPU training
- Flash attention 2.8+ for transformer
- **Never call `.item()`, `.cpu()`, or `.numpy()` on tensors in the training loop.** These force a CPU/GPU sync and stall the pipeline. Only sync at explicit logging boundaries (e.g. `get_metrics`).

## Code Style

- **Indentation: tabs, not spaces.** All Python files in this repo use tabs.
- make the code clean and readable, prefer a single descriptive comment above a code block over comments on every line
- use the conventions already in place (types, shapes, lowercase comments)
- imports go at the top of the file unless there's a good reason for inline
- group imports with blank lines between groups: stdlib, then third-party (torch, etc.), then proteus — no blank lines within a group
- fail loud by default — only silence errors when there's a clear reason (e.g. best-effort web fetches)
- never worry about backwards compatibility unless told to. 
- try to use `enum.StrEnum` for string constants that are used at many places, avoid magic strings, to the same effect, avoid magic numbers, these should be enums as well, or driven by a config

## Config Conventions

- when adding a new field with a default to a base config, always explicitly set that field in derived configs (like `DefaultData`) — never rely on silent inheritance

## Testing

- only run tests if the changes being made are actually covered by existing tests — don't run the test suite just for the sake of it
- test complex, non-trivial logic — tests should test this directly, not work around it
- mark every test with `@pytest.mark.cpu` or `@pytest.mark.gpu`
- default to CPU tests — mock GPU dependencies where possible
- only use `gpu` when the thing being tested genuinely requires a GPU (e.g. custom CUDA kernels)
- mock all network access (S3, web requests, downloads), credentials, MLflow and any other logging/tracking services — tests must run offline with no credentials
- use shared fixtures in `conftest.py`, keep them minimal and reusable
- if a fixture is only used by a subset of tests, define a conftest for that subset in its own directory — don't pollute the root conftest with module-specific fixtures
- test files: `test_<module>.py`, test functions: `test_<thing_being_tested>`
- test directory mirrors source: `tests/test_<module>/` for `proteus/<module>/` — only group tests in the same dir if they share intent
- use pytest as the framework
- for tensor operations, create small tensors with known values

## Data

- never hardcode data paths — always pull from config
- keep parsing logic pure: take raw data in, return tensors out
- new dataset formats should register through the construct registry
- avoid downloading data at import time — downloads should be explicit and user-triggered
- data processing should be deterministic and reproducible

## Branching & PRs

- branch naming: `jaime/{feature,fix,debug,chore,etc}/<descriptive-lowercase-name>`
- PR titles: casual lowercase
