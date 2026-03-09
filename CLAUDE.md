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

## Branching & PRs

- branch naming: `jaime/{feature,fix,debug,chore,etc}/<descriptive-lowercase-name>`
- PR titles: casual lowercase
