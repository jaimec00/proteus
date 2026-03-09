# CLAUDE.data.md

## Guidelines

- never hardcode data paths — always pull from config
- keep parsing logic pure: take raw data in, return tensors out
- new dataset formats should register through the construct registry
- avoid downloading data at import time — downloads should be explicit and user-triggered
- data processing should be deterministic and reproducible
