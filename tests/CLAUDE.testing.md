# CLAUDE.testing.md

## What to Test

- test public APIs only
- if it's exposed to the user, it needs a test

## Test Markers

- mark every test with `@pytest.mark.cpu` or `@pytest.mark.gpu`
- default to CPU tests — mock GPU dependencies where possible
- only use `gpu` when the thing being tested genuinely requires a GPU (e.g. custom CUDA kernels)

## No External Dependencies

- mock all network access (S3, web requests, downloads)
- mock credentials and authentication
- mock MLflow and any other logging/tracking services
- tests must run offline with no credentials

## Fixtures

- use shared fixtures in `conftest.py`
- keep fixtures minimal and reusable
- if a fixture is only used by a subset of tests, define a conftest for that subset in its own directory, dont pollute the root conftest with module specific fixtures

## Naming

- test files: `test_<module>.py`
- test functions: `test_<thing_being_tested>`

## Structure

- test directory mirrors source: `tests/test_<module>/` for `proteus/<module>/`
- the structure of the tests directory is somewhat similar in spirit to proteus, but only put tests in the same dir if they share intent (eg all data loader tests in the same dir)
- use pytest as the framework
- for tensor operations, create small tensors with known values
