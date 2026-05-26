# Contributing to roboeval

Thank you for your interest in contributing! This document covers how to get set up, the conventions used in this project, and the process for submitting changes.

## Getting started

```bash
git clone https://github.com/quarq/roboeval
cd roboeval
pip install -e ".[dev]"
```

## Running the tests

```bash
python -m unittest discover -s tests -v
```

All tests must pass before a PR is merged. The CI matrix runs Python 3.11 and 3.12.

## Running the demo

```bash
python3 demo.py
```

Output is written to `runs/demo/` (gitignored).

## Code style

- **Python 3.11+** — no compatibility shims needed.
- **No external runtime dependencies** — the core `roboeval` package must stay stdlib-only. Optional extras (e.g. `torch`) belong under `[project.optional-dependencies]` in `pyproject.toml`.
- Format and lint with [ruff](https://docs.astral.sh/ruff/): `ruff check . && ruff format .`
- Type annotations are encouraged but not required for every function.

## Submitting a pull request

1. Fork the repo and create a feature branch from `main`.
2. Make your changes and add/update tests as needed.
3. Ensure `python -m unittest discover -s tests -v` is clean.
4. Open a PR with a clear description of what changes and why.

## Reporting bugs

Open an issue at https://github.com/quarq/roboeval/issues with:

- Python version and OS
- Minimal reproduction snippet
- Expected vs. actual behavior

## Feature requests

Open an issue tagged `enhancement`. Describe the use-case and the API you have in mind.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License that covers this project.
