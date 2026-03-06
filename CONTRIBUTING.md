# Contributing

Thanks for contributing to DifferentialLab.

## Setup

```bash
pip install -e ".[dev]"
```

Optional documentation toolchain:

```bash
pip install -e ".[docs]"
```

## Before opening a PR

1. Run tests:

```bash
pytest
```

2. Run lint checks:

```bash
ruff check src tests
```

3. If typing checks are used in your workflow:

```bash
mypy src
```

4. Update docs for user-visible changes:
- `README.md`
- `docs/` pages
- API refs if public interfaces changed

## Plugin contributions (`complex_problems`)

If you add a plugin:

- implement package structure (`problem.py`, `ui.py`, `solver.py`, `result_dialog.py`)
- register it in `src/complex_problems/problem_registry.py`
- add solver + registry tests
- update `docs/complex-problems.md` and `docs/api/complex_problems.rst`

## Style

- Keep numerical kernels separated from GUI code.
- Validate UI inputs before solver execution.
- Expose structured result dataclasses with explicit `metadata` and `magnitudes`.
