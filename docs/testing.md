# Testing Guide

## Test Stack

- Framework: `pytest`
- Coverage plugin: `pytest-cov`
- Configuration: `pyproject.toml` (`[tool.pytest.ini_options]`)
- Lint/format: `ruff`
- Type checking: repo-local `pyright` configuration in `pyproject.toml`

Run commands from an activated `.venv`.

## Run All Tests

```bash
pytest
```

## Run Focused Suites

Core solver and pipeline:

```bash
pytest tests/test_ode_solver.py
pytest tests/test_pde_solver.py
pytest tests/test_difference_solver.py
pytest tests/test_pipeline.py
pytest tests/test_equation_parser.py
```

Complex problem modules:

```bash
pytest tests/test_complex_problems_registry.py
pytest tests/test_coupled_oscillators_model.py
pytest tests/test_membrane_2d_solver.py
pytest tests/test_nonlinear_waves_solver.py
pytest tests/test_schrodinger_td_solver.py
pytest tests/test_antenna_radiation_solver.py
pytest tests/test_aerodynamics_2d_solver.py
pytest tests/test_pipe_flow_solver.py
```

Frontend and utilities:

```bash
pytest tests/frontend
pytest tests/utils
pytest tests/config
pytest tests/transforms
```

## What to Validate Before Merge

- New feature has focused tests.
- Nearby regression tests still pass.
- Numerical tests assert stability, conservation, or physically meaningful ranges.
- Registry/UI dispatch tests cover new plugin IDs.
- Docs match the behavior that users can actually run.

## Numerical Test Strategy

Use small but representative setups:

- keep runtime low
- avoid overfitting to exact floating-point values
- assert finite outputs and meaningful ranges
- verify drift and invariants with tolerances, not strict equality
- include at least one failure/validation path when adding user input

## Common Failure Patterns

- Too aggressive `dt` or too coarse a grid for nonlinear models
- Changed defaults breaking baseline expected metrics
- Missing plugin registration
- UI imports failing after renamed module paths
- Docs examples drifting away from actual command names
- Logger or configuration tests leaking global state

## CI-Ready Command Sequence

```bash
ruff check . --fix
ruff format .
pytest
pyright
```

Run `pyright` when it is installed in your environment.

Docs verification:

```bash
pip install -e ".[docs]"
cd docs
make html
```

On Windows, use:

```bat
cd docs
make.bat html
```
