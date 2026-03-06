# Testing Guide

## Test stack

- Framework: `pytest`
- Coverage plugin: `pytest-cov`
- Configuration: `pyproject.toml` (`[tool.pytest.ini_options]`)

## Run all tests

```bash
pytest
```

## Run focused suites

```bash
pytest tests/test_complex_problems_registry.py
pytest tests/test_membrane_2d_solver.py
pytest tests/test_nonlinear_waves_solver.py
pytest tests/test_schrodinger_td_solver.py
pytest tests/test_antenna_radiation_solver.py
pytest tests/test_aerodynamics_2d_solver.py
pytest tests/test_pipe_flow_solver.py
```

## What to validate before merge

- New feature has dedicated tests.
- Existing tests for nearby modules still pass.
- Numerical tests assert stability or conservation within tolerances.
- Registry/UI dispatch tests cover new plugin IDs.

## Numerical test strategy

Use small but representative setups:

- keep runtime low
- avoid overfitting to exact floating-point values
- assert physically meaningful ranges and finite outputs
- verify drift and invariants with tolerances, not strict equality

## Common failure patterns

- Too aggressive `dt` or coarse grid for nonlinear models
- Changed defaults breaking baseline expected metrics
- Missing plugin registration
- UI imports failing due renamed module paths

## CI-ready command sequence

```bash
ruff check src tests
pytest
```

If typing is enabled in your workflow:

```bash
mypy src
```
