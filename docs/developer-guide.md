# Developer Guide

This guide is for contributors extending DifferentialLab internals or adding new problem modules.

## Local environment

```bash
pip install -e ".[dev]"
```

Optional docs toolchain:

```bash
pip install -e ".[docs]"
```

## Repository layout

- `src/config`: schema, constants, paths, equation catalogs
- `src/solver`: parser + numerical solvers + statistics
- `src/frontend`: Tk dialogs and plotting embedding
- `src/complex_problems`: plugin subsystem
- `src/pipeline.py`: orchestration for standard solve flow
- `tests`: pytest suite
- `docs`: Sphinx + MyST docs

## Coding conventions

- Keep compute kernels independent from UI code.
- Prefer typed dataclasses for solver outputs.
- Validate user input in UI before solver invocation.
- Use shared helpers in `complex_problems/common` for safe expressions and background solve handling.
- Keep plugin defaults computationally reasonable (interactive-scale runtime).

### UI text conventions

- Use Unicode math formatting in labels when it improves readability (`xₘᵢₙ`, `Nₓ`, `|ψ|²`, ...).
- If a Unicode subscript is unavailable for a symbol, use `base_subscript` fallback (for example `N_θ`, `N_φ`).
- Prefer non-selectable labels for static help/description text.
- Keep copy-oriented text areas only where users intentionally need to paste symbols/expressions (custom-equation and transform helpers).

## Adding a new complex problem plugin

1. Create package under `src/complex_problems/<plugin_id>/`.
2. Implement:
   - `problem.py` with descriptor and `open_dialog`
   - `ui.py` for inputs
   - `solver.py` for numerics
   - `result_dialog.py` for plots
   - optional `model.py`
3. Export in package `__init__.py`.
4. Register module in `src/complex_problems/problem_registry.py`.
5. Add tests in `tests/`:
   - solver behavior
   - registry dispatch
6. Update docs (`docs/complex-problems.md` and `docs/api/complex_problems.rst`).

## Typical quality loop

1. Implement feature.
2. Run focused tests.
3. Run broader regression set.
4. Update docs and examples.

## Useful commands

```bash
pytest
pytest tests/test_complex_problems_registry.py
ruff check src tests
mypy src
```

## Documentation workflow

```bash
pip install -e ".[docs]"
cd docs
make html
```

For Windows:

```bat
make.bat html
```

## Release hygiene checklist

- Version updated where required.
- Changelog entry added.
- Tests pass in CI scope.
- Docs updated for user-visible changes.
- New plugin appears in registry and guide pages.
