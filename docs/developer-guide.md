# Developer Guide

This guide is for contributors extending DifferentialLab internals or adding
new problem modules.

## Local Environment

Use the project-local virtual environment `.venv`.

Windows:

```bat
bin\setup.bat --dev
.venv\Scripts\activate
```

Linux/macOS:

```bash
./bin/setup.sh --dev
source .venv/bin/activate
```

Manual equivalent:

```bash
pip install -e ".[dev]"
```

Optional docs toolchain:

```bash
pip install -e ".[docs]"
```

## Repository Layout

- `src/config`: constants, paths, `.env` schema, and equation catalogs
- `src/solver`: parsers, validators, numerical solvers, notation, and statistics
- `src/frontend`: Tkinter dialogs, theme, embedded plots, and UI helpers
- `src/plotting`: Matplotlib plotting and animation metadata helpers
- `src/transforms`: function parser and transform engine
- `src/complex_problems`: plugin subsystem for specialized models
- `src/pipeline.py`: orchestration for the standard solve flow
- `src/utils`: logging, exports, exceptions, update checks, and shared expression parsing
- `tests`: pytest suite
- `docs`: Sphinx + MyST documentation

## Coding Conventions

- Include typing in function definitions.
- Keep compute kernels independent from UI code.
- Prefer typed dataclasses for solver outputs.
- Validate user input before solver invocation.
- Reuse helpers in `complex_problems/common` instead of duplicating dialog logic.
- Keep plugin defaults computationally reasonable for interactive use.
- Keep public-facing docs aligned with `pyproject.toml`, `ENV_SCHEMA`, and the plugin registry.

## UI Text Conventions

- Use readable mathematical notation in labels when it improves clarity.
- If a Unicode subscript is unavailable for a symbol, use `base_subscript`
  fallback style, for example `N_theta` or `N_phi`.
- Prefer non-selectable labels for static help/description text.
- Keep copy-oriented text areas only where users intentionally need to paste
  symbols or expressions.

## Adding a New Complex Problem Plugin

1. Create a package under `src/complex_problems/<plugin_id>/`.
2. Implement:
   - `problem.py` with descriptor and `open_dialog`
   - `ui.py` for inputs
   - `solver.py` for numerics
   - `result_dialog.py` for plots
   - `model.py` for physical/math helpers when useful
3. Export public pieces in the package `__init__.py`.
4. Register the module in `src/complex_problems/problem_registry.py`.
5. Add tests:
   - solver behavior
   - registry dispatch
   - UI construction when practical
6. Update docs:
   - `docs/complex-problems.md`
   - `docs/api/complex_problems.rst`
   - `CHANGELOG.md`

## Typical Quality Loop

From an activated `.venv`:

```bash
ruff check . --fix
ruff format .
pytest
pyright
```

Run `pyright` when it is installed in your environment. For focused work, run
the nearest tests first, then the broader set.

Useful focused commands:

```bash
pytest tests/test_complex_problems_registry.py
pytest tests/test_membrane_2d_solver.py
pytest tests/test_nonlinear_waves_solver.py
pytest tests/test_schrodinger_td_solver.py
pytest tests/test_antenna_radiation_solver.py
pytest tests/test_aerodynamics_2d_solver.py
pytest tests/test_pipe_flow_solver.py
```

## Documentation Workflow

```bash
pip install -e ".[docs]"
cd docs
make html
```

For Windows:

```bat
make.bat html
```

Documentation should be updated when a change affects:

- user-visible behavior
- configuration keys or defaults
- supported equations, plugins, transforms, or exports
- public modules covered by API docs
- setup, run, test, or release workflow

## Release Hygiene Checklist

- Version updated where required.
- `CHANGELOG.md` entry added.
- Relevant docs updated.
- Tests pass in the expected scope.
- `ruff check . --fix` and `ruff format .` have been run.
- `pyright` passes before merge or push when it is available.
- New plugin appears in the registry, guide page, and API docs.
