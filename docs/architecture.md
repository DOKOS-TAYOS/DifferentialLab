# Architecture

DifferentialLab is organized around clear subsystem boundaries:

- `config`: constants, paths, environment schema, and predefined equation catalog
- `solver`: numerical engines, validation, notation, statistics, and expression parsing
- `plotting`: Matplotlib figure builders and animation metadata
- `frontend`: Tkinter dialogs, theme setup, embedded plotting, and UI utilities
- `pipeline`: orchestration for the standard `Solve Equation` path
- `complex_problems`: plugin-based specialized simulation workflows
- `transforms`: scalar function transform tools
- `utils`: logging, export, exceptions, update checks, and shared expression helpers

## High-Level Project Layout

```text
src/
  main_program.py
  pipeline.py
  config/
    equations/
  solver/
  plotting/
  transforms/
  complex_problems/
    common/
  frontend/
    ui_dialogs/
  utils/
tests/
docs/
```

## Application Startup

1. `main_program.py` ensures `src/` is importable for direct runs.
2. `initialize_and_validate_config()` loads `.env` and validates `ENV_SCHEMA`.
3. `get_logger(__name__)` configures the `differential_lab` logger.
4. The update checker may compare the local version with `UPDATE_CHECK_URL`.
5. `get_output_dir()` ensures the export directory exists.
6. `MainMenu` starts the Tkinter main loop.

## Standard Solve Path

```text
EquationDialog
  -> ParametersDialog
  -> run_solver_pipeline
  -> validation + parser + solver dispatch
  -> statistics + metadata
  -> ResultDialog
```

The key design goal is to keep the general solver stack independent from
UI-specific behavior. Plot generation is deferred to result dialogs so users can
switch views without re-solving.

## Solver Dispatch

`run_solver_pipeline()` handles these paths:

- scalar ODEs through `solve_ode()` or `solve_multipoint()`
- vector ODEs through `get_vector_ode_function()` and `solve_ode()`
- difference equations through `solve_difference()`
- 2D PDEs through `solve_pde_2d()`

The returned `SolverResult` is data-only: solution arrays, statistics,
metadata, equation type, grids, and notation context.

## Predefined Equation Catalog

The catalog is loaded from:

- `src/config/equations/ode.yaml`
- `src/config/equations/vector_ode.yaml`
- `src/config/equations/difference.yaml`
- `src/config/equations/pde.yaml`

Current catalog size:

| Type | Entries |
|---|---:|
| ODE | 48 |
| Vector ODE | 50 |
| Difference equation | 10 |
| PDE | 12 |

`solver.predefined.load_predefined_equations()` caches the parsed catalog after
the first successful load.

## Complex Problems Architecture

`complex_problems` uses a lazy plugin registry.

Core pieces:

- `base.py`: `ProblemDescriptor` and `ComplexProblem` protocol
- `problem_registry.py`: lazy registration and dispatch
- `complex_problems_dialog.py`: selector window
- `problem_docs.py`: structured help metadata shown in plugin dialogs
- `common/`: shared helpers for plugin modules

Plugin contract:

- `problem.py` exposes `PROBLEM` with descriptor and `open_dialog(parent)`
- `ui.py` gathers and validates input
- `solver.py` returns a structured result dataclass
- `result_dialog.py` renders module-specific diagnostics
- `model.py` contains physical or numerical helper logic when useful

Registered plugins:

- `coupled_oscillators`
- `membrane_2d`
- `nonlinear_waves`
- `schrodinger_td`
- `antenna_radiation`
- `aerodynamics_2d`
- `pipe_flow`

## Configuration Lifecycle

1. `.env` is loaded from the project root.
2. Values are validated against `ENV_SCHEMA` in `src/config/env.py`.
3. Invalid values are corrected to defaults and logged.
4. Runtime reads values through `get_env_from_schema(key)`.
5. The in-app `Settings` dialog rewrites `.env` and restarts the app.

`LOG_FILE`, `LOG_MAX_BYTES`, and `LOG_BACKUP_COUNT` configure rotating file logs.
If file logging cannot start, the logger falls back to console output.

## API and Import Strategy

- Public APIs are re-exported in package `__init__.py` files.
- Heavy or GUI-specific imports are delayed where practical.
- Internal modules import siblings directly to avoid circular re-export issues.
- The console entry point `differential-lab` maps to `main_program:main`.

## Error Handling and Safety

- User expressions are AST-validated before evaluation.
- Solver input is validated before dispatch.
- Solver failures are converted into user-facing dialog errors where possible.
- Long-running plugin solvers use background helpers and loading dialogs.
- Export paths are centralized through `config.paths`.

## Quality Tooling

- Tests: `pytest`
- Lint and formatting: `ruff`
- Type checking: repo-local `pyright` configuration in `pyproject.toml`
- Docs build: Sphinx + MyST

Run the main quality loop from an activated `.venv`:

```bash
ruff check . --fix
ruff format .
pytest
pyright
```

Run `pyright` when it is installed in your environment.
