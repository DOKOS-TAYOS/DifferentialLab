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

- scalar ODEs through `solve_ode()` or `solve_multipoint()`, with typed IVP
  options/events, an all-at-start IVP shortcut, and BVP routing for every other
  compatible endpoint-only condition set
- vector ODEs through `get_vector_ode_function()` and `solve_ode()`
- difference equations through `solve_difference()`
- scalar linear elliptic 2D PDEs through `solve_pde_2d()`, with affine-residual,
  component-aware ellipticity, structured boundary, periodic-topology, sparse-solve,
  and algebraic-diagnostic helpers behind the compatible facade
- linear strongly elliptic Vector PDE systems through `solve_vector_pde_2d()`, with
  matrix-affinity probing, sampled principal-symbol validation, explicit component
  boundary broadcasting, and component-major sparse block assembly
- scalar linear elliptic 3D PDEs through `solve_pde_3d()`, with a strict 3x3 principal
  matrix, rectangular six-face topology, periodic wrapping, 19-point mixed-derivative
  assembly, and public `(nz, ny, nx)` output

The returned `SolverResult` is data-only: solution arrays, statistics,
metadata, equation type, grids, and notation context. The lower-level
`ODESolution`, `BVPSolution`, `PDESolution`, `PDESolution3D`, and
`VectorPDESolution` carry structured diagnostics;
the condition estimate is deliberately limited to small systems so diagnostics do not
densify large sparse matrices.

`ode_solver.py` keeps the legacy IVP and shooting surfaces, adds a bounded
deterministic SciPy BVP wrapper, and exposes work counters/status/event results.
Safe GUI event expressions are compiled and checked for one finite real scalar
result by `equation_parser.py`; they reach the solver as ordinary callables, so
the numerical layer remains independent of Tk.

PDE responsibilities are split without introducing a generic N-dimensional
framework:

- `pde_solver.py`: backwards-compatible scalar-2D facade;
- `pde_types.py`: public coefficient, boundary, solution, and diagnostic types;
- `pde_validation.py`: grid, affinity, ellipticity, and connected-component checks;
- `pde_boundary.py`: legacy adaptation and Dirichlet/Neumann/Robin/periodic handling;
- `pde_assembly.py`: sparse stencil assembly, periodic wrapping, solve checks, and diagnostics.
- `pde_system_solver.py`: dedicated public Vector PDE facade and component-major output mapping.
- `pde_3d_solver.py`: dedicated rectangular scalar-3D validation, boundary topology,
  sparse assembly, and z/y/x output mapping.

## Predefined Equation Catalog

The catalog is loaded from:

- `src/config/equations/ode.yaml`
- `src/config/equations/vector_ode.yaml`
- `src/config/equations/difference.yaml`
- `src/config/equations/pde.yaml`
- `src/config/equations/pde_3d.yaml`
- `src/config/equations/vector_pde.yaml`

Current catalog size:

| Type | Entries |
|---|---:|
| ODE | 48 |
| Vector ODE | 50 |
| Difference equation | 10 |
| PDE | 12 |
| PDE 3D | 4 |
| Vector PDE | 1 |

`solver.predefined.load_predefined_equations()` caches the parsed catalog after
the first successful load.

Predefined PDE entries may also provide an optional `default_boundary_conditions`
mapping for the standard rectangular boundary controls. It is used only to seed
the Parameters dialog; unspecified faces retain homogeneous Dirichlet defaults and
users can edit every value before solving.

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
