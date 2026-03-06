# Architecture

DifferentialLab is organized around clear subsystem boundaries:

- `config`: validated environment and constants
- `solver`: numerical engines and expression parsing
- `plotting`: matplotlib figure builders
- `frontend`: Tkinter dialogs and embedding
- `pipeline`: orchestration for the standard `Solve` path
- `complex_problems`: plugin-based specialized workflows
- `transforms`: scalar transform tools
- `utils`: shared infrastructure (logging, export, exceptions, update checks)

## High-level project layout

```text
src/
  main_program.py
  pipeline.py
  config/
  solver/
  plotting/
  transforms/
  complex_problems/
  frontend/
  utils/
```

## Standard solve path

```text
EquationDialog -> ParametersDialog -> run_solver_pipeline
  -> validation + parser + solver
  -> statistics + export
  -> ResultDialog
```

Key design goal: keep the general solver stack independent from UI-specific behavior.

## Complex Problems architecture

`complex_problems` uses a lazy plugin registry.

Core pieces:

- `base.py`: `ProblemDescriptor` and `ComplexProblem` protocol
- `problem_registry.py`: lazy registration and dispatch
- `complex_problems_dialog.py`: selector window
- `common/`: shared helpers for plugin modules
  - background execution (`run_solver_with_loading`)
  - safe expression compilation
  - numeric UI parsing helpers

Plugin contract:

- `problem.py` exposes `PROBLEM` with descriptor + `open_dialog(parent)`
- `ui.py` gathers validated input
- `solver.py` returns structured result dataclass
- `result_dialog.py` renders module-specific diagnostics
- `model.py` optional physical/math helpers

## Registered complex problem plugins

- `coupled_oscillators`
- `membrane_2d`
- `nonlinear_waves`
- `schrodinger_td`
- `antenna_radiation`
- `aerodynamics_2d`
- `pipe_flow`

## Configuration lifecycle

1. `main_program.py` calls `initialize_and_validate_config()`.
2. `.env` is loaded and validated against `ENV_SCHEMA`.
3. Invalid values are corrected to defaults and logged.
4. Runtime reads values through `get_env_from_schema(key)`.
5. In-app `Configuration` rewrites `.env` and restarts the app.

## API and import strategy

- Public APIs are re-exported in package `__init__.py` files.
- Heavy libraries are imported lazily where practical.
- Internal modules import siblings directly to avoid circular re-export issues.

## Error handling and safety

- Expression inputs are AST-validated before evaluation.
- Solver failures propagate as user-facing dialog errors, not hard crashes.
- Plugin solvers run in background threads with loading/progress dialogs.

## Extensibility principles

- Keep numeric kernels separate from UI code.
- Use result dataclasses with explicit fields (`metadata`, `magnitudes`, raw fields).
- Add plugin modules without changing generic solver pipeline.
- Register new plugins only in `problem_registry.py`.
