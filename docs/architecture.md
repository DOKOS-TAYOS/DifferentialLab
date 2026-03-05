# Architecture

DifferentialLab follows a modular architecture with clear separation between
configuration, numerical solving, plotting, transforms, the GUI, and shared
utilities.

## Project Layout

```
src/
├── main_program.py          # Entry point, update check on startup
├── pipeline.py              # Orchestrates the full solve workflow
├── config/                  # Configuration & constants
│   ├── constants.py         # App name, version, solver methods, statistics
│   ├── env.py               # .env loading, schema, validation
│   ├── paths.py             # Output file path management
│   ├── equations.py         # Python-callable ODE functions
│   ├── difference_equations.py  # Python-callable difference equation functions
│   └── equations/           # Predefined ODE/difference/PDE definitions (by type)
│       ├── ode.yaml
│       ├── vector_ode.yaml
│       ├── difference.yaml
│       └── pde.yaml
├── solver/                  # Numerical solving engine
│   ├── equation_parser.py   # Safe expression parsing (AST-validated)
│   ├── ode_solver.py        # solve_ivp wrappers, shooting method
│   ├── difference_solver.py # Iterative solver for recurrence relations
│   ├── pde_solver.py        # Finite-difference 2D elliptic PDE solver
│   ├── error_metrics.py     # Residual error, Jacobian evals for ODEs
│   ├── predefined.py        # YAML equation loader with caching
│   ├── notation.py          # FNotation, f-notation to y-notation translation
│   ├── statistics.py        # Post-solve statistical analysis (1D and 2D)
│   └── validators.py        # Input validation
├── plotting/                # Matplotlib output
│   └── plot_utils.py        # Solution plots, phase portraits, surface/contour, save
├── transforms/              # Function transforms
│   ├── function_parser.py   # Safe parsing of scalar f(x) expressions
│   └── transform_engine.py  # Fourier, Laplace, Taylor, Hilbert, Z-transform
├── complex_problems/        # Special problems with custom UIs (experimental, in development)
│   ├── complex_problems_dialog.py  # Problem selection dialog
│   ├── problem_registry.py  # Registry of available problems
│   └── coupled_oscillators/ # Coupled harmonic oscillators
│       ├── model.py         # Physical model
│       ├── solver.py        # Numerical solver
│       ├── ui.py            # Problem-specific dialog
│       └── result_dialog.py # Result visualization
├── frontend/                # Tkinter/ttk GUI
│   ├── theme.py             # Dark theme, colour helpers
│   ├── window_utils.py      # Window centering, modal helpers
│   ├── plot_embed.py        # Embed matplotlib in Tk
│   ├── ui_main_menu.py      # Main menu window
│   └── ui_dialogs/          # Dialog windows
│       ├── equation_dialog.py
│       ├── parameters_dialog.py
│       ├── result_dialog.py
│       ├── config_dialog.py
│       ├── help_dialog.py
│       ├── transform_dialog.py
│       ├── loading_dialog.py
│       ├── collapsible_section.py
│       ├── scrollable_frame.py
│       ├── keyboard_nav.py
│       └── tooltip.py
└── utils/                   # Shared utilities
    ├── exceptions.py        # Custom exception hierarchy
    ├── expression_parser_shared.py  # SAFE_MATH, AST validation for parsers
    ├── export.py            # CSV / JSON / MP4 export
    ├── logger.py            # Logging setup
    └── update_checker.py    # Weekly version check, git pull
```

## Module Responsibilities

### `config`

Loads and validates all settings from the `.env` file.  `ENV_SCHEMA` defines
every configurable key with its type, default, and optional allowed values.
Constants (app name, version, solver method list, statistics catalogue) live
in `constants.py`.  `paths.py` manages timestamped output filenames.
`equations.py` and `difference_equations.py` provide Python-callable functions
for complex ODEs and recurrence relations defined in code.

### `solver`

Pure computation with no GUI dependencies.  `equation_parser` safely parses
user-written ODE, vector ODE, PDE, and difference expressions via AST
validation and compiles them into callables. Includes helper `_compile_and_test()`
for compile-and-validate logic and `_load_config_function()` to load Python
functions from config modules.  `ode_solver` wraps `scipy.integrate.solve_ivp`
with a `solve_multipoint` shooting-method extension and a shared
`_resolve_solver_params()` helper to eliminate duplicate parameter resolution.
`difference_solver` iterates recurrence relations.  `pde_solver` solves 2D
elliptic PDEs with finite differences.  `error_metrics` computes residual error
and Jacobian evaluations.  `notation` provides the f-notation layer: `FNotation`
dataclass, `rewrite_f_expression()` to translate user `f[0]`/`f[1]`/`f[i,k]`
syntax to internal `y[j]` arrays for SciPy compatibility, and
`generate_derivative_labels()` for formatted plot labels (f, f′, f₀, f′₀).
`statistics` computes post-solve magnitudes (mean, RMS, period, energy, etc.)
for 1D and 2D data, with `all_stats` linked to `AVAILABLE_STATISTICS` in
`config.constants`.  `validators` checks all user inputs before solving and
includes `_ordinal()` helper for correct ordinal suffix generation and
module-level constants for repeated values.

### `plotting`

Creates matplotlib figures from solution data.  `plot_utils.py` includes
`_get_colors()` helper for colormap selection with consistent fallback logic.
`create_solution_plot` renders the solution with configurable line style,
colours, and derivatives.  `create_phase_plot` renders the phase portrait
(f vs f′ or f vs df/dx).  `create_phase_3d_plot` renders 3D parametric
trajectories for vector ODEs with 3+ components.  `create_surface_plot` and
`create_contour_plot` handle 2D scalar fields.  `create_vector_animation_plot`
and `create_vector_animation_3d` support vector ODE visualization.  All visual
parameters are read from `config.env`.

### `transforms`

Function transforms for scalar expressions.  `function_parser` parses `f(x)`
expressions safely and uses vectorized evaluation (passes full NumPy arrays to
`eval()` for 10–100× speedup on array operations).  `transform_engine` applies
Fourier (FFT), Laplace (real axis), Taylor series, Hilbert (discrete), and
Z-transform. Includes shared helpers `_compute_taylor_coeffs()` and
`_compute_laplace_samples()` for identical computation blocks across
`apply_transform` and `get_transform_coefficients`.  Supports curve view (f vs x)
and coefficients view (a_i vs i). Laplace transform bounds (`laplace_s_min`,
`laplace_s_max`) are passed as function parameters to `apply_transform` and
`get_transform_coefficients` (defaults 0.1 and 10.0).

### `complex_problems` *(experimental)*

Special problems with custom UIs, parameters, and visualizations. **Still in
development**; may contain bugs.  The
`problem_registry` holds `ProblemDescriptor` entries; each problem provides
an `open_dialog(parent)` callable.  `complex_problems_dialog` lists available
problems and launches the selected one.  The `coupled_oscillators` submodule
implements a one-dimensional chain of N harmonic oscillators: `model.py`
defines the physics, `solver.py` computes the solution, `ui.py` provides the
parameter dialog, and `result_dialog.py` shows mode shapes and time evolution.
`loading_dialog` in `frontend.ui_dialogs` displays progress during long solves.

### `frontend`

The Tkinter/ttk user interface.  `theme.py` builds a dark theme from env
colours.  `window_utils.py` provides window sizing and centering (`center_window`,
`fit_and_center`), modal dialog setup (`make_modal`), and a reusable
`bind_wraplength()` helper for dynamic text wrapping across dialogs.
`plot_embed.py` embeds matplotlib figures in Tkinter with a refactored
`_bind_resize_handler()` for consistent canvas resize handling.
`ui_main_menu.py` is the main window with six buttons: Solve, Function
Transform, Complex Problems, Information, Configuration, and Quit.  Each
dialog in `ui_dialogs/` handles one step of the workflow (equation selection,
parameter input, result display, configuration, help, transforms, loading).
Shared widgets (`ScrollableFrame`, `CollapsibleSection`, `ToolTip`,
`keyboard_nav`) live alongside the dialogs. The `result_dialog.py` uses a
shared `_save_export_file()` handler for CSV and JSON export buttons.

### `utils`

Cross-cutting concerns: a custom exception hierarchy rooted at
`DifferentialLabError`, `expression_parser_shared` providing `SAFE_MATH`,
`normalize_unicode_escapes`, and `validate_expression_ast` used by equation
parsers for safe expression evaluation, CSV/JSON/MP4 export helpers with
consistent public API, a logging setup that reads `LOG_LEVEL`, `LOG_FILE`, and
`LOG_CONSOLE` from the environment (now imports `get_project_root()` from
`config.paths` to avoid path duplication), and `update_checker` for weekly
version checks and optional git pull with improved git stash detection logic.

### `pipeline`

The bridge between the GUI and the solver.  `run_solver_pipeline` executes
the full workflow: validate inputs, resolve ODE/difference/PDE function,
solve, compute statistics, and export files.  It returns a `SolverResult`
dataclass containing raw solution data (`x`, `y`, `statistics`, `metadata`,
`notation`) — no pre-built matplotlib figures.  The result dialog creates plots
interactively from this data, allowing users to select derivatives, phase-space
axes, and visualization modes without re-solving.  Detects 1D vs multivariate
from `variables` and routes to ODE or PDE solver accordingly.

## Data Flow

**Solve workflow (ODEs, difference equations, PDEs):**

```
User input (GUI)
       │
       ▼
  EquationDialog ──► ParametersDialog
                           │
                           ▼
                   run_solver_pipeline()
                     ┌─────┼──────────┐
                     ▼     ▼          ▼
               validate  parse    solve_ode /
               inputs    expr     solve_difference /
                             solve_pde_2d /
                             solve_multipoint
                                      │
                           ┌──────────┼──────────┐
                           ▼          ▼          ▼
                      statistics   export   SolverResult
                           │          │     (x, y, stats,
                           └──────────┼     metadata, notation)
                                      ▼
                              ResultDialog (GUI)
                              creates plots interactively
```

**Complex Problems workflow:**

```
User input (GUI)
       │
       ▼
  ComplexProblemsDialog ──► Problem-specific UI (e.g. CoupledOscillatorsDialog)
                                      │
                                      ▼
                              Problem-specific solver
                                      │
                                      ▼
                              Problem-specific result dialog
```

## Import Strategy

Each module's `__init__.py` re-exports its public API so external code can
write `from solver import solve_ode`.  **Within** a module, files import from
siblings directly (`from solver.equation_parser import parse_expression`) to
avoid circular imports.

Heavy libraries (SciPy, Matplotlib) are imported lazily inside functions to
keep startup time low.  Dialog classes are also imported lazily when the user
clicks a button.

## Configuration Lifecycle

1. At startup, `main_program.py` calls `initialize_and_validate_config()`,
   which loads `.env` via `python-dotenv` and validates every value against
   `ENV_SCHEMA`.
2. Invalid values are silently corrected to their defaults with a log warning.
3. Code reads settings on demand via `get_env_from_schema(key)`.
4. The Configuration dialog writes a new `.env` and restarts the process.

## Design Patterns

**Module-level constants**: Repeated scalar values (e.g., `_MAX_FPS`, `subscripts`,
`_MAX_GRID_POINTS`) are defined at module level rather than inline or inside
functions.

**Lazy imports**: Heavy dependencies (SciPy, Matplotlib) are imported inside
functions to minimize startup time. Configuration-heavy modules use lazy imports
to avoid circular dependencies.

**Public vs private API**: Helper functions prefixed with `_` are module-private;
public functions have no underscore prefix. Export public APIs via `__init__.py`.

**Error handling**: Expression parsing uses AST validation before compilation to
provide clear error messages early. Solver exceptions are caught and displayed
to the user via messageboxes rather than crashing.

**f-notation vs internal y-notation**: Users write equations with `f[0]`, `f[1]`,
`f[i,k]` (function, derivatives, vector components). The `notation` module
translates these to internal `y[j]` arrays required by SciPy's `solve_ivp`.
Plot labels and CSV headers use formatted Unicode (f, f′, f₀, f′₀) for clarity.
