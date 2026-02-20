# Architecture

DifferentialLab follows a modular architecture with clear separation between
configuration, numerical solving, plotting, the GUI, and shared utilities.

## Project Layout

```
src/
├── main_program.py          # Entry point
├── pipeline.py              # Orchestrates the full solve workflow
├── config/                  # Configuration & constants
│   ├── constants.py         # App name, version, solver methods, statistics
│   ├── env.py               # .env loading, schema, validation
│   ├── paths.py             # Output file path management
│   └── equations.yaml       # Predefined ODE definitions
├── solver/                  # Numerical solving engine
│   ├── equation_parser.py   # Safe expression parsing (AST-validated)
│   ├── ode_solver.py        # solve_ivp wrappers, shooting method
│   ├── predefined.py        # YAML equation loader with caching
│   ├── statistics.py        # Post-solve statistical analysis
│   └── validators.py        # Input validation
├── plotting/                # Matplotlib output
│   └── plot_utils.py        # Solution plots, phase portraits, save
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
│       ├── collapsible_section.py
│       ├── scrollable_frame.py
│       ├── keyboard_nav.py
│       └── tooltip.py
└── utils/                   # Shared utilities
    ├── exceptions.py        # Custom exception hierarchy
    ├── export.py            # CSV / JSON export
    └── logger.py            # Logging setup
```

## Module Responsibilities

### `config`

Loads and validates all settings from the `.env` file.  `ENV_SCHEMA` defines
every configurable key with its type, default, and optional allowed values.
Constants (app name, version, solver method list, statistics catalogue) live
in `constants.py`.  `paths.py` manages timestamped output filenames.

### `solver`

Pure computation with no GUI dependencies.  `equation_parser` safely parses
user-written ODE expressions via AST validation and compiles them into
callables.  `ode_solver` wraps `scipy.integrate.solve_ivp` with a
`solve_multipoint` shooting-method extension.  `statistics` computes
post-solve magnitudes (mean, RMS, period, energy, etc.).  `validators`
checks all user inputs before solving.

### `plotting`

Creates matplotlib figures from solution data.  `create_solution_plot`
renders `y(x)` with configurable line style, colours, and derivatives.
`create_phase_plot` renders the `(y, y')` portrait for second-order ODEs.
All visual parameters are read from `config.env`.

### `frontend`

The Tkinter/ttk user interface.  `theme.py` builds a dark theme from env
colours.  `ui_main_menu.py` is the main window.  Each dialog in
`ui_dialogs/` handles one step of the workflow (equation selection,
parameter input, result display, configuration, help).  Shared widgets
(`ScrollableFrame`, `CollapsibleSection`, `ToolTip`, `keyboard_nav`) live
alongside the dialogs.

### `utils`

Cross-cutting concerns: a custom exception hierarchy rooted at
`DifferentialLabError`, CSV/JSON export helpers, and a logging setup that
reads `LOG_LEVEL`, `LOG_FILE`, and `LOG_CONSOLE` from the environment.

### `pipeline`

The bridge between the GUI and the solver.  `run_solver_pipeline` executes
the full workflow: validate inputs, parse the expression, solve the ODE,
compute statistics, generate plots, and export files.  It returns a
`SolverResult` dataclass consumed by the results dialog.

## Data Flow

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
               inputs    expr     solve_multipoint
                                      │
                           ┌──────────┼──────────┐
                           ▼          ▼          ▼
                      statistics   plots     export
                           │          │          │
                           └──────────┼──────────┘
                                      ▼
                              SolverResult
                                      │
                                      ▼
                              ResultDialog (GUI)
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
