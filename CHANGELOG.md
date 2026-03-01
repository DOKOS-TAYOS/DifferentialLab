# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - Unreleased

### Fixed

- **Ordinal suffix generation**: `validators.py` now generates correct ordinal strings ("1st", "2nd", "3rd", "4th") instead of incorrect suffixes like "2th", "3th". Added `_ordinal()` helper function.
- **Git stash-pop reliability**: `update_checker.py` now correctly detects when a git stash was created by checking for "Saved working directory" in stdout, replacing the fragile "No local changes" check.
- **Dead code in env.py**: Removed unreachable `optional_fields` set in the `get_env()` function's string-type branch (the set only contained `SOLVER_MAX_STEP` which has type `float`).
- **Silent dict-to-None conversion**: `predefined.py` now explicitly checks for empty partial_derivatives dict instead of relying on falsy value conversion.

### Changed

- **Code deduplication**: Extracted 11 reusable helper functions across 8 modules, eliminating ~250 lines of duplicated code and improving maintainability.
  - `solver/ode_solver.py`: Extracted `_resolve_solver_params()` to eliminate duplicate solver parameter resolution logic (was copy-pasted between `solve_ode` and `solve_multipoint`).
  - `solver/equation_parser.py`: Extracted `_compile_and_test()` helper for expression compilation and validation (eliminated 3 instances of identical compile-then-test logic) and `_load_config_function()` for loading named functions from config modules (eliminated 2 duplicate instances).
  - `transforms/transform_engine.py`: Extracted `_compute_taylor_coeffs()` for Taylor series computation (eliminated duplication in `apply_transform` and `get_transform_coefficients`) and `_compute_laplace_samples()` for Laplace integration (eliminated duplication). Also fixed hardcoded Laplace bounds (0.1, 10.0) in `get_transform_coefficients()` to use configurable `laplace_s_min`/`laplace_s_max` environment parameters.
  - `plotting/plot_utils.py`: Extracted `_get_colors()` helper for colormap fallback logic (eliminated 2 instances of identical fallback code).
  - `frontend/plot_embed.py`: Extracted `_bind_resize_handler()` to consolidate canvas resize event handling (eliminated duplication in `embed_animation_plot_in_tk` and `embed_plot_in_tk`). Moved `_MAX_FPS = 30` to module level for consistency.
  - `frontend/window_utils.py`: Added `bind_wraplength()` utility function to consolidate label wraplength binding pattern across 4 dialog files (`equation_dialog.py`, `parameters_dialog.py`, `config_dialog.py`, `help_dialog.py`).
  - `frontend/ui_dialogs/result_dialog.py`: Extracted `_save_export_file()` shared handler to merge duplicate CSV/JSON export button handlers.
  - `utils/logger.py`: Now imports `get_project_root()` from `config.paths` instead of recomputing the path inline.
  - `solver/statistics.py`: Linked `all_stats` set to `AVAILABLE_STATISTICS` from `config.constants` instead of maintaining a disconnected hardcoded set.
  - `utils/export.py`: Removed pointless wrapper `export_json_to_path()` and fixed `export_all_results()` to consistently use the public API.
- **Performance optimization**: Vectorized `function_parser.py` scalar function evaluation, achieving 10–100× speedup for array operations by passing full NumPy arrays directly to `eval()` instead of iterating element-wise.
- **Type safety**: Changed `EquationType` in `solver/predefined.py` from `str` alias to `Literal["ode", "difference", "pde", "vector_ode"]` for better type checking and IDE support. Removed duplicate definition in `pipeline.py`.
- **UI consistency**: Fixed Spanish label "Duración (s):" to "Duration (s):" in `plot_embed.py` for UI language consistency.
- **Module-level constants**: Moved repeated inline values to module level in `solver/validators.py` (`subscripts`, `_is_finite()` helper, `_MAX_GRID_POINTS`).
- **Lazy imports**: Moved `import time` to module level in `utils/update_checker.py`.
- **Exception handling**: Simplified redundant exception types in `frontend/theme.py` (removed unnecessary `tk.TclError` from except clause since it's already a subclass of `Exception`).
- **Pipeline refactoring**: Extracted `is_2d_pde = is_pde and len(vars_list) >= 2` as a local variable in `pipeline.py` to replace 4 scattered occurrences of the same condition.

### Verification

- All existing tests pass without modification.
- No breaking changes; all refactoring is internal.
- Ruff linting shows no new issues.
- Type checking (mypy) improved by stricter `Literal` type definitions.

## [0.2.2]

### Added

- **Save CSV / Save JSON buttons** in the Results dialog: new "Export Data" section with buttons that open the file explorer so the user can choose the save path for solution data (CSV) and metadata/statistics (JSON).
- **Equation categorization**: equations are now grouped by category for easier browsing and selection.

### Changed

- **No automatic file saving**: graphs, CSV, and JSON files are no longer saved automatically. Plots can be saved via the Matplotlib toolbar; CSV and JSON via the new Save buttons in the Results dialog.
- **MP4 animation export**: the Export MP4 button now opens the file explorer to let the user choose the save path instead of writing to a fixed output directory.
- **Logging improvements**: enhanced log output for better debugging and traceability.
- **UI improvements**: various refinements to the interface, including layout organization and button placement.

## [0.2.1]

### Added

- **Automatic update check on startup**: checks weekly (configurable) if a newer version is available in the repository. If so, prompts the user to update; on acceptance, runs `git pull` while preserving `input/`, `output/`, and `.env`. Configurable via `CHECK_UPDATES`, `CHECK_UPDATES_FORCE`, and `UPDATE_CHECK_URL` in `.env`. See `utils/update_checker.py`.
- **Solver error metrics in results**: the Solver Info section now shows rtol, atol, residual max/mean/rms (how well the solution satisfies the ODE), and Jacobian evaluations (for implicit methods). Computed via `solver/error_metrics.py` for ODE and vector ODE runs.
- **Phase portrait for first-order ODEs**: phase plot is now always shown for ODEs. For first-order equations it displays y vs dy/dx (numerical derivative); for second-order and above it shows y vs y′ as before.

- **Function Transforms module**: new "Transforms" button in the main menu opens a dialog to enter a scalar function f(x), apply mathematical transforms, and visualize or export the result. Supports: Original (f(x)), Fourier (FFT), Laplace (real axis), Taylor series, Hilbert (discrete), and Z-transform (discrete).
- **Display mode**: second combobox to choose between "Curve (f vs x)" and "Coefficients (a_i vs i)". Coefficients view shows Taylor a_i, Fourier |F[k]|, Laplace L(s_i), Hilbert |H[k]|, or Z-transform samples x[n] vs index.
- **`transforms/` module**: `function_parser` for safe parsing of f(x) expressions, `transform_engine` with `apply_transform`, `get_transform_coefficients`, `TransformKind`, and `DisplayMode`.
- **Transform help dialog**: collapsible sections (About, Function Input, Transformations, Display Mode, Export) with grab handling so collapsibles work when opened from the modal Transform dialog.
- **Vector ODE support**: equations can now be vector-valued [f₀(x), f₁(x), …] with f_i'' = h_i(x, f₀, f₁, …, f₀', f₁'). The system detects vector mode via `vector_expressions` or `equation_type: vector_ode`. Scalar ODEs and PDEs behave unchanged.
- **`parse_vector_expression` and `get_vector_ode_function`**: parse lists of expressions into vector ODE callables. State layout: [f₀, f₀', f₁, f₁', …].
- **Animation tab**: for vector ODEs, a new "f_i(x) Animation" tab with Tkinter Scale, Play (▶) and Stop (■) buttons, and duration entry. Chain-style plot (points + lines, verticals to x-axis) instead of bars. Duration (seconds) controls both playback speed and MP4 export length. Playback capped at 30 fps to avoid matplotlib overload.
- **3D vector tab**: "f_i(x) 3D" tab showing x (independent), component index i, and f_i(x) as a surface.
- **Predefined vector equations**: Coupled Harmonic Oscillators, Double Pendulum (Linearized), Three Coupled Oscillators, 20 Coupled Harmonic Oscillators, Damped Coupled System in `equations.yaml`.
- **PDE (multivariate) support**: equations can now have multiple independent variables (e.g. f(x,y)). Select "PDE (multivariate)" in the equation dialog. For f(x,y), the output is a 3D surface plot by default; the user can choose 2D contour instead. Domain and grid size are configurable per dimension.
- **`solver/pde_solver.py`**: finite-difference solver for 2D elliptic PDEs (-u_xx - u_yy = f). Uses 5-point stencil and zero Dirichlet boundary conditions.
- **`parse_pde_rhs_expression`**: parses RHS expressions for PDEs using variable names (x, y, …) and parameters.
- **`create_surface_plot` and `create_contour_plot`**: 3D surface and 2D contour plots for 2D scalar fields.
- **Predefined PDE equations**: Poisson 2D (-∇²u = f) and Laplace 2D (∇²u = 0) in `equations.yaml`.
- **`variables` and `partial_derivatives`** in `PredefinedEquation`: schema extended for multivariate equations.
- **`compute_statistics_2d`**: statistics (mean, std, max, min, integral) for 2D scalar fields.
- **2D CSV export**: `export_all_results` accepts optional `y_grid` for x,y,u column format.
- **`export_animation_to_mp4`**: exports vector animation as MP4 with user-specified duration; frames downsampled to 500 max to avoid memory exhaustion; explicit ffmpeg availability check.

### Fixed

- **PDE grid size**: validation limits grid to 1000 points per axis to avoid excessive memory allocation. User-entered values above this show a dialog before attempting solve.
- **Uncaught solver exceptions**: `MemoryError`, `OSError`, and other exceptions in the solver pipeline are now caught and shown in a messagebox instead of crashing with a traceback.

### Changed

- **Results dialog sections**: Magnitudes, Statistics, Solver Info, and Output Files are now collapsible (expand/collapse via clickable headers with arrow indicators).
- **Pipeline**: detects 1D vs multivariate from `variables`; routes to ODE or PDE solver accordingly.
- **Equation dialog**: added PDE type; passes `variables` to parameters dialog.
- **Parameters dialog**: for PDE, shows y_min/y_max, grid points per axis, and plot type (3D/2D).
- **Animation playback**: Tkinter Scale instead of matplotlib Slider (fixes unresponsive slider when embedded); duration (seconds) replaces pts/s for both Play and MP4 export; playback capped at 30 fps.

- **Difference equations (recurrence relations)**: the project now supports both differential equations (ODEs) and difference equations. Select "Difference (recurrence)" in the equation dialog to solve recurrences of the form y_{n+order} = f(n, y_n, y_{n+1}, ...). Use n for the index, y[0] for y_n, y[1] for y_{n+1}, etc.
- **Predefined difference equations**: geometric growth, logistic map, Fibonacci recurrence, second-order linear recurrence, discrete logistic (cobweb model).
- **`config/difference_equations.py`**: Python functions for difference equations (geometric_growth, logistic_map, fibonacci, linear_recurrence_2, cobweb_model).
- **`solver/difference_solver.py`**: iterative solver for difference equations.
- **ODEs defined via Python functions**: equations in `equations.yaml` can now use `function_name` to reference a function in `config.equations.py` instead of (or in addition to) the `expression` string. This allows defining complex differential equations in code rather than a single expression.
- **`config/equations.py`**: new module with Python-callable ODE functions: harmonic oscillator, damped oscillator, exponential growth, logistic, Van der Pol, pendulum, RC circuit, free fall, time-dependent Schrödinger equation, Lorenz system, Duffing oscillator, Lotka-Volterra, rigid body Euler equations, and Bloch equations.
- **New predefined equations**: Time-dependent Schrödinger equation, Lorenz system, Duffing oscillator, Lotka-Volterra predator-prey, rigid body Euler equations, Bloch equations.

### Changed

- **`formula` is now required**: all predefined equations must have a `formula` field for display; equations without it are skipped on load.
- **`expression` optional when `function_name` is set**: equations using `function_name` no longer need an `expression`; execution uses the imported function.
- **Pipeline and validators**: `run_solver_pipeline` and `validate_all_inputs` now accept either `expression` or `function_name` (keyword-only arguments).

## [0.2.0] - 2026-02-20

### Added

- **Sphinx documentation**: full `docs/` tree with Sphinx `conf.py`, `Makefile`, and `make.bat`; builds with `sphinx-build` using the Read the Docs theme.
- **Narrative docs**: Getting Started, User Guide, Configuration Reference, and Architecture pages written in Markdown (MyST-compatible and valid standalone Markdown).
- **API reference**: auto-generated RST pages for every module (`config`, `solver`, `plotting`, `pipeline`, `frontend`, `utils`) using `autodoc`, `napoleon`, and `sphinx-autodoc-typehints`.
- **Intersphinx links**: cross-references to Python, NumPy, SciPy, and Matplotlib documentation.

### Changed

- **README.md**: removed stale Colorama dependency from the table, added a Documentation section with local build instructions.
- **`.gitignore`**: added `docs/_build/` to ignore Sphinx build output.
- **Window sizing helper**: extracted `fit_and_center()` in `window_utils.py` to replace repetitive sizing/centering boilerplate across five dialog classes (`ui_main_menu`, `equation_dialog`, `parameters_dialog`, `config_dialog`, `help_dialog`).
- **Plot helper functions**: extracted `_new_figure()` and `_finalize_plot()` in `plot_utils.py` to centralize figure creation and axis styling, reducing duplication between `create_solution_plot` and `create_phase_plot`.
- **Applied `FONT_AXIS_STYLE`**: the previously unused env variable now controls axis label font style in all plots.
- **Inlined trivial wrapper**: merged `_validate_ode_expression()` into `validate_all_inputs()` in `validators.py`.
- **Lazy imports**: moved `os` and `sys` inside `_on_config()` in `ui_main_menu.py` (only needed on restart).
- **Linting fixes**: removed trailing whitespace in `theme.py`, reformatted long import line in `pipeline.py`.

### Removed

- **Unused dependency**: removed `colorama` from `requirements.txt` and `pyproject.toml`.
- **Dead code**: removed `ConfigurationError` exception class (defined but never raised) and its test.
- **Unused config option**: removed `UI_ENTRY_WIDTH` from `ENV_SCHEMA`, `.env.example`, and the configuration dialog.
- **Deprecated AST node**: removed `ast.Index` from the expression parser's allowed-nodes list (deprecated since Python 3.9, unused in 3.12+).

## [0.1.3] - 2026-02-20

### Added

- **Unicode mathematical symbols**: predefined equations now display proper mathematical notation (∂, ∫, √, π, θ, ω, etc.) in labels and descriptions throughout the UI.

### Changed

- **Linter configuration**: enhanced Ruff settings in `pyproject.toml` with explicit rule selection for code quality, import sorting, and type checking.
- **Import optimization**: reorganized imports across multiple modules for better readability and consistency; removed unused imports in `pipeline.py`, `validators.py`, and `theme.py`.
- **UI refinements**: improved widget spacing and alignment in dialogs; adjusted padding values for better visual consistency across different screen sizes.
- **Configuration handling**: streamlined environment variable loading and validation in `config_loader.py`; improved error messages for invalid configuration values.

### Fixed

- **Minor UI bugs**: corrected focus behavior in entry widgets when switching between tabs; fixed occasional layout glitches in collapsible sections.
- **Type hints**: added missing type annotations in several utility functions for better IDE support and static analysis.

## [0.1.2] - 2026-02-20

### Added

- **Derivative selection**: new "Derivatives to Plot" section in both the Predefined and Custom equation tabs, with checkboxes for each order (y, y′, y″, …); validates that at least one derivative is selected before solving.
- **Dynamic derivative checkboxes**: in the Custom tab the derivative checkboxes regenerate automatically whenever the ODE order changes.
- **Quick evaluation-points buttons**: `+` and `−` buttons next to the evaluation-points field in the Parameters dialog multiply or divide the value by 10 (order-of-magnitude steps), with a minimum of 10 points enforced.
- **Solver method descriptions**: a dynamic label in the Parameters dialog shows a short description of the currently selected solver method, updating on every combobox change.

### Changed

- **Consistent dialog sizing**: all dialogs now follow a unified sizing pattern — requested content size plus padding, clamped between a per-dialog minimum and a screen-ratio cap (90–92 % of screen dimensions). Minimum sizes: Parameters 740 × 700, Equation 820 × 650, Configuration 800 × 700, Information 900 × 750, Results 1200 × 700.
- **Selected derivatives propagated end-to-end**: the `selected_derivatives` list flows from `EquationDialog` → `ParametersDialog` → `run_solver_pipeline` → plotting functions, so only the chosen derivatives appear in the output plots.


## [0.1.1] - 2026-02-19

### Added

- **Reusable `ScrollableFrame` widget** (`scrollable_frame.py`) with cross-platform mousewheel support, recursive child binding, and canvas-width sync on resize.
- **Keyboard navigation module** (`keyboard_nav.py`) with arrow-key navigation across a 2-D widget grid and Enter/Return to invoke buttons.
- **Collapsible sections** in the Information and Configuration dialogs, with unicode arrow indicators and click-to-toggle headers.
- **Human-readable Information dialog** rewritten with eight clearly structured sections (About, How to Use, Custom Expressions, Predefined Equations, Statistics, Solver Methods, Output Files, Configuration).
- **Description labels** for every configuration option, rendered below each field with dynamic wraplength.
- **Application restart** after saving configuration, using `os.execv` to relaunch with the updated `.env`.

### Changed

- **Buttons always at the bottom**: all dialogs now have a fixed bottom button bar (packed before scrollable content) so action buttons are never clipped or scrolled away.
- **Focus highlighting**: all interactive widgets (buttons, entries, comboboxes, checkbuttons, spinboxes) now visually highlight on keyboard focus via ttk style maps.
- **Window sizing**: `center_window` now supports `preserve_size` mode and max screen-ratio caps; all windows are non-resizable.
- **Larger text inputs**: `TEntry`, `TSpinbox`, and `TCombobox` styles now use the full base font size with increased padding (6 px).
- **Results dialog layout**: redesigned with a left panel (magnitudes, statistics, solver info, output files) and a right panel (plots in a notebook), using grid layout with a fixed-width left column (480 px).
- **Results dialog dimensions**: window is wider (up to 92 % of screen width) and taller (up to 88 % of screen height, max 900 px).
- **Scroll region updates**: deferred via `after_idle` and a 50 ms timer after section toggles, ensuring the scroll region always reflects the actual content size.
- **Configuration dialog**: wider (800 × 700), with collapsible section groups and entry width increased to 30 characters.
- **Main menu**: arrow-key navigation across Solve / Configuration / Information / Quit, with initial focus on Solve.

### Fixed

- Buttons being cut off when window dimensions were too small for the content at certain font sizes or DPI settings.
- Scroll not reaching the bottom of expanded sections due to `bbox("all")` being read before tkinter finished laying out the content.
- Collapsible sections re-packing content at the wrong position; each section now uses a wrapper frame so content always appears directly below its header.

## [0.1.0] - 2026-02-18

### Added

- **ODE Solving Engine** using SciPy (`solve_ivp`, `solve_bvp`) with support for six numerical methods: RK45, RK23, DOP853, Radau, BDF, and LSODA.
- **Predefined equations** defined in YAML: harmonic oscillator, damped oscillator, logistic equation, Van der Pol, pendulum, and more.
- **Custom equation input** with Python-syntax expression parser and validation.
- **Desktop GUI** built with Tkinter/ttk, including dialogs for equation input, parameter configuration, results display, and help.
- **Professional plotting** with Matplotlib: solution plots, phase plots, configurable line styles, markers, fonts, and grid. Export to PNG, JPG, and PDF.
- **Statistical analysis**: mean, RMS, standard deviation, max/min, integral, zero crossings, period, amplitude, and energy estimates.
- **Data export** in CSV and JSON formats.
- **Configuration system** via `.env` file and in-app configuration dialog, covering UI theme, plot style, solver defaults, file paths, and logging.
- **Logging** with configurable log levels (DEBUG through CRITICAL).
- **Cross-platform install/run scripts** (`install.bat`/`install.sh`, `bin/setup.bat`/`bin/setup.sh`, `bin/run.bat`/`bin/run.sh`).
- **Tooltip widgets** for improved UX in the GUI.

[0.2.1]: https://github.com/DOKOS-TAYOS/DifferentialLab/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/DOKOS-TAYOS/DifferentialLab/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/DOKOS-TAYOS/DifferentialLab/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/DOKOS-TAYOS/DifferentialLab/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/DOKOS-TAYOS/DifferentialLab/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/DOKOS-TAYOS/DifferentialLab/releases/tag/v0.1.0
