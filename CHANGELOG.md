# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[Unreleased]: https://github.com/DOKOS-TAYOS/DifferentialLab/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/DOKOS-TAYOS/DifferentialLab/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/DOKOS-TAYOS/DifferentialLab/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/DOKOS-TAYOS/DifferentialLab/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/DOKOS-TAYOS/DifferentialLab/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/DOKOS-TAYOS/DifferentialLab/releases/tag/v0.1.0
