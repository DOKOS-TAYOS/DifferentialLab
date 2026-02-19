# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/DOKOS-TAYOS/DifferentialLab/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/DOKOS-TAYOS/DifferentialLab/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/DOKOS-TAYOS/DifferentialLab/releases/tag/v0.1.0
