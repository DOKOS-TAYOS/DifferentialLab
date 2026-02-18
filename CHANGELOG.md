# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/DOKOS-TAYOS/ode-solver/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DOKOS-TAYOS/ode-solver/releases/tag/v0.1.0
