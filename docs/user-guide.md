# User Guide

This guide covers the normal workflow: define problem, configure parameters, solve, inspect and export results.

## Main menu

Buttons:

- `Solve`: general equation workflows (ODE, difference equations, PDE, vector ODE)
- `Function Transform`: transform scalar functions
- `Complex Problems`: plugin workflows with dedicated UI and visualizations
- `Information`: in-app help panel
- `Configuration`: edit `.env` values through UI
- `Quit`: close application

## Solve workflow

### 1. Choose equation type

In `Solve`, choose one of:

- ODE
- Difference equation
- PDE
- Vector ODE

You can use predefined equations (YAML catalog) or custom expressions.

### 2. Define equation

For custom expressions:

- ODE notation: `f[0]`, `f[1]`, `f[2]`, ...
- Vector notation: `f[i,k]` (component `i`, derivative order `k`)
- Difference notation: `f[0]` for current term, `n` as index

Typical safe math functions are available (`sin`, `cos`, `exp`, `log`, `sqrt`, etc.).

### 3. Configure numeric parameters

- Domain bounds and sample points
- Initial/boundary values
- Solver method (ODE)
- Statistics to compute
- PDE visualization mode

### 4. Solve and inspect

Result dialog includes:

- statistics and metadata
- interactive plots/tabs
- derivative/axis selection without re-solving
- export options (CSV/JSON/figures and MP4 when applicable)

## Function Transform workflow

In `Function Transform`:

1. Provide `f(x)` expression and optional parameters.
2. Select transform (`Fourier`, `Laplace`, `Taylor`, `Hilbert`, `Z-transform`).
3. Switch between curve view and coefficient view.
4. Export transformed data and plots.

## Complex Problems workflow

`Complex Problems` opens a selector dialog with a left module list and a right details panel.

Each plugin has its own:

- configuration UI
- specialized solver
- dedicated result dialog and diagnostics

Most plugin dialogs include a collapsed `How to configure` section with:

- short equation summary
- physical interpretation
- parameter meaning
- expected visualizations

UI notation uses Unicode math formatting where possible. When a Unicode subscript does not exist (for example theta/phi), the UI uses `base_subscript` style such as `N_θ`, `N_φ`.

For current modules and details, see [Complex Problems Guide](complex-problems.md).

## Exports

By default, solves generate output files under `output/`:

- `solution_*.csv`
- `solution_*.json`
- `solution_*.png` (or selected format)

Animation-capable dialogs can also export MP4.

## Configuration workflow

Use `Configuration` in the main menu to edit environment-backed settings.

Categories include:

- UI look and behavior
- plot style/fonts/animation
- solver defaults and tolerances
- logging and update checks

Saving from the dialog restarts the app so settings apply cleanly.

See [Configuration Reference](configuration.md) for full key details.
