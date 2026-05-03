# User Guide

This guide covers the normal workflow: choose or define a problem, configure
parameters, solve, inspect results, and export outputs.

## Main Menu

Buttons:

- `Solve Equation`: general equation workflows
- `Function Transform`: scalar function transform workflows
- `Advanced Problems`: specialized complex-problem models with dedicated UI and visualizations
- `Help & About`: in-app help panel
- `Settings`: edit `.env` values through the GUI
- `Exit`: close the application

## Solve Workflow

### 1. Choose an equation type

The standard solver path currently supports:

- ODE
- Difference equation
- PDE
- Vector ODE

You can use the predefined YAML catalog or write custom expressions.
The current catalog has 120 entries:

| Type | Entries |
|---|---:|
| ODE | 48 |
| Vector ODE | 50 |
| Difference equation | 10 |
| PDE | 12 |

The catalog files live in `src/config/equations/`.

### 2. Define the equation

For custom expressions:

- ODE notation: `f[0]`, `f[1]`, `f[2]`, ...
- Vector notation: `f[i,k]`, where `i` is the component and `k` is the derivative order
- Difference notation: `f[0]` for the current term and `n` for the index
- PDE expressions may use variables such as `x`, `y`, `f`, `fx`, `fy`, `fxx`, `fxy`, and `fyy`

Typical safe math functions are available (`sin`, `cos`, `exp`, `log`, `sqrt`, etc.).

### 3. Configure numeric parameters

Common settings include:

- domain bounds and output sample points
- initial values or boundary values
- ODE solver method
- statistics to compute
- PDE operator and visualization mode

For 2D PDEs, the current UI supports rectangular domains and optional custom
mask expressions. Boundary conditions can be Dirichlet or Neumann on rectangular
edges, or on the contour of a masked domain.

### 4. Solve and inspect

Result dialogs include:

- metadata and solver quality information
- selected statistics
- interactive plots/tabs
- derivative, component, or axis selection without re-solving
- export options for data and figures

## Function Transform Workflow

In `Function Transform`:

1. Provide an `f(x)` expression and domain.
2. Select transform type.
3. Tune transform-specific parameters when shown.
4. Switch between curve view and coefficient view.
5. Export transformed data and plots.

Available transform types:

- Original function sampling
- Fourier (FFT)
- Laplace on the real axis
- Taylor series
- Hilbert (discrete)
- Z-transform (discrete)

## Complex Problems Workflow

`Advanced Problems` opens the complex-problem selector dialog with a left module
list and a right details panel.

Each plugin has its own:

- configuration UI
- specialized solver
- background execution/loading behavior where useful
- dedicated result dialog and diagnostics

Current modules are:

- `coupled_oscillators`
- `membrane_2d`
- `nonlinear_waves`
- `schrodinger_td`
- `antenna_radiation`
- `aerodynamics_2d`
- `pipe_flow`

Most plugin dialogs include a collapsed `How to configure` section with:

- equation summary
- physical interpretation
- parameter meaning
- expected visualizations

For module details, see [Complex Problems Guide](complex-problems.md).

## Exports

By default, solves write generated files under `output/`.

Common outputs:

- `solution_*.csv`
- `solution_*.json`
- `solution_*.png` or another selected static figure format
- MP4 animations when the active result dialog supports animation

## Configuration Workflow

Use `Settings` in the main menu to edit environment-backed settings.

Categories include:

- UI colors, fonts, padding, and tooltips
- plot style, fonts, phase-space markers, contour/surface style, and animation
- solver defaults and tolerances
- logging and update checks

Saving from the dialog restarts the app so settings apply cleanly.

See [Configuration Reference](configuration.md) for the current key list.
