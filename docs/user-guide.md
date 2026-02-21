# User Guide

This guide walks through a typical workflow: launching the application,
selecting or writing an ODE (or difference equation, PDE, or vector ODE),
configuring solver parameters, and inspecting results.  It also covers the
function transforms feature.

## Launching the Application

When DifferentialLab starts you see the **main menu** with five buttons:

- **Solve** -- open the equation selector and solve an ODE, difference
  equation, PDE, or vector ODE.
- **Transforms** -- apply mathematical transforms (Fourier, Laplace, Taylor,
  Hilbert, Z-transform) to scalar functions.
- **Configuration** -- edit all application settings (theme, plot style,
  solver defaults, logging).
- **Information** -- view help, usage instructions, and reference material.
- **Quit** -- close the application.

Navigate between buttons with the arrow keys and press Enter to activate.

On startup, if a newer version is available and update checking is enabled,
you may be prompted to update (input/, output/, and .env are preserved).

## Solving an Equation

### Step 1: Choose an Equation

Click **Solve** to open the equation dialog.  Choose the equation type:
**ODE**, **Difference (recurrence)**, **PDE (multivariate)**, or **Vector ODE**.

#### Predefined Tab

Select from the list on the left.  When an equation is highlighted its
description and parameters appear on the right.  You can adjust parameter
values (e.g. angular frequency, damping coefficient) and select which
derivatives to plot.

Available predefined equations include:

| Equation                    | Type   | Key Parameters        |
|-----------------------------|--------|-----------------------|
| Simple Harmonic Oscillator  | ODE    | angular frequency     |
| Damped Oscillator           | ODE    | frequency, damping    |
| Van der Pol, Pendulum       | ODE    | nonlinearity, gravity  |
| Lorenz, Lotka-Volterra      | ODE    | system parameters     |
| Geometric Growth            | Diff   | growth rate           |
| Logistic Map, Fibonacci     | Diff   | recurrence params     |
| Poisson 2D, Laplace 2D      | PDE    | domain, grid          |
| Coupled Oscillators         | Vector | coupling, masses      |

#### Custom Tab

**ODEs**: Write the highest derivative as a Python expression.  Use `y[0]`
for the function, `y[1]` for the first derivative, `y[2]` for the second, etc.
The independent variable is `x`.

**Difference equations**: Use `y[0]` for y_n, `y[1]` for y_{n+1}, etc., and
`n` for the index.

**Vector ODEs**: Enter a list of expressions `[f₀'', f₁'', ...]` for coupled
systems.

**PDEs**: Use `x`, `y` (and optionally more) as independent variables.

Available math functions: `sin`, `cos`, `tan`, `exp`, `log`, `log10`, `sqrt`,
`abs`, `sinh`, `cosh`, `tanh`, `arcsin`, `arccos`, `arctan`, `floor`, `ceil`,
`sign`, `heaviside`, `pi`, `e`.

Unicode parameters are supported.  You can type `\u03C9` to get `ω`, or paste
the character directly.

**Example** -- damped oscillator (order 2):

```
Expression:   -2*γ*y[1] - ω**2*y[0]
Parameters:   ω=1.0, γ=0.1
```

### Step 2: Configure Parameters

After clicking **Next**, the parameters dialog appears:

- **Domain**: set `x_min`, `x_max`, and the number of evaluation points.
  Use the `+`/`-` buttons to change points by an order of magnitude.
  For PDEs, set `y_min`/`y_max` and grid points per axis.
- **Initial conditions**: one row per derivative.  Each row has a value
  and an `x₀` point (for multi-point boundary conditions).
- **Solver method**: choose from the dropdown (ODEs only).  A description
  of the selected method is shown below.
- **Statistics**: select which magnitudes to compute (mean, RMS, period,
  amplitude, energy, etc.).
- **Plot type** (PDEs): choose 3D surface or 2D contour.

### Step 3: View Results

After clicking **Solve**, the results window shows:

- **Left panel**: computed statistics (magnitudes, extrema, period, energy,
  residual error metrics), solver information (method, success status,
  evaluation count), and paths to the exported files.  Sections are
  collapsible.
- **Right panel**: solution plot `y(x)` with selected derivatives.  For
  second-order and higher ODEs, a **Phase Portrait** tab appears.  For
  vector ODEs, **Animation** and **3D** tabs are available.  For PDEs,
  a surface or contour plot is shown.

The plot is interactive (zoom, pan) via the matplotlib toolbar at the bottom.
Vector animations can be exported to MP4.

## Function Transforms

Click **Transforms** from the main menu to open the transform dialog.  Enter a
scalar function `f(x)` (e.g. `sin(x)`, `exp(-x**2)`), optionally with
parameters (`a=1.0, b=2`), and choose a transform:

- **Original (f(x))**: plot the function as-is.
- **Fourier (FFT)**: discrete Fourier transform.
- **Laplace (real axis)**: Laplace transform along the real axis.
- **Taylor series**: Taylor expansion around a center point.
- **Hilbert (discrete)**: discrete Hilbert transform.
- **Z-transform (discrete)**: discrete Z-transform.

Use the display mode combobox to switch between **Curve (f vs x)** and
**Coefficients (a_i vs i)**.  Export to CSV or PNG from the dialog.

## Output Files

Every solve produces at least three files in the output directory (default
`output/`):

| File                          | Contents                              |
|-------------------------------|---------------------------------------|
| `solution_YYYYMMDD_HHMMSS.csv` | Tabular `x, y0, y1, ...` columns     |
| `solution_YYYYMMDD_HHMMSS.json`| Full metadata and computed statistics |
| `solution_YYYYMMDD_HHMMSS.png` | Plot image (format configurable)      |

For vector ODE animations, an MP4 file can be exported from the results dialog.

## Editing Configuration

Open **Configuration** from the main menu.  Settings are grouped into
collapsible sections:

- UI Theme
- Plot Style
- Plot Markers
- Plot Fonts
- Solver Defaults
- File Paths
- Logging
- Update Check

After saving, the application restarts automatically so that changes take
effect.  You can also edit the `.env` file directly with any text editor.

See the [Configuration Reference](configuration.md) for a complete list of all
settings.
