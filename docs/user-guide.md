# User Guide

This guide walks through a typical workflow: launching the application,
selecting or writing an ODE, configuring solver parameters, and inspecting
results.

## Launching the Application

When DifferentialLab starts you see the **main menu** with four buttons:

- **Solve** -- open the equation selector and solve an ODE.
- **Configuration** -- edit all application settings (theme, plot style,
  solver defaults, logging).
- **Information** -- view help, usage instructions, and reference material.
- **Quit** -- close the application.

Navigate between buttons with the arrow keys and press Enter to activate.

## Solving an Equation

### Step 1: Choose an Equation

Click **Solve** to open the equation dialog, which has two tabs:

#### Predefined Tab

Select from the list on the left.  When an equation is highlighted its
description and parameters appear on the right.  You can adjust parameter
values (e.g. angular frequency, damping coefficient) and select which
derivatives to plot.

Available predefined equations:

| Equation                    | Order | Key Parameters        |
|-----------------------------|-------|-----------------------|
| Simple Harmonic Oscillator  | 2     | angular frequency     |
| Damped Oscillator           | 2     | frequency, damping    |
| Exponential Growth / Decay  | 1     | growth rate           |
| Logistic Equation           | 1     | growth rate, capacity |
| Van der Pol Oscillator      | 2     | nonlinearity          |
| Simple Pendulum             | 2     | gravity, length       |
| RC Circuit (Discharge)      | 1     | resistance, capacitance |
| Free Fall with Drag         | 2     | gravity, drag, mass   |

#### Custom Tab

Write the highest derivative as a Python expression.  Use `y[0]` for the
function, `y[1]` for the first derivative, `y[2]` for the second, and so on.
The independent variable is `x`.

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
- **Initial conditions**: one row per derivative.  Each row has a value
  and an `x₀` point (for multi-point boundary conditions).
- **Solver method**: choose from the dropdown.  A description of the
  selected method is shown below.
- **Statistics**: select which magnitudes to compute (mean, RMS, period,
  amplitude, energy, etc.).

### Step 3: View Results

After clicking **Solve**, the results window shows:

- **Left panel**: computed statistics (magnitudes, extrema, period, energy),
  solver information (method, success status, evaluation count), and paths
  to the exported files.
- **Right panel**: solution plot `y(x)` with selected derivatives.  For
  second-order and higher ODEs, a **Phase Portrait** tab also appears.

The plot is interactive (zoom, pan) via the matplotlib toolbar at the bottom.

## Output Files

Every solve produces three files in the output directory (default `output/`):

| File                          | Contents                              |
|-------------------------------|---------------------------------------|
| `solution_YYYYMMDD_HHMMSS.csv` | Tabular `x, y0, y1, ...` columns     |
| `solution_YYYYMMDD_HHMMSS.json`| Full metadata and computed statistics |
| `solution_YYYYMMDD_HHMMSS.png` | Plot image (format configurable)      |

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

After saving, the application restarts automatically so that changes take
effect.  You can also edit the `.env` file directly with any text editor.

See the [Configuration Reference](configuration.md) for a complete list of all
settings.
