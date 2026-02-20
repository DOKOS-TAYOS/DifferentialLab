# DifferentialLab

**Numerical ODE solver with a graphical interface for scientists, engineers, and students.**

DifferentialLab solves ordinary differential equations numerically using
SciPy's integration engine. It provides a desktop GUI built with Tkinter/ttk,
predefined equations from physics and engineering, a custom expression editor,
publication-ready matplotlib plots, and statistical analysis of solutions.

## Key Features

- Six numerical methods (RK45, RK23, DOP853, Radau, BDF, LSODA)
- Eight predefined ODEs (harmonic oscillator, pendulum, Van der Pol, etc.)
- Custom equations in Python syntax with safe evaluation
- Configurable plots, markers, and fonts via `.env` or in-app dialog
- CSV, JSON, and image export
- Statistical magnitudes: mean, RMS, period, amplitude, energy, and more

---

## User Documentation

- [Getting Started](getting-started.md) -- installation, setup, first run
- [User Guide](user-guide.md) -- walk-through of the complete workflow
- [Configuration Reference](configuration.md) -- every `.env` setting explained

## Developer Documentation

- [Architecture](architecture.md) -- module structure, data flow, design decisions
- [API Reference](api/index.md) -- auto-generated from source docstrings

## Project

- [Changelog](changelog.md) -- release history

<!-- Sphinx toctree (hidden, only drives sidebar navigation in built docs) -->

```{toctree}
:hidden:
:caption: User Documentation

getting-started
user-guide
configuration
```

```{toctree}
:hidden:
:caption: Developer Documentation

architecture
api/index
```

```{toctree}
:hidden:
:caption: Project

changelog
```
