![DifferentialLab Logo](_static/DifferentialLab_logo.png)

# DifferentialLab

**Numerical ODE, difference equation, and PDE solver with a graphical interface for scientists, engineers, and students.**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://github.com/DOKOS-TAYOS/DifferentialLab/blob/main/license.md)
[![Version](https://img.shields.io/badge/version-0.2.1-blue.svg?style=for-the-badge)](https://github.com/DOKOS-TAYOS/DifferentialLab)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

[🐛 **Report Bug**](https://github.com/DOKOS-TAYOS/DifferentialLab/issues) • [💡 **Request Feature**](https://github.com/DOKOS-TAYOS/DifferentialLab/issues)

---

DifferentialLab solves ordinary differential equations, difference equations,
and PDEs numerically using SciPy's integration engine. It provides a desktop
GUI built with Tkinter/ttk, predefined equations from physics and engineering,
a custom expression editor, publication-ready matplotlib plots, statistical
analysis of solutions, and a function transforms module (Fourier, Laplace,
Taylor, Hilbert, Z-transform).

## Key Features

- **ODEs**: Six numerical methods (RK45, RK23, DOP853, Radau, BDF, LSODA)
- **Difference equations**: Recurrence relations (geometric growth, logistic map, Fibonacci, etc.)
- **PDEs**: 2D elliptic solver (Poisson, Laplace)
- **Vector ODEs**: Coupled systems with animation and 3D visualization
- **Predefined equations**: Harmonic oscillator, pendulum, Van der Pol, Lorenz, Lotka-Volterra, and more
- **Function transforms**: Fourier (FFT), Laplace, Taylor series, Hilbert, Z-transform
- **Custom equations**: Python syntax with safe evaluation
- **Plots**: Solution curves, phase portraits, surface/contour for PDEs, vector animation
- **Statistics**: Mean, RMS, period, amplitude, energy, residual error metrics
- **Export**: CSV, JSON, PNG/JPG/PDF, MP4 animation

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

```{toctree}
:maxdepth: 2
:caption: Documentation
:hidden:

getting-started
user-guide
configuration
architecture
api/index
changelog
```
