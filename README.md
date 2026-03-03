<div align="center">

![DifferentialLab Logo](images/DifferentialLab_logo.png)

# DifferentialLab

**Numerical ODE, difference equation, and PDE solver with a graphical interface for scientists, engineers, and students.**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](license.md)
[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg?style=for-the-badge)](https://github.com/DOKOS-TAYOS/DifferentialLab)
[![Status](https://img.shields.io/badge/status-Beta-orange.svg?style=for-the-badge)](https://github.com/DOKOS-TAYOS/DifferentialLab)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)

[📖 **Documentation**](docs/index.md) • [🐛 **Report Bug**](https://github.com/DOKOS-TAYOS/DifferentialLab/issues) • [💡 **Request Feature**](https://github.com/DOKOS-TAYOS/DifferentialLab/issues)

</div>

---

## Features

- **ODEs**: Six methods (RK45, RK23, DOP853, Radau, BDF, LSODA) via SciPy
- **Difference equations**: Recurrence relations (geometric growth, logistic map, Fibonacci, etc.)
- **PDEs**: 2D elliptic solver (Poisson, Laplace) plus general operator-based PDEs with configurable derivative terms
- **Vector ODEs**: Coupled systems with animation, 3D phase-space trajectories, and surface visualization
- **Unified f-notation**: Write equations using `f[0]` (function), `f[1]` (first derivative), `f[i,k]` (component i, derivative k for vector ODEs)
- **Predefined equations**: Harmonic oscillator, pendulum, Van der Pol, Lorenz, Lotka-Volterra, Duffing, Schrödinger, and more
- **Function transforms**: Fourier (FFT), Laplace, Taylor series, Hilbert, Z-transform
- **Custom equations**: Write any ODE, difference equation, or PDE in Python syntax
- **Interactive result tabs**: Select derivatives to plot, choose phase-space axes, switch visualization modes without re-solving
- **Professional plots**: Solution curves, phase portraits (2D/3D), surface/contour, vector animation
- **Statistics**: Mean, max/min, period, energy, RMS, residual error metrics
- **Export**: CSV, JSON, PNG/JPG/PDF, MP4 animation
- **Configurable** via `.env` file or in-app Configuration dialog
- **Desktop GUI** built with Tkinter/ttk

## Requirements

- Python 3.12 or higher
- Windows 10/11, macOS 10.14+, or Linux
- 4 GB RAM minimum

## Quick Start

### Installation (first-time setup)

**Windows:**
```
install.bat
```

**Linux/macOS:**
```bash
chmod +x install.sh
./install.sh
```

This clones the repository (if needed) and runs setup.

### Manual setup (existing clone)

1. Create virtual environment and install dependencies:
```bash
# Windows
bin\setup.bat

# Linux/macOS
chmod +x bin/setup.sh
./bin/setup.sh
```

2. Run the application:
```bash
# Windows
bin\run.bat

# Linux/macOS
./bin/run.sh
```

Or run directly:
```bash
python src/main_program.py
```

## Configuration

Copy `.env.example` to `.env` and customize, or use the in-app **Configuration** dialog.

Available settings:
- UI theme (colors, fonts, padding)
- Plot style (line, markers, fonts, grid)
- Solver defaults (method, tolerances)
- File paths and output format
- Logging

## Documentation

Full documentation is built with [Sphinx](https://www.sphinx-doc.org/) and hosted on Read the Docs.

To build the docs locally:

```bash
pip install -e ".[docs]"
cd docs
make html          # Linux/macOS
make.bat html      # Windows
```

The output will be in `docs/_build/html/`.

Documentation contents:
- **Getting Started** — installation, setup, first run
- **User Guide** — walk-through of the complete workflow
- **Configuration Reference** — every `.env` setting explained
- **Architecture** — module structure and design decisions
- **API Reference** — auto-generated from source docstrings

## Dependencies

| Package        | Version     | Purpose                     |
|----------------|-------------|-----------------------------|
| NumPy          | >= 2.0      | Numerical computations      |
| Matplotlib     | >= 3.10     | Plotting and visualization  |
| SciPy          | >= 1.15     | ODE solving engine          |
| python-dotenv  | >= 1.0      | Environment configuration   |
| PyYAML         | >= 6.0      | Equation definitions        |

## License

MIT License. See [license.md](license.md).

Third-party licenses: see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Author

**Alejandro Mata Ali**
