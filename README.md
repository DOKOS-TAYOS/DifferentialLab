<div align="center">

![DifferentialLab Logo](images/DifferentialLab_logo.png)

# DifferentialLab

**Numerical ODE, difference equation, and PDE solver with a graphical interface for scientists, engineers, and students.**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](license.md)
[![Version](https://img.shields.io/badge/version-0.2.1-blue.svg?style=for-the-badge)](https://github.com/DOKOS-TAYOS/DifferentialLab)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)

[📖 **Documentation**](docs/index.md) • [🐛 **Report Bug**](https://github.com/DOKOS-TAYOS/DifferentialLab/issues) • [💡 **Request Feature**](https://github.com/DOKOS-TAYOS/DifferentialLab/issues)

</div>

---

## Features

- **ODEs**: Six methods (RK45, RK23, DOP853, Radau, BDF, LSODA) via SciPy
- **Difference equations**: Recurrence relations (geometric growth, logistic map, Fibonacci, etc.)
- **PDEs**: 2D elliptic solver (Poisson, Laplace)
- **Vector ODEs**: Coupled systems with animation and 3D visualization
- **Predefined equations**: Harmonic oscillator, pendulum, Van der Pol, Lorenz, Lotka-Volterra, and more
- **Function transforms**: Fourier (FFT), Laplace, Taylor series, Hilbert, Z-transform
- **Custom equations**: Write any ODE, difference equation, or PDE in Python syntax
- **Professional plots**: Solution curves, phase portraits, surface/contour, vector animation
- **Statistics**: Mean, max/min, period, energy, RMS, residual error metrics
- **Export**: CSV, JSON, PNG/JPG/PDF, MP4 animation
- **Configurable** via `.env` file or in-app configuration dialog
- **Desktop GUI** built with Tkinter/ttk

## Requirements

- Python 3.12 or higher
- Windows 10/11, macOS 10.14+, or Linux
- 4 GB RAM minimum

## Quick Start

### Installation

**Windows:**
```
install.bat
```

**Linux/macOS:**
```bash
chmod +x install.sh
./install.sh
```

### Manual Setup

1. Clone the repository:
```bash
git clone https://github.com/DOKOS-TAYOS/DifferentialLab.git
cd DifferentialLab
```

2. Run setup:
```bash
# Windows
bin\setup.bat

# Linux/macOS
chmod +x bin/setup.sh
./bin/setup.sh
```

3. Run the application:
```bash
# Windows
bin\run.bat

# Linux/macOS
./bin/run.sh
```

## Configuration

Copy `.env.example` to `.env` and customize:
```bash
cp .env.example .env
```

Or use the in-app **Configuration** dialog to edit settings visually.

Available configuration sections:
- UI theme (colors, fonts, padding)
- Plot style (line, markers, fonts, grid)
- Solver defaults (method, tolerances)
- File paths and output format
- Logging

## Dependencies

| Package        | Version     | Purpose                     |
|----------------|-------------|-----------------------------|
| NumPy          | >= 2.0      | Numerical computations      |
| Matplotlib     | >= 3.10     | Plotting and visualization  |
| SciPy          | >= 1.15     | ODE solving engine          |
| python-dotenv  | >= 1.0      | Environment configuration   |
| PyYAML         | >= 6.0      | Equation definitions        |

## Documentation

Full documentation is built with [Sphinx](https://www.sphinx-doc.org/) and
hosted on Read the Docs.

To build the docs locally:

```bash
pip install -e ".[docs]"
cd docs
make html          # Linux/macOS
make.bat html      # Windows
```

The output will be in `docs/_build/html/`.

Documentation contents:
- **Getting Started** -- installation, setup, first run
- **User Guide** -- walk-through of the complete workflow
- **Configuration Reference** -- every `.env` setting explained
- **Architecture** -- module structure and design decisions
- **API Reference** -- auto-generated from source docstrings

## License

MIT License. See [license.md](license.md).

Third-party licenses: see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Author

**Alejandro Mata Ali**
