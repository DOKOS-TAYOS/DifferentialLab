# DifferentialLab

**Numerical ODE solver with a graphical interface for scientists, engineers, and students.**

---

## Features

- **Numerical ODE solving** powered by SciPy (`solve_ivp`, `solve_bvp`)
- **Predefined equations**: harmonic oscillator, damped oscillator, logistic equation, Van der Pol, pendulum, and more
- **Custom equations**: write any ODE in Python syntax
- **Professional plots** with matplotlib (publication-ready, fully configurable)
- **Statistics and magnitudes**: mean, max/min, period, energy, RMS, integral, amplitude
- **Multiple output formats**: CSV data, JSON statistics, PNG plots
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
