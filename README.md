<div align="center">

![DifferentialLab Logo](images/DifferentialLab_logo.png)

# DifferentialLab

Desktop numerical lab for ODEs, vector ODEs, difference equations, 2D PDEs,
function transforms, and specialized scientific simulation workflows.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](license.md)
[![Version](https://img.shields.io/badge/version-0.4.1-blue.svg?style=for-the-badge)](https://github.com/DOKOS-TAYOS/DifferentialLab)
[![Status](https://img.shields.io/badge/status-Beta-orange.svg?style=for-the-badge)](https://github.com/DOKOS-TAYOS/DifferentialLab)

[Documentation](docs/index.md) |
[Report Bug](https://github.com/DOKOS-TAYOS/DifferentialLab/issues) |
[Request Feature](https://github.com/DOKOS-TAYOS/DifferentialLab/issues)

</div>

## Current State

- App version: `0.4.1`
- Python: `>=3.12`
- Package name: `differential-lab`
- GUI: Tkinter/ttk with embedded Matplotlib figures
- Predefined catalog: 120 equations loaded from YAML
  - 48 ODEs
  - 50 vector ODE systems
  - 10 difference equations
  - 12 2D PDE examples
- Complex problem plugins: 7 registered modules
- Quality tooling: `pytest`, `ruff`, and a repo-local `pyright` configuration
- Documentation: Sphinx + MyST under `docs/`

## What It Solves

- Scalar ODEs with SciPy integrators (`RK45`, `RK23`, `DOP853`, `Radau`, `BDF`, `LSODA`)
- Vector ODE systems with component-aware notation and visualizations
- Difference equations and recurrence systems
- 2D PDE workflows with rectangular or masked domains, Dirichlet/Neumann boundaries,
  and several finite-difference operators
- Function transforms: Fourier, Laplace, Taylor, Hilbert, and Z-transform
- Specialized complex-problem models with custom UI, solvers, diagnostics, and result dialogs

## Core Features

- Predefined equation catalog in `src/config/equations/*.yaml`
- Safe expression parsing with AST validation
- Unified `f[...]` notation (`f[0]`, `f[1]`, `f[i,k]`)
- Interactive result dialogs with derivative/component selection and dynamic redraw
- CSV, JSON, static figure, and MP4 animation exports where supported
- Environment-backed configuration through `.env` and the in-app `Settings` dialog
- Rotating application logs with optional console output

## Complex Problems

The main menu exposes these through `Advanced Problems`; internally this is the
`Complex Problems` plugin subsystem.

Current modules:

- `coupled_oscillators`: 1D coupled oscillator chains and FPUT-style variants
- `membrane_2d`: 2D nonlinear membrane lattice
- `nonlinear_waves`: NLSE and KdV pseudo-spectral propagation
- `schrodinger_td`: 1D/2D time-dependent Schrodinger solver
- `antenna_radiation`: far-field patterns and antenna metrics
- `aerodynamics_2d`: 2D incompressible obstacle-flow approximations
- `pipe_flow`: steady and transient 1D pipe-flow models

## Requirements

- Python `>=3.12`
- Windows 10/11, macOS, or Linux
- Tkinter available in the Python runtime
- A virtual environment named `.venv` is the expected local setup

## Quick Start

### First-time setup

Windows:

```bat
install.bat
```

Linux/macOS:

```bash
chmod +x install.sh
./install.sh
```

### Existing clone

Windows:

```bat
bin\setup.bat
bin\run.bat
```

Linux/macOS:

```bash
chmod +x bin/setup.sh bin/run.sh
./bin/setup.sh
./bin/run.sh
```

For development dependencies:

```bat
bin\setup.bat --dev
```

or on Linux/macOS:

```bash
./bin/setup.sh --dev
```

Direct run from an activated environment:

```bash
python src/main_program.py
```

Installed console entry point:

```bash
differential-lab
```

## Documentation

- [Documentation Home](docs/index.md)
- [Getting Started](docs/getting-started.md)
- [User Guide](docs/user-guide.md)
- [Complex Problems Guide](docs/complex-problems.md)
- [Configuration Reference](docs/configuration.md)
- [Architecture](docs/architecture.md)
- [Developer Guide](docs/developer-guide.md)
- [Testing](docs/testing.md)
- [API Reference](docs/api/index.md)

To build docs locally:

```bash
pip install -e ".[docs]"
cd docs
make html      # Linux/macOS
make.bat html  # Windows
```

Output directory: `docs/_build/html/`.

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Recommended checks before sharing changes:

```bash
ruff check . --fix
ruff format .
pytest
pyright
```

Run `pyright` when it is installed in your environment.

Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License. See [license.md](license.md).

Third-party licenses: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
