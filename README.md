<div align="center">

![DifferentialLab Logo](images/DifferentialLab_logo.png)

# DifferentialLab

Numerical ODE, difference-equation, and PDE solver with a desktop GUI for science and engineering workflows.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](license.md)
[![Version](https://img.shields.io/badge/version-0.4.1-blue.svg?style=for-the-badge)](https://github.com/DOKOS-TAYOS/DifferentialLab)
[![Status](https://img.shields.io/badge/status-Beta-orange.svg?style=for-the-badge)](https://github.com/DOKOS-TAYOS/DifferentialLab)

[Documentation](docs/index.md) |
[Report Bug](https://github.com/DOKOS-TAYOS/DifferentialLab/issues) |
[Request Feature](https://github.com/DOKOS-TAYOS/DifferentialLab/issues)

</div>

## What It Solves

- ODEs with SciPy integrators (`RK45`, `RK23`, `DOP853`, `Radau`, `BDF`, `LSODA`)
- Difference equations (recurrence systems)
- PDEs (elliptic 2D solver and operator-based PDE workflows)
- Vector ODE systems with dedicated visualization modes
- Function transforms (Fourier, Laplace, Taylor, Hilbert, Z-transform)

## Core Features

- Predefined equation catalog loaded from YAML (`config/equations/*.yaml`)
- Custom equation parsing with safe AST validation
- Unified `f[...]` notation (`f[0]`, `f[1]`, `f[i,k]`)
- Interactive result dialogs (derivative selection, phase-space selection, dynamic redraw)
- Export to CSV, JSON, static figures, and MP4 animations
- Configurable UI/plot/solver behavior via `.env` or in-app configuration dialog

## Complex Problems (Plugin Mode)

`Complex Problems` is a plugin-style subsystem for specialized models with custom UI, solver, and result dialogs.

Current modules:

- `coupled_oscillators` (1D coupled oscillators and FPUT variants)
- `membrane_2d` (2D coupled nonlinear membrane)
- `nonlinear_waves` (NLSE and KdV)
- `schrodinger_td` (time-dependent Schrodinger in 1D/2D)
- `antenna_radiation` (far-field patterns and antenna metrics)
- `aerodynamics_2d` (2D incompressible obstacle flow approximations)
- `pipe_flow` (steady and transient 1D pipe-flow models)

## Requirements

- Python `>=3.12`
- Windows 10/11, macOS, or Linux
- Tkinter available in the Python runtime (GUI requirement)

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

Direct run:

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

Run tests:

```bash
pytest
```

Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License. See [license.md](license.md).

Third-party licenses: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
