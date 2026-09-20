<div align="center">

![DifferentialLab Logo](images/DifferentialLab_logo.png)

# DifferentialLab

Desktop numerical lab for ODEs, vector ODEs, difference equations, scalar 2D/3D PDEs,
vector 2D PDEs, function transforms, and specialized scientific simulation workflows.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](license.md)
[![Status](https://img.shields.io/badge/status-Beta-orange.svg?style=for-the-badge)](https://github.com/DOKOS-TAYOS/DifferentialLab)
[![CI](https://github.com/DOKOS-TAYOS/DifferentialLab/actions/workflows/python-app.yml/badge.svg)](https://github.com/DOKOS-TAYOS/DifferentialLab/actions/workflows/python-app.yml)

[Documentation](docs/index.md) |
[Report Bug](https://github.com/DOKOS-TAYOS/DifferentialLab/issues) |
[Request Feature](https://github.com/DOKOS-TAYOS/DifferentialLab/issues)

</div>

## Current State

- Python: `>=3.12`
- Package name: `differential-lab`
- GUI: Tkinter/ttk with embedded Matplotlib figures
- Predefined catalog: 125 equations loaded from YAML
  - 48 ODEs
  - 50 vector ODE systems
  - 10 difference equations
  - 12 2D PDE examples
  - 4 scalar 3D PDE examples
  - 1 coupled Vector PDE example
- Advanced Problems: 9 registered, lazily loaded plugins
- Quality tooling: `pytest`, `ruff`, and a repo-local `pyright` configuration
- Documentation: Sphinx + MyST under `docs/`

## What It Solves

- Scalar ODEs with SciPy IVP integrators, safe terminal/directional events, and a
  dedicated endpoint-BVP backend with shooting retained for true multipoint conditions
- Vector ODE systems with component-aware notation and visualizations
- Difference equations and recurrence systems
- Scalar linear elliptic 2D PDE workflows with rectangular or masked domains,
  structured Dirichlet/Neumann/Robin boundaries, non-duplicated periodic axes,
  component-aware ellipticity validation, and algebraic diagnostics
- Linear strongly elliptic Vector PDE systems in 2D with matrix-valued coupling,
  component-aware boundaries, sparse block assembly, component/magnitude views,
  component/magnitude spatial axis sweeps, and
  Cartesian quiver, stream, and radial/tangential visualizations for exactly
  two-component systems
- Scalar linear elliptic PDEs on rectangular 3D grids with all six second-order
  coefficients, Dirichlet/Neumann/Robin faces, periodic axes, sparse diagnostics,
  pre-run memory advice, selectable coordinate-labelled orthogonal result slices, and
  selectable spatial axis sweeps
- Cartesian scalar PDE views plus polar re-sampling for visualization; coordinate transforms
  are display-only and never alter the solved equation
- Function transforms: Fourier, Laplace, Taylor, Hilbert, and Z-transform
- Advanced Problems subsystem with specialized models, custom UI, solvers, diagnostics, and result dialogs

## Core Features

- Predefined equation catalog in `src/config/equations/*.yaml`
- Safe expression parsing with AST validation
- Unified `f[...]` notation (`f[0]`, `f[1]`, `f[i,k]`)
- Interactive result dialogs with derivative/component selection and dynamic redraw
- CSV, JSON, and static figure exports, plus MP4 export for every interactive animation result view
- Environment-backed configuration through `.env` and the in-app `Settings` dialog
- Rotating application logs with optional console output

## Security

- Report security issues privately using [SECURITY.md](SECURITY.md).
- Local `.env`, generated outputs, update-check state, logs, caches, and virtual environments
  are ignored by git.
- GitHub checks include Dependabot, CodeQL default setup, and a weekly Python dependency audit.

## Advanced Problems

`Advanced Problems` is a nine-plugin subsystem. Internally, its Python package
is named `complex_problems`; each plugin provides its own configuration dialog,
solver, structured result, and result dialog.

Gravitational N-Body Dynamics adds curated Figure-eight, Lagrange equilateral, and
Pythagorean three-body studies plus configurable 2D/3D systems of 2--100 point masses.
It uses self-consistent user-selected units; positive softening epsilon is explicitly
the Plummer-softened model, while epsilon zero is exact Newtonian point gravity.

Fermi-Pasta-Ulam-Tsingou Experiment is a dedicated fixed-end alpha/beta chain workflow.
It uses Velocity Verlet for recommended long-time exploration and offers legacy RK4 for
historical comparison. Its exact nonlinear Hamiltonian is distinct from the linear
normal-mode energy diagnostic; the result notebook includes modal recurrence fidelity,
entropy/participation, strain structures, phase space, and sequential recurrence scaling.

Current modules:

- `coupled_oscillators`: 1D coupled oscillator chains and FPUT-style variants
- `membrane_2d`: 2D nonlinear membrane lattice
- `nonlinear_waves`: NLSE and KdV pseudo-spectral propagation
- `schrodinger_td`: 1D/2D time-dependent Schrodinger solver
- `antenna_radiation`: far-field patterns and antenna metrics
- `aerodynamics_2d`: 2D incompressible obstacle-flow approximations
- `pipe_flow`: steady and transient 1D pipe-flow models
- `fput_experiment`: dedicated Fermi-Pasta-Ulam-Tsingou recurrence and strain studies

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
python -m pip_audit
```

Run `pyright` when it is installed in your environment.

Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License. See [license.md](license.md).

Asset provenance and attribution notes: [NOTICE](NOTICE).

Third-party licenses: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

Citation metadata: [CITATION.cff](CITATION.cff).
