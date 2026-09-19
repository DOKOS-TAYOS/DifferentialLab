# Getting Started

## Requirements

- Python 3.12 or newer
- Windows, macOS, or Linux
- Tkinter available in your Python runtime
- A project-local virtual environment named `.venv`

The setup scripts create `.venv` automatically. If you do the setup manually,
create the environment yourself before installing the package.

## Installation

### First-time clone and setup

These scripts are intended for users who do not already have the repository.
They clone the project, run setup, and create a desktop shortcut where possible.

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

For normal application use:

```bat
bin\setup.bat
```

Linux/macOS:

```bash
chmod +x bin/setup.sh
./bin/setup.sh
```

For development work, install the development extras:

```bat
bin\setup.bat --dev
```

Linux/macOS:

```bash
./bin/setup.sh --dev
```

### Manual setup

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"   # tests, linting, and typing-related development work
pip install -e ".[docs]"  # Sphinx documentation build dependencies
```

## Run the application

Windows:

```bat
bin\run.bat
```

Linux/macOS:

```bash
./bin/run.sh
```

Run modes:

```bat
bin\run.bat --prod        :: Windows production mode, uses pythonw
bin\run.bat --dev         :: Windows foreground mode, useful for reading errors
bin\run.bat --background  :: Windows background mode, writes logs/run.log
```

```bash
./bin/run.sh --prod        # Linux/macOS background mode
./bin/run.sh --dev         # Linux/macOS foreground mode
./bin/run.sh --background  # Linux/macOS background mode
```

Direct run from an activated environment:

```bash
python src/main_program.py
```

Installed entry point:

```bash
differential-lab
```

## First Run Checklist

1. Open `Solve Equation`.
2. Run a predefined equation to confirm the solver and result dialog work.
3. Open `Settings` and save your preferred UI/plot defaults.
4. Confirm `output/` receives CSV, JSON, plot, or animation exports.
5. Optionally open `Advanced Problems` and run one module with default parameters.

## Build Docs Locally

```bash
pip install -e ".[docs]"
cd docs
make html      # Linux/macOS
make.bat html  # Windows
```

Open `docs/_build/html/index.html`.

## Troubleshooting

- Virtual environment missing:
  - Run `bin\setup.bat` on Windows or `./bin/setup.sh` on Linux/macOS.
- `ModuleNotFoundError` on direct run:
  - Run from the project root, or use `differential-lab` after `pip install -e .`.
- Tkinter unavailable:
  - Install Tk support for your Python distribution.
- Windows production mode starts but no window appears:
  - Run `bin\run.bat --dev` to see the error in the terminal.
- Linux/macOS background mode starts but no window appears:
  - Run `./bin/run.sh --dev`, or inspect `logs/run.log`.
