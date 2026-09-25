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
They clone the project and run setup. On Linux, they also add an application-menu
launcher and a Desktop launcher when the XDG Desktop directory exists.

Windows:

```bat
install.bat
```

Linux/macOS:

```bash
./install.sh
```

### Existing clone

For normal application use:

```bat
bin\setup.bat
```

Linux/macOS:

```bash
./bin/setup.sh
```

The setup and run scripts use `.venv` directly, so manual activation is not
required. You can still activate it for manual development commands. Using
`./bin/run.sh` or `.venv/bin/differential-lab` also avoids import and dependency
mismatches caused by running with the system Python.

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

Manual run from an activated environment (activation is optional for the scripts):

```bash
python src/main_program.py
```

Installed entry point:

```bash
differential-lab
```

## First Run Checklist

1. Open `Solve Equation`.
2. Use the searchable predefined-equation browser, select an equation, and choose `Continue`.
3. Review the retained Configuration form, then choose `Solve`.
4. Inspect the Results workspace summary, metrics, diagnostics, and available visualization controls.
5. Use the fixed footer's CSV or JSON action, the Matplotlib toolbar for a static figure, or MP4 only from an animated view.
6. Open `Settings` and save your preferred UI/plot defaults.
7. Optionally open `Advanced Problems` and run one module with default parameters.

## Build Docs Locally

```bash
pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
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
