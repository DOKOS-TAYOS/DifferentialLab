# Getting Started

## Requirements

- Python 3.12 or newer
- Windows, macOS, or Linux
- Tkinter available in your Python runtime

## Installation

### Recommended bootstrap scripts

Windows:

```bat
install.bat
```

Linux/macOS:

```bash
chmod +x install.sh
./install.sh
```

These scripts set up the repository and execute environment setup.

### Manual setup (already cloned repository)

Windows:

```bat
bin\setup.bat
```

Linux/macOS:

```bash
chmod +x bin/setup.sh
./bin/setup.sh
```

Or manually:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"   # tests + linting + typing
pip install -e ".[docs]"  # documentation build dependencies
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

Direct run:

```bash
python src/main_program.py
```

Installed entry point:

```bash
differential-lab
```

## First run checklist

1. Open `Solve` and run a predefined equation to validate solver output.
2. Open `Configuration` and save your preferred UI/plot defaults.
3. Confirm `output/` receives CSV/JSON/plot exports.
4. Optionally open `Complex Problems` and run one plugin with default parameters.

## Build docs locally

```bash
pip install -e ".[docs]"
cd docs
make html      # Linux/macOS
make.bat html  # Windows
```

Open `docs/_build/html/index.html`.

## Troubleshooting

- Virtual environment missing:
  - Run `bin/setup.bat` or `./bin/setup.sh`.
- `ModuleNotFoundError` on direct run:
  - Run from project root, or use `differential-lab` after `pip install -e .`.
- Tkinter unavailable:
  - Install Tk for your OS/Python distribution.
- `pythonw` not found on Linux/macOS in helper script:
  - Run `python src/main_program.py` directly.
