# Getting Started

## Requirements

- **Python 3.12** or higher
- Windows 10/11, macOS 10.14+, or Linux
- 4 GB RAM minimum

## Installation

### Quick Install (Recommended)

**Windows:**

```
install.bat
```

**Linux / macOS:**

```bash
chmod +x install.sh
./install.sh
```

These scripts clone the repository (if needed) and run setup, which creates a
virtual environment, installs all dependencies, and generates a default `.env`
file.

### Manual Setup (existing clone)

If you need to clone the repository first:

```bash
git clone https://github.com/DOKOS-TAYOS/DifferentialLab.git
cd DifferentialLab
```

If you already have the repository cloned:

**Windows:**

```
bin\setup.bat
```

**Linux / macOS:**

```bash
chmod +x bin/setup.sh
./bin/setup.sh
```

Alternatively, set up manually:

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # Linux / macOS
   source .venv/bin/activate
   ```

2. Install the package in editable mode:

   ```bash
   pip install -e .
   ```

   For development (tests, linting, type-checking):

   ```bash
   pip install -e ".[dev]"
   ```

   For building documentation:

   ```bash
   pip install -e ".[docs]"
   ```

3. Copy the example configuration:

   ```bash
   cp .env.example .env
   ```

## Running the Application

**With the helper scripts:**

```bash
# Windows
bin\run.bat

# Linux / macOS
./bin/run.sh
```

**Directly via Python** (from project root):

```bash
python src/main_program.py
```

**Via the installed entry point:**

```bash
differential-lab
```

## Dependencies

| Package       | Version     | Purpose                    |
|---------------|-------------|----------------------------|
| NumPy         | >= 2.0, < 3.0 | Numerical computations   |
| Matplotlib    | >= 3.10, < 4.0 | Plotting and visualization |
| SciPy         | >= 1.15, < 2.0 | ODE solving engine       |
| python-dotenv | >= 1.0, < 2.0 | Environment configuration |
| PyYAML        | >= 6.0, < 7.0 | Equation definitions      |

## Building the Documentation

```bash
pip install -e ".[docs]"
cd docs
```

**Linux / macOS:**

```bash
make html
```

**Windows:**

```
make.bat html
```

The built HTML will be in `docs/_build/html/`. Open `index.html` in a browser
to view it locally.

## Troubleshooting

**Virtual environment not found**

If `bin\run.bat` or `./bin/run.sh` reports that the virtual environment is
missing, run `bin\setup.bat` (Windows) or `./bin/setup.sh` (Linux/macOS) first
to create it and install dependencies.

**ModuleNotFoundError when running directly**

If you get `ModuleNotFoundError` when running `python src/main_program.py`,
ensure you are in the project root directory (the folder containing `src/`).
Alternatively, use the installed entry point: `differential-lab` (after
`pip install -e .`).

**Git not found (install scripts)**

The `install.bat` and `install.sh` scripts require Git. Install it from
<https://git-scm.com/downloads> before running them.
