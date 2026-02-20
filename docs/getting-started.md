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

These scripts create a virtual environment, install all dependencies, and
generate a default `.env` file.

### Manual Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/DOKOS-TAYOS/DifferentialLab.git
   cd DifferentialLab
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # Linux / macOS
   source .venv/bin/activate
   ```

3. Install the package in editable mode:

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

4. Copy the example configuration:

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

**Directly via Python:**

```bash
cd src
python main_program.py
```

**Via the installed entry point:**

```bash
differential-lab
```

## Dependencies

| Package       | Version  | Purpose                    |
|---------------|----------|----------------------------|
| NumPy         | >= 2.0   | Numerical computations     |
| Matplotlib    | >= 3.10  | Plotting and visualization |
| SciPy         | >= 1.15  | ODE solving engine         |
| python-dotenv | >= 1.0   | Environment configuration  |
| PyYAML        | >= 6.0   | Equation definitions       |

## Building the Documentation

```bash
pip install -e ".[docs]"
cd docs
make html
```

The built HTML will be in `docs/_build/html/`. Open `index.html` in a browser
to view it locally.

On Windows without `make`, use the batch script:

```
cd docs
make.bat html
```
