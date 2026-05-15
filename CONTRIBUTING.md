# Contributing

Thanks for contributing to DifferentialLab.

## Setup

Use a project-local `.venv`.

Windows:

```bat
bin\setup.bat --dev
.venv\Scripts\activate
```

Linux/macOS:

```bash
./bin/setup.sh --dev
source .venv/bin/activate
```

Manual equivalent:

```bash
pip install -e ".[dev]"
```

Optional documentation toolchain:

```bash
pip install -e ".[docs]"
```

## Before Opening a PR

1. Run formatting and lint fixes:

```bash
ruff check . --fix
ruff format .
```

2. Run tests:

```bash
pytest
```

3. Run type checks when `pyright` is available:

```bash
pyright
```

4. Run the dependency audit when security-sensitive files changed:

```bash
python -m pip_audit
```

5. Update documentation for user-visible changes:

- `README.md`
- `docs/` pages
- `docs/api/` pages if public modules changed
- `CHANGELOG.md`

## Plugin Contributions (`complex_problems`)

If you add a plugin:

- implement package structure (`problem.py`, `ui.py`, `solver.py`, `result_dialog.py`)
- add `model.py` when physical or numerical helper logic is non-trivial
- register it in `src/complex_problems/problem_registry.py`
- add solver and registry tests
- update `docs/complex-problems.md` and `docs/api/complex_problems.rst`
- update `CHANGELOG.md`

## Style

- Include typing in function definitions.
- Keep numerical kernels separated from GUI code.
- Validate UI inputs before solver execution.
- Expose structured result dataclasses with explicit metadata.
- Reuse shared helpers before adding new duplicated dialog logic.
