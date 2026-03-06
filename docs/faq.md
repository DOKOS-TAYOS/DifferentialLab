# FAQ and Troubleshooting

## The app does not start

- Confirm Python 3.12+ is installed.
- Run setup script (`bin/setup.bat` or `./bin/setup.sh`).
- Activate virtual environment and run `python src/main_program.py` to inspect direct errors.

## Tkinter import errors

The GUI requires Tkinter support in your Python installation.

- Linux: install `python3-tk` (or distro equivalent).
- Windows/macOS: reinstall Python ensuring Tk support is included.

## `ModuleNotFoundError` when running manually

Run from project root (directory that contains `src/`).

Alternative:

```bash
pip install -e .
differential-lab
```

## Complex problem runs too slowly

- Reduce grid size (`nx`, `ny`) and simulation horizon (`t_max`).
- Increase output sampling interval where available.
- Start from defaults and scale up incrementally.

## Numerical blow-up or NaNs

- Reduce `dt`.
- Use more stable model settings.
- Validate boundary-condition selection.
- Check drift/invariant metrics in result dialogs.

## Update checks fail

- Disable in `.env` (`CHECK_UPDATES=false`) for offline/restricted environments.
- Verify `UPDATE_CHECK_URL` is reachable if enabled.

## Sphinx docs build errors

- Install docs extras: `pip install -e ".[docs]"`.
- Build from `docs/` directory.
- Remove stale `docs/_build/` if needed and rebuild.
