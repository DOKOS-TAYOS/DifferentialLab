# FAQ and Troubleshooting

## The app does not start

- Confirm Python 3.12+ is installed.
- Run setup again:
  - Windows: `bin\setup.bat`
  - Linux/macOS: `./bin/setup.sh`
- Run in foreground/dev mode to see errors:
  - Windows: `bin\run.bat --dev`
  - Linux/macOS: `./bin/run.sh --dev`

## Tkinter import errors

The GUI requires Tkinter support in your Python installation.

- Linux: install `python3-tk` or the equivalent package for your distribution.
- Windows/macOS: reinstall Python and ensure Tk support is included.

## `ModuleNotFoundError` when running manually

Run from the project root, the directory that contains `src/`.

Alternative:

```bash
pip install -e .
differential-lab
```

## The Windows production run gives no useful error

`bin\run.bat` defaults to production mode and starts with `pythonw`, which does
not keep a visible console.

Use this while debugging:

```bat
bin\run.bat --dev
```

## Linux/macOS background mode starts but no window appears

Run in foreground mode:

```bash
./bin/run.sh --dev
```

Or inspect:

```text
logs/run.log
```

## Complex problem runs too slowly

- Reduce grid size (`nx`, `ny`) and simulation horizon (`t_max`).
- Increase output sampling interval where available.
- Start from defaults and scale up incrementally.
- For animation-heavy results, reduce stored frames before increasing resolution.

## Numerical blow-up or NaNs

- Reduce `dt`.
- Use more stable model settings.
- Validate boundary-condition selection.
- Check drift/invariant metrics in result dialogs.
- Start from a low-resolution run before increasing grid size.

## Update checks fail

- Disable in `.env` with `CHECK_UPDATES=false` for offline or restricted environments.
- Verify `UPDATE_CHECK_URL` is reachable if checks are enabled.
- Run with `LOG_CONSOLE=true` to see update-check logging in the terminal.

## Logs are not written

- Check `LOG_FILE`, `LOG_MAX_BYTES`, and `LOG_BACKUP_COUNT` in `.env`.
- Relative `LOG_FILE` paths are resolved from the project root.
- Nested paths such as `logs/app.log` are supported.
- If file logging fails, the application falls back to console logging.

## Sphinx docs build errors

- Install docs extras: `pip install -e ".[docs]"`.
- Build from the `docs/` directory.
- Remove stale `docs/_build/` if needed and rebuild.

Windows:

```bat
cd docs
make.bat html
```

Linux/macOS:

```bash
cd docs
make html
```
