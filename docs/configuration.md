# Configuration Reference

DifferentialLab reads configuration from `.env` in the project root.

- `.env.example` mirrors the current schema defaults.
- Setup scripts copy `.env.example` to `.env` if `.env` does not exist.
- The in-app `Settings` dialog is the safest way to edit values.
- On startup, values are validated against `ENV_SCHEMA` in `src/config/env.py`.
- Invalid values are replaced with defaults and logged.

## How Values Are Interpreted

- `bool`: accepts `true/false`, `1/0`, `yes/no`
- `int` and `float`: parsed numerically, with range checks where defined
- `str`: must be non-empty; some keys enforce a fixed option list

## UI Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `UI_BACKGROUND` | `str` | `#181818` | Main background color. |
| `UI_FOREGROUND` | `str` | `#CCCCCC` | Default text color. |
| `UI_BUTTON_BG` | `str` | `#1F1F1F` | Button and input background color. |
| `UI_BUTTON_WIDTH` | `int` | `14` | Main-menu button width in characters. |
| `UI_BUTTON_FG` | `str` | `lime green` | Primary button text color. |
| `UI_BUTTON_FG_CANCEL` | `str` | `red2` | Cancel/destructive button text color. |
| `UI_BUTTON_FG_ACCENT2` | `str` | `yellow` | Secondary accent button text color. |
| `UI_FONT_SIZE` | `int` | `16` | Base UI font size in points. |
| `UI_FONT_FAMILY` | `str` | `Bahnschrift` | UI font family. |
| `UI_PADDING` | `int` | `8` | General spacing between UI elements. |

## Tooltip Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `UI_TOOLTIP_DELAY_MS` | `int` | `500` | Delay before showing a tooltip. |
| `UI_TOOLTIP_WRAPLENGTH` | `int` | `350` | Maximum tooltip text width before wrapping. |
| `UI_TOOLTIP_PADX` | `int` | `8` | Horizontal tooltip padding. |
| `UI_TOOLTIP_PADY` | `int` | `4` | Vertical tooltip padding. |

Tooltip font size is derived from `UI_FONT_SIZE`:

```text
tooltip_size = max(6, round(UI_FONT_SIZE * 0.5))
```

There is no separate `.env` key for tooltip font size.

## Plot Settings

### Layout and Style

| Key | Type | Default | Description |
|---|---|---|---|
| `PLOT_FIGSIZE_WIDTH` | `int` | `12` | Figure width in inches. |
| `PLOT_FIGSIZE_HEIGHT` | `int` | `6` | Figure height in inches. |
| `DPI` | `int` | `100` | Figure resolution, valid range `50..1000`. |
| `PLOT_SHOW_TITLE` | `bool` | `true` | Show plot titles. |
| `PLOT_SHOW_GRID` | `bool` | `true` | Draw plot grid lines. |
| `PLOT_LINE_COLOR` | `str` | `royalblue` | Main solution curve color. |
| `PLOT_LINE_WIDTH` | `float` | `1.5` | Main curve line thickness. |
| `PLOT_LINE_STYLE` | `str` | `-` | One of `-`, `--`, `-.`, `:`. |
| `PLOT_COLOR_SCHEME` | `str` | `Set1` | Matplotlib colormap for extra curves. |

### Markers

| Key | Type | Default | Description |
|---|---|---|---|
| `PLOT_MARKER_FORMAT` | `str` | `o` | Marker shape. |
| `PLOT_MARKER_SIZE` | `int` | `3` | Marker size in points. |
| `PLOT_MARKER_FACE_COLOR` | `str` | `crimson` | Marker fill color. |
| `PLOT_MARKER_EDGE_COLOR` | `str` | `crimson` | Marker edge color. |

### Phase-Space and 3D/Contour

| Key | Type | Default | Description |
|---|---|---|---|
| `PLOT_PHASE_START_COLOR` | `str` | `green` | Start marker color in phase-space plots. |
| `PLOT_PHASE_END_COLOR` | `str` | `red` | End marker color in phase-space plots. |
| `PLOT_PHASE_MARKER_SIZE` | `int` | `8` | Start/end marker size. |
| `PLOT_SURFACE_CMAP` | `str` | `viridis` | Colormap for 3D surface and contour plots. |
| `PLOT_CONTOUR_LEVELS` | `int` | `20` | Number of contour levels. |
| `PLOT_GRID_ALPHA` | `float` | `0.3` | Grid-line transparency. |
| `PLOT_SURFACE_ALPHA` | `float` | `0.9` | 3D surface transparency. |
| `PLOT_COLORBAR_SHRINK` | `float` | `0.6` | Colorbar shrink factor. |

### Animation

| Key | Type | Default | Description |
|---|---|---|---|
| `PLOT_ANIMATION_LINE_WIDTH` | `float` | `2.0` | Line width for vector animation plots. |
| `PLOT_VLINES_LINE_WIDTH` | `float` | `1.5` | Width of animation vertical guide lines. |
| `PLOT_VLINES_ALPHA` | `float` | `0.6` | Transparency of animation vertical guide lines. |
| `PLOT_ANIMATION_Y_MARGIN` | `float` | `0.1` | Margin added to animation y-axis limits. |
| `ANIMATION_MAX_FPS` | `int` | `30` | Maximum embedded animation playback FPS. |

## Matplotlib Font Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `FONT_FAMILY` | `str` | `serif` | Matplotlib font family. |
| `FONT_TITLE_SIZE` | `str` | `xx-large` | Plot title font size. |
| `FONT_TITLE_WEIGHT` | `str` | `semibold` | Plot title font weight. |
| `FONT_AXIS_SIZE` | `int` | `16` | Axis label font size. |
| `FONT_AXIS_STYLE` | `str` | `italic` | Axis label font style. |
| `FONT_TICK_SIZE` | `int` | `12` | Tick label font size. |

## Solver Defaults

| Key | Type | Default | Description |
|---|---|---|---|
| `SOLVER_MAX_STEP` | `float` | `0.0` | Maximum step size; `0.0` means automatic. |
| `SOLVER_RTOL` | `float` | `1e-8` | Relative tolerance. |
| `SOLVER_ATOL` | `float` | `1e-10` | Absolute tolerance. |
| `SOLVER_NUM_POINTS` | `int` | `1000` | Number of evaluation points in the output grid. |

Supported ODE methods:

- `RK45`
- `RK23`
- `DOP853`
- `Radau`
- `BDF`
- `LSODA`

## Logging and Update Checks

| Key | Type | Default | Description |
|---|---|---|---|
| `LOG_LEVEL` | `str` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `LOG_FILE` | `str` | `differential_lab.log` | Log file path. Relative paths resolve from the project root. |
| `LOG_MAX_BYTES` | `int` | `1048576` | Maximum active log-file size before rotation. |
| `LOG_BACKUP_COUNT` | `int` | `3` | Number of rotated backup files to keep. |
| `LOG_CONSOLE` | `bool` | `false` | Also print log messages to the terminal. |
| `CHECK_UPDATES` | `bool` | `true` | Check for updates on startup. |
| `UPDATE_CHECK_INTERVAL_DAYS` | `int` | `7` | Days between automatic update checks. |
| `CHECK_UPDATES_FORCE` | `bool` | `false` | Force update checks on every startup. |
| `UPDATE_CHECK_URL` | `str` | `https://raw.githubusercontent.com/DOKOS-TAYOS/DifferentialLab/main/pyproject.toml` | Remote `pyproject.toml` used for version comparison. |

Logging notes:

- `LOG_FILE` can be a simple filename such as `differential_lab.log`.
- `LOG_FILE` can also be a nested relative path such as `logs/app.log`.
- Missing parent directories are created automatically for log files.
- Logs rotate when the active file reaches `LOG_MAX_BYTES`.
- If file logging cannot start, DifferentialLab falls back to console logging.

## Practical Recommendations

- Keep solver tolerances strict for stiff or nonlinear systems.
- `SOLVER_NUM_POINTS` profile:
  - quick/exploratory: `1000-5000`
  - high-resolution/export: `20000-100000`
- Keep `ANIMATION_MAX_FPS` moderate, for example `20-30`.
- Enable `LOG_CONSOLE=true` while debugging startup or configuration problems.
- Keep log rotation enabled unless you intentionally want a single growing log file.

## Source of Truth

If this page and runtime behavior differ, runtime behavior is authoritative.
The canonical schema lives in `src/config/env.py` (`ENV_SCHEMA`).
