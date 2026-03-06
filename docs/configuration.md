# Configuration Reference

DifferentialLab reads configuration from `.env` in the project root.

- Use `.env.example` as template.
- Or use the in-app `Configuration` dialog (recommended).
- On startup, values are validated against `ENV_SCHEMA` (`src/config/env.py`).
- Invalid values are automatically replaced with defaults and logged.

## How values are interpreted

- `bool`: accepts `true/false`, `1/0`, `yes/no`
- `int` and `float`: parsed numerically with range checks where defined
- `str`: non-empty; some keys enforce enumerated options

## UI settings

| Key | Default |
|---|---|
| `UI_BACKGROUND` | `#181818` |
| `UI_FOREGROUND` | `#CCCCCC` |
| `UI_BUTTON_BG` | `#1F1F1F` |
| `UI_BUTTON_WIDTH` | `14` |
| `UI_BUTTON_FG` | `lime green` |
| `UI_BUTTON_FG_CANCEL` | `red2` |
| `UI_BUTTON_FG_ACCENT2` | `yellow` |
| `UI_FONT_SIZE` | `16` |
| `UI_FONT_FAMILY` | `Bahnschrift` |
| `UI_PADDING` | `8` |

## Tooltip settings

| Key | Default |
|---|---|
| `UI_TOOLTIP_DELAY_MS` | `500` |
| `UI_TOOLTIP_WRAPLENGTH` | `350` |
| `UI_TOOLTIP_PADX` | `8` |
| `UI_TOOLTIP_PADY` | `4` |

Tooltip font size is derived automatically from `UI_FONT_SIZE`:

- `tooltip_size = max(6, round(UI_FONT_SIZE * 0.5))`
- There is no separate `.env` key for tooltip font size.

## Plot settings

### Layout and style

| Key | Default |
|---|---|
| `PLOT_FIGSIZE_WIDTH` | `12` |
| `PLOT_FIGSIZE_HEIGHT` | `6` |
| `DPI` | `100` |
| `PLOT_SHOW_TITLE` | `true` |
| `PLOT_SHOW_GRID` | `true` |
| `PLOT_LINE_COLOR` | `royalblue` |
| `PLOT_LINE_WIDTH` | `1.5` |
| `PLOT_LINE_STYLE` | `-` |
| `PLOT_COLOR_SCHEME` | `Set1` |

### Markers

| Key | Default |
|---|---|
| `PLOT_MARKER_FORMAT` | `o` |
| `PLOT_MARKER_SIZE` | `3` |
| `PLOT_MARKER_FACE_COLOR` | `crimson` |
| `PLOT_MARKER_EDGE_COLOR` | `crimson` |

### Phase-space and 3D/contour

| Key | Default |
|---|---|
| `PLOT_PHASE_START_COLOR` | `green` |
| `PLOT_PHASE_END_COLOR` | `red` |
| `PLOT_PHASE_MARKER_SIZE` | `8` |
| `PLOT_SURFACE_CMAP` | `viridis` |
| `PLOT_CONTOUR_LEVELS` | `20` |
| `PLOT_GRID_ALPHA` | `0.3` |
| `PLOT_SURFACE_ALPHA` | `0.9` |
| `PLOT_COLORBAR_SHRINK` | `0.6` |

### Animation

| Key | Default |
|---|---|
| `PLOT_ANIMATION_LINE_WIDTH` | `2.0` |
| `PLOT_VLINES_LINE_WIDTH` | `1.5` |
| `PLOT_VLINES_ALPHA` | `0.6` |
| `PLOT_ANIMATION_Y_MARGIN` | `0.1` |
| `ANIMATION_MAX_FPS` | `30` |

## Matplotlib font settings

| Key | Default |
|---|---|
| `FONT_FAMILY` | `serif` |
| `FONT_TITLE_SIZE` | `xx-large` |
| `FONT_TITLE_WEIGHT` | `semibold` |
| `FONT_AXIS_SIZE` | `16` |
| `FONT_AXIS_STYLE` | `italic` |
| `FONT_TICK_SIZE` | `12` |

## Solver defaults

| Key | Default |
|---|---|
| `SOLVER_MAX_STEP` | `0.0` |
| `SOLVER_RTOL` | `1e-8` |
| `SOLVER_ATOL` | `1e-10` |
| `SOLVER_NUM_POINTS` | `1000` |

Supported methods in UI:

- `RK45`
- `RK23`
- `DOP853`
- `Radau`
- `BDF`
- `LSODA`

## Logging and update checks

| Key | Default |
|---|---|
| `LOG_LEVEL` | `INFO` |
| `LOG_FILE` | `differential_lab.log` |
| `LOG_CONSOLE` | `false` |
| `CHECK_UPDATES` | `true` |
| `UPDATE_CHECK_INTERVAL_DAYS` | `7` |
| `CHECK_UPDATES_FORCE` | `false` |
| `UPDATE_CHECK_URL` | `https://raw.githubusercontent.com/DOKOS-TAYOS/DifferentialLab/main/pyproject.toml` |

## Practical recommendations

- Keep solver tolerances strict for stiff/nonlinear systems.
- `SOLVER_NUM_POINTS` profile:
  - Quick/exploratory: `1000-5000`
  - High-resolution/publication: `20000-100000`
- Keep `ANIMATION_MAX_FPS` moderate (20-30) to avoid UI saturation.
- Enable `LOG_CONSOLE=true` while debugging.

## Source of truth

If this page and runtime behavior differ, runtime behavior is authoritative.
The canonical schema lives in `src/config/env.py` (`ENV_SCHEMA`).
