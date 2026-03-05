# Configuration Reference

DifferentialLab is configured via a `.env` file in the project root.  Copy
`.env.example` to `.env` to get started, or use the in-app **Configuration**
dialog (Main Menu > Configuration) which writes the file for you.

After saving through the dialog the application restarts automatically.

## UI Theme

| Variable              | Type | Default      | Description                                    |
|-----------------------|------|--------------|------------------------------------------------|
| `UI_BACKGROUND`       | str  | `#181818`    | Main background colour (hex).                  |
| `UI_FOREGROUND`       | str  | `#CCCCCC`    | Default text colour (hex).                     |
| `UI_BUTTON_BG`        | str  | `#1F1F1F`    | Button / input background colour (hex).        |
| `UI_BUTTON_WIDTH`     | int  | `14`         | Width of main-menu buttons in characters.      |
| `UI_BUTTON_FG`        | str  | `lime green` | Primary button text colour.                    |
| `UI_BUTTON_FG_CANCEL` | str  | `red2`       | Cancel / destructive button text colour.       |
| `UI_BUTTON_FG_ACCENT2`| str  | `yellow`     | Secondary accent button text colour.           |
| `UI_FONT_SIZE`        | int  | `16`         | Base font size in points.                      |
| `UI_FONT_FAMILY`      | str  | `Bahnschrift`| Font family for the interface.                 |
| `UI_PADDING`          | int  | `8`          | Padding in pixels between UI elements.         |

Colours accept any Tk colour name (e.g. `lime green`, `crimson`) or a hex
code (`#RRGGBB`).

## UI Tooltips

| Variable              | Type | Default | Description                                    |
|-----------------------|------|---------|------------------------------------------------|
| `UI_TOOLTIP_DELAY_MS` | int  | `500`   | Delay in milliseconds before showing a tooltip. |
| `UI_TOOLTIP_WRAPLENGTH` | int | `350`   | Maximum width in pixels before tooltip text wraps. |
| `UI_TOOLTIP_PADX`    | int  | `8`     | Horizontal padding inside tooltip.             |
| `UI_TOOLTIP_PADY`    | int  | `4`     | Vertical padding inside tooltip.               |

## Plot Style

| Variable              | Type  | Default      | Description                                   |
|-----------------------|-------|--------------|-----------------------------------------------|
| `PLOT_FIGSIZE_WIDTH`  | int   | `12`         | Plot width in inches.                         |
| `PLOT_FIGSIZE_HEIGHT` | int   | `6`          | Plot height in inches.                        |
| `DPI`                 | int   | `100`        | Dots per inch (50--1000).                     |
| `PLOT_SHOW_TITLE`     | bool  | `true`       | Show a title above the plot.                  |
| `PLOT_SHOW_GRID`      | bool  | `true`       | Draw a background grid.                       |
| `PLOT_LINE_COLOR`     | str   | `royalblue`  | Colour of the main solution curve.            |
| `PLOT_LINE_WIDTH`     | float | `1.5`        | Line thickness in points.                     |
| `PLOT_LINE_STYLE`     | str   | `-`          | `-` solid, `--` dashed, `-.` dash-dot, `:` dotted. |
| `PLOT_COLOR_SCHEME`   | str   | `Set1`       | Matplotlib colormap for extra derivatives.    |

## Plot Markers

| Variable                | Type | Default   | Description                      |
|-------------------------|------|-----------|----------------------------------|
| `PLOT_MARKER_FORMAT`    | str  | `o`       | `o` circle, `s` square, `^` triangle, `d` diamond, `*` star. |
| `PLOT_MARKER_SIZE`      | int  | `3`       | Marker size in points.           |
| `PLOT_MARKER_FACE_COLOR`| str  | `crimson` | Marker fill colour.              |
| `PLOT_MARKER_EDGE_COLOR`| str  | `crimson` | Marker edge colour.              |

## Plot Phase-Space

| Variable                 | Type | Default | Description                              |
|---------------------------|------|---------|------------------------------------------|
| `PLOT_PHASE_START_COLOR`  | str  | `green` | Colour of the start marker in phase-space plots. |
| `PLOT_PHASE_END_COLOR`    | str  | `red`   | Colour of the end marker in phase-space plots.   |
| `PLOT_PHASE_MARKER_SIZE`  | int  | `8`     | Size of start/end markers in phase-space plots.  |

## Plot Fonts

Configured directly under Plot Style in the Configuration dialog.

| Variable            | Type | Default     | Description                                  |
|---------------------|------|-------------|----------------------------------------------|
| `FONT_FAMILY`       | str  | `serif`     | Font family for matplotlib plots.  Options: `serif`, `sans-serif`, `monospace`, `cursive`, `fantasy`. |
| `FONT_TITLE_SIZE`   | str  | `xx-large`  | Title font size (matplotlib size string).    |
| `FONT_TITLE_WEIGHT` | str  | `semibold`  | Title font weight.  Options: `normal`, `bold`, `light`, `semibold`, `heavy`. |
| `FONT_AXIS_SIZE`    | int  | `16`        | Axis label font size in points.              |
| `FONT_AXIS_STYLE`   | str  | `italic`    | Axis label font style.  Options: `normal`, `italic`, `oblique`. |
| `FONT_TICK_SIZE`    | int  | `12`        | Tick label font size in points.              |

## Plot 3D / Contour

| Variable               | Type  | Default  | Description                              |
|------------------------|-------|----------|------------------------------------------|
| `PLOT_SURFACE_CMAP`    | str   | `viridis`| Matplotlib colormap for 3D surface and contour plots. |
| `PLOT_CONTOUR_LEVELS`  | int   | `20`     | Number of contour levels in 2D contour plots. |
| `PLOT_GRID_ALPHA`      | float | `0.3`    | Transparency of the grid lines (0–1).    |
| `PLOT_SURFACE_ALPHA`   | float | `0.9`    | Transparency of 3D surfaces (0–1).      |
| `PLOT_COLORBAR_SHRINK` | float | `0.6`    | Shrink factor for the colorbar (0–1).    |

## Plot Animation

| Variable                  | Type  | Default | Description                                    |
|---------------------------|-------|---------|------------------------------------------------|
| `PLOT_ANIMATION_LINE_WIDTH` | float | `2.0`  | Line width for vector animation plot.         |
| `PLOT_VLINES_LINE_WIDTH`  | float | `1.5`   | Line width for vertical lines in animation.   |
| `PLOT_VLINES_ALPHA`       | float | `0.6`   | Transparency of vertical lines (0–1).         |
| `PLOT_ANIMATION_Y_MARGIN` | float | `0.1`   | Margin added to y-axis limits in animation.  |
| `ANIMATION_MAX_FPS`       | int   | `30`    | Maximum frames per second for animation playback. |

## Solver Defaults

The default integration method is the first in the available list (`RK45`). Use the
Parameters dialog to select a different method per run.

| Variable            | Type  | Default  | Description                     |
|---------------------|-------|----------|---------------------------------|
| `SOLVER_MAX_STEP`   | float | `0.0`    | Maximum step size (0 = automatic). |
| `SOLVER_RTOL`           | float | `1e-8`   | Relative tolerance.                          |
| `SOLVER_ATOL`           | float | `1e-10`  | Absolute tolerance.                          |
| `SOLVER_NUM_POINTS`     | int   | `1000`   | Number of evaluation points in the grid.     |

### Solver Methods

| Method   | Description                                              |
|----------|----------------------------------------------------------|
| `RK45`   | Runge-Kutta 4(5) -- general-purpose explicit method.     |
| `RK23`   | Runge-Kutta 2(3) -- low-order, faster per step.          |
| `DOP853` | Runge-Kutta 8(5,3) -- high-order explicit method.        |
| `Radau`  | Implicit Runge-Kutta (Radau IIA) -- stiff problems.      |
| `BDF`    | Backward Differentiation Formula -- stiff problems.      |
| `LSODA`  | Adams/BDF auto-switching -- stiff/non-stiff detection.   |

## Logging & Update

| Variable                   | Type | Default                  | Description                          |
|----------------------------|------|--------------------------|--------------------------------------|
| `LOG_LEVEL`                | str  | `INFO`                   | Verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `LOG_FILE`                 | str  | `differential_lab.log`   | Log file name (project root).        |
| `LOG_CONSOLE`              | bool | `false`                  | Also print logs to the terminal.     |
| `CHECK_UPDATES`            | bool | `true`                   | Check for updates on startup (once per week). |
| `UPDATE_CHECK_INTERVAL_DAYS` | int  | `7`                    | Days between automatic update checks. |
| `CHECK_UPDATES_FORCE`      | bool | `false`                  | Force update check on every startup. |
| `UPDATE_CHECK_URL`         | str  | *(main branch)*          | URL to pyproject.toml for version check. |
