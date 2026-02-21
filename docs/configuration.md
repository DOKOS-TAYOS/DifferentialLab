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

## Plot Fonts

| Variable            | Type | Default     | Description                                  |
|---------------------|------|-------------|----------------------------------------------|
| `FONT_FAMILY`       | str  | `serif`     | Font family for matplotlib plots.  Options: `serif`, `sans-serif`, `monospace`, `cursive`, `fantasy`. |
| `FONT_TITLE_SIZE`   | str  | `xx-large`  | Title font size (matplotlib size string).    |
| `FONT_TITLE_WEIGHT` | str  | `semibold`  | Title font weight.  Options: `normal`, `bold`, `light`, `semibold`, `heavy`. |
| `FONT_AXIS_SIZE`    | int  | `16`        | Axis label font size in points.              |
| `FONT_AXIS_STYLE`   | str  | `italic`    | Axis label font style.  Options: `normal`, `italic`, `oblique`. |
| `FONT_TICK_SIZE`    | int  | `12`        | Tick label font size in points.              |

## Solver Defaults

| Variable                | Type  | Default  | Description                                 |
|-------------------------|-------|----------|---------------------------------------------|
| `SOLVER_DEFAULT_METHOD` | str   | `RK45`   | Default integration method.  Options: `RK45`, `RK23`, `DOP853`, `Radau`, `BDF`, `LSODA`. |
| `SOLVER_MAX_STEP`       | float | `0.0`    | Maximum step size (0 = automatic).           |
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

## File Paths

| Variable          | Type | Default  | Description                              |
|-------------------|------|----------|------------------------------------------|
| `FILE_OUTPUT_DIR` | str  | `output` | Directory for CSV, JSON, and plot files. |
| `FILE_PLOT_FORMAT`| str  | `png`    | Image format: `png`, `jpg`, or `pdf`.    |

## Logging

| Variable      | Type | Default                  | Description                          |
|---------------|------|--------------------------|--------------------------------------|
| `LOG_LEVEL`   | str  | `INFO`                   | Verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `LOG_FILE`    | str  | `differential_lab.log`   | Log file name (project root).        |
| `LOG_CONSOLE` | bool | `false`                  | Also print logs to the terminal.     |

## Update Check

| Variable           | Type | Default | Description                                      |
|--------------------|------|---------|--------------------------------------------------|
| `CHECK_UPDATES`    | bool | `true`  | Check for updates on startup (once per week).   |
| `CHECK_UPDATES_FORCE` | bool | `false` | Force update check on every startup.         |
| `UPDATE_CHECK_URL` | str  | *(main branch)* | URL to pyproject.toml for version check. |
