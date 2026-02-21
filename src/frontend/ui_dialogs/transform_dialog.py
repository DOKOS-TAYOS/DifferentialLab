"""Transform dialog — enter function, apply Fourier/Laplace/Taylor, visualize and export."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Callable

import numpy as np

from config import (
    generate_output_basename,
    get_csv_path,
    get_env_from_schema,
)
from frontend.plot_embed import embed_plot_in_tk
from frontend.theme import get_font
from frontend.ui_dialogs.collapsible_section import CollapsibleSection
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import center_window, make_modal
from transforms import (
    DisplayMode,
    TransformKind,
    apply_transform,
    get_transform_coefficients,
    parse_scalar_function,
)
from utils import EquationParseError, get_logger

if TYPE_CHECKING:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

logger = get_logger(__name__)

_LEFT_WIDTH = 320  # Width of controls panel (similar to ResultDialog layout)


class TransformDialog:
    """Dialog for entering a function, applying transforms, and exporting data.

    Args:
        parent: Parent window.
    """

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Function Transforms")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self._func: object = None  # Callable | None
        self._current_x: np.ndarray | None = None
        self._current_y: np.ndarray | None = None
        self._current_xlabel: str = "x"
        self._current_ylabel: str = "y"
        self._y_original: np.ndarray | None = None  # For Taylor overlay
        self._show_coefficients: bool = False  # Display mode: curve vs coefficients
        self._canvas: FigureCanvasTkAgg | None = None
        self._fig: Figure | None = None
        self._ax = None

        self._build_ui()

        # Size window from plot dimensions in .env (like ResultDialog)
        pad: int = get_env_from_schema("UI_PADDING")
        screen_w = self.win.winfo_screenwidth()
        screen_h = self.win.winfo_screenheight()
        fig_w: int = get_env_from_schema("PLOT_FIGSIZE_WIDTH")
        fig_h: int = get_env_from_schema("PLOT_FIGSIZE_HEIGHT")
        aspect: float = fig_w / fig_h if fig_h else 2.0
        win_w = int(screen_w * 0.88)
        right_w = win_w - _LEFT_WIDTH - 3 * pad
        plot_h = int(right_w / aspect)
        chrome_h = 30 + 46 + 2 * pad  # toolbar + button bar + padding
        win_h = min(max(plot_h + chrome_h, 500), int(screen_h * 0.92))

        center_window(self.win, win_w, win_h, max_width_ratio=0.92, resizable=True)
        self.win.minsize(_LEFT_WIDTH + 500, 500)
        make_modal(self.win, parent)

    def _build_ui(self) -> None:
        """Build the dialog layout."""
        pad: int = get_env_from_schema("UI_PADDING")
        _font = get_font()
        btn_bg: str = get_env_from_schema("UI_BUTTON_BG")
        fg: str = get_env_from_schema("UI_FOREGROUND")

        # ── Fixed bottom button bar ──
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=pad, pady=pad)

        btn_help = ttk.Button(
            btn_frame, text="Help",
            command=self._on_help,
        )
        btn_help.pack(side=tk.LEFT, padx=pad)
        ToolTip(btn_help, "Show help about the Transforms section.")

        btn_export = ttk.Button(
            btn_frame, text="Export CSV",
            command=self._on_export,
        )
        btn_export.pack(side=tk.LEFT, padx=pad)
        ToolTip(btn_export, "Export the transformed data points to a CSV file.")

        btn_close = ttk.Button(
            btn_frame, text="Close", style="Cancel.TButton",
            command=self.win.destroy,
        )
        btn_close.pack(side=tk.LEFT, padx=pad)

        setup_arrow_enter_navigation([[btn_help, btn_export, btn_close]])
        ttk.Separator(self.win, orient=tk.HORIZONTAL).pack(
            side=tk.BOTTOM, fill=tk.X,
        )

        # ── Main content ──
        content = ttk.Frame(self.win)
        content.pack(fill=tk.BOTH, expand=True, padx=pad, pady=pad)

        # ── Left: controls ──
        left = ttk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, pad))

        # Function
        func_lf = ttk.LabelFrame(left, text="Function f(x)", padding=pad)
        func_lf.pack(fill=tk.X, pady=(0, pad))

        ttk.Label(
            func_lf,
            text="Use x as variable. Example: sin(x), exp(-a*x)",
            style="Small.TLabel",
        ).pack(anchor=tk.W)
        self._func_entry = tk.Text(
            func_lf, height=2, width=30,
            bg=btn_bg, fg=fg, insertbackground=fg, font=_font,
        )
        self._func_entry.insert("1.0", "sin(x)")
        self._func_entry.pack(fill=tk.X, pady=(4, pad))

        ttk.Label(func_lf, text="Parameters (name=value, comma-separated):").pack(
            anchor=tk.W
        )
        self._params_entry = ttk.Entry(func_lf, width=32, font=_font)
        self._params_entry.pack(fill=tk.X, pady=(4, pad))
        ToolTip(self._params_entry, "E.g.: a=1.0, omega=2.0")

        # Range
        range_lf = ttk.LabelFrame(left, text="Range", padding=pad)
        range_lf.pack(fill=tk.X, pady=(0, pad))

        row1 = ttk.Frame(range_lf)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="x_min:").pack(side=tk.LEFT)
        self._x_min_var = tk.StringVar(value="-10")
        ttk.Entry(row1, textvariable=self._x_min_var, width=10, font=_font).pack(
            side=tk.LEFT, padx=(4, pad)
        )
        ttk.Label(row1, text="x_max:").pack(side=tk.LEFT, padx=(pad, 0))
        self._x_max_var = tk.StringVar(value="10")
        ttk.Entry(row1, textvariable=self._x_max_var, width=10, font=_font).pack(
            side=tk.LEFT, padx=(4, pad)
        )

        # Transform
        trans_lf = ttk.LabelFrame(left, text="Transformation", padding=pad)
        trans_lf.pack(fill=tk.X, pady=(0, pad))

        self._transform_var = tk.StringVar(value=TransformKind.ORIGINAL.value)
        self._transform_combo = ttk.Combobox(
            trans_lf,
            textvariable=self._transform_var,
            values=[k.value for k in TransformKind],
            state="readonly",
            width=28,
            font=_font,
        )
        self._transform_combo.pack(fill=tk.X, pady=(0, pad))
        self._transform_combo.bind("<<ComboboxSelected>>", self._on_transform_change)

        # Display mode (curve vs coefficients)
        ttk.Label(trans_lf, text="Display:", style="Small.TLabel").pack(anchor=tk.W)
        self._display_var = tk.StringVar(value=DisplayMode.CURVE.value)
        display_combo = ttk.Combobox(
            trans_lf,
            textvariable=self._display_var,
            values=[k.value for k in DisplayMode],
            state="readonly",
            width=26,
            font=_font,
        )
        display_combo.pack(fill=tk.X, pady=(2, pad))
        display_combo.bind("<<ComboboxSelected>>", self._on_display_change)

        # Taylor options (shown only when Taylor is selected)
        self._taylor_frame = ttk.Frame(trans_lf)
        self._taylor_frame.pack(fill=tk.X, pady=(0, pad))
        ttk.Label(self._taylor_frame, text="Order:").pack(side=tk.LEFT)
        self._taylor_order_var = tk.StringVar(value="5")
        ttk.Spinbox(
            self._taylor_frame,
            from_=1, to=15, width=5,
            textvariable=self._taylor_order_var,
            font=_font,
        ).pack(side=tk.LEFT, padx=(4, pad))
        ttk.Label(self._taylor_frame, text="Center:").pack(side=tk.LEFT, padx=(pad, 0))
        self._taylor_center_var = tk.StringVar(value="0")
        ttk.Entry(
            self._taylor_frame,
            textvariable=self._taylor_center_var,
            width=8,
            font=_font,
        ).pack(side=tk.LEFT, padx=(4, 0))

        # Apply button
        self._btn_apply = ttk.Button(
            left, text="Apply / Update plot",
            command=self._on_apply,
        )
        self._btn_apply.pack(fill=tk.X, pady=pad)
        ToolTip(self._btn_apply, "Parse function and apply selected transformation.")

        # ── Right: plot ──
        plot_frame = ttk.Frame(content)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._plot_container = ttk.Frame(plot_frame)
        self._plot_container.pack(fill=tk.BOTH, expand=True)

        self._on_apply()

    def _on_transform_change(self, _event: object) -> None:
        """When transform selection changes, refresh the plot."""
        self._on_apply()

    def _on_display_change(self, _event: object) -> None:
        """When display mode changes, refresh the plot."""
        self._on_apply()

    def _on_help(self) -> None:
        """Show help window for the Transforms section."""
        _TransformHelpDialog(self.win, on_close=lambda: self.win.grab_set())

    def _parse_inputs(self) -> tuple[object, float, float, dict[str, float]]:
        """Parse function expression, range, and parameters.

        Returns:
            (func, x_min, x_max, params).

        Raises:
            ValueError: On parse error.
        """
        from solver.equation_parser import normalize_unicode_escapes

        expr = normalize_unicode_escapes(
            self._func_entry.get("1.0", tk.END).strip()
        )
        if not expr:
            raise ValueError("Please enter a function expression.")

        params: dict[str, float] = {}
        raw = self._params_entry.get().strip()
        if raw:
            for pair in raw.split(","):
                pair = pair.strip()
                if "=" in pair:
                    name, val_str = pair.split("=", 1)
                    name = normalize_unicode_escapes(name.strip())
                    try:
                        params[name] = float(val_str.strip())
                    except ValueError:
                        raise ValueError(f"Parameter '{name}' must be a number.")

        try:
            x_min = float(self._x_min_var.get())
            x_max = float(self._x_max_var.get())
        except ValueError:
            raise ValueError("x_min and x_max must be numbers.")
        if x_min >= x_max:
            raise ValueError("x_min must be less than x_max.")

        func = parse_scalar_function(expr, params)
        return func, x_min, x_max, params

    def _on_apply(self) -> None:
        """Parse inputs, apply transform, and update the plot."""
        try:
            func, x_min, x_max, _params = self._parse_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc), parent=self.win)
            return
        except EquationParseError as exc:
            messagebox.showerror("Parse Error", str(exc), parent=self.win)
            return

        kind_str = self._transform_var.get()
        try:
            kind = TransformKind(kind_str)
        except ValueError:
            kind = TransformKind.ORIGINAL

        taylor_order = 5
        taylor_center: float | None = None
        if kind == TransformKind.TAYLOR:
            try:
                taylor_order = int(self._taylor_order_var.get())
                taylor_order = max(1, min(15, taylor_order))
            except ValueError:
                pass
            try:
                taylor_center = float(self._taylor_center_var.get())
            except ValueError:
                taylor_center = (x_min + x_max) / 2

        n_points = 512
        if kind == TransformKind.LAPLACE:
            n_points = 200

        display_str = self._display_var.get()
        try:
            display_mode = DisplayMode(display_str)
        except ValueError:
            display_mode = DisplayMode.CURVE
        self._show_coefficients = display_mode == DisplayMode.COEFFICIENTS

        try:
            if self._show_coefficients:
                x, y, xlabel, ylabel = get_transform_coefficients(
                    func,
                    kind,
                    x_min,
                    x_max,
                    n_points=n_points,
                    taylor_order=taylor_order,
                    taylor_center=taylor_center,
                )
            else:
                x, y, xlabel, ylabel = apply_transform(
                    func,
                    kind,
                    x_min,
                    x_max,
                    n_points=n_points,
                    taylor_order=taylor_order,
                    taylor_center=taylor_center,
                )
        except Exception as exc:
            logger.exception("Transform failed")
            messagebox.showerror(
                "Transform Error",
                f"Could not compute transform: {exc}",
                parent=self.win,
            )
            return

        self._func = func
        self._current_x = x
        self._current_y = y
        self._current_xlabel = xlabel
        self._current_ylabel = ylabel

        if kind == TransformKind.TAYLOR and not self._show_coefficients:
            from transforms import compute_function_samples
            x_orig, y_orig = compute_function_samples(func, x_min, x_max, n_points)
            self._y_original = y_orig
        else:
            self._y_original = None

        self._redraw_plot()

    def _redraw_plot(self) -> None:
        """Redraw the plot with current data."""
        import matplotlib.pyplot as plt

        from config import get_env_from_schema
        from plotting.plot_utils import _apply_plot_style, _finalize_plot

        if self._current_x is None or self._current_y is None:
            return

        x = self._current_x
        y = self._current_y
        xlabel = self._current_xlabel
        ylabel = self._current_ylabel

        if self._fig is None:
            _apply_plot_style()
            width: int = get_env_from_schema("PLOT_FIGSIZE_WIDTH")
            height: int = get_env_from_schema("PLOT_FIGSIZE_HEIGHT")
            dpi: int = get_env_from_schema("DPI")
            self._fig, self._ax = plt.subplots(
                figsize=(width, height), dpi=dpi
            )
            self._canvas = embed_plot_in_tk(self._fig, self._plot_container)
        else:
            self._ax = self._fig.axes[0] if self._fig.axes else None
            if self._ax is None:
                self._ax = self._fig.add_subplot(111)

        self._ax.clear()
        line_color: str = get_env_from_schema("PLOT_LINE_COLOR")
        line_width: float = get_env_from_schema("PLOT_LINE_WIDTH")

        if self._show_coefficients:
            markerline, stemlines, baseline = self._ax.stem(
                x, y,
                linefmt="-",
                markerfmt="o",
                basefmt=" ",
                label=ylabel,
            )
            plt.setp(markerline, color=line_color)
            plt.setp(stemlines, color=line_color)
        else:
            self._ax.plot(x, y, color=line_color, linewidth=line_width, label=ylabel)
            if self._y_original is not None:
                self._ax.plot(
                    x, self._y_original,
                    color="coral", linewidth=line_width * 0.8,
                    linestyle="--", label="f(x)",
                )
        if self._y_original is not None:
            self._ax.legend()

        _finalize_plot(
            self._ax,
            title="",
            xlabel=xlabel,
            ylabel=ylabel,
            legend=(self._y_original is not None),
        )
        self._fig.tight_layout()
        if self._canvas:
            self._canvas.draw_idle()

    def _on_export(self) -> None:
        """Export the transformed data to CSV."""
        if self._current_x is None or self._current_y is None:
            messagebox.showwarning(
                "No Data",
                "Apply a transformation first to generate data.",
                parent=self.win,
            )
            return

        default_path = get_csv_path(
            generate_output_basename(prefix="transform")
        )
        filepath = filedialog.asksaveasfilename(
            parent=self.win,
            defaultextension=".csv",
            initialfile=default_path.name,
            initialdir=str(default_path.parent),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not filepath:
            return

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        headers = [self._current_xlabel, self._current_ylabel]
        data = np.column_stack([self._current_x, self._current_y])
        with open(path, "w", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data.tolist())

        logger.info("Transform data exported: %s", path)
        messagebox.showinfo(
            "Export Complete",
            f"Data exported to:\n{path}",
            parent=self.win,
        )


_TRANSFORM_HELP_ABOUT = (
    "This section lets you enter a scalar function f(x), apply mathematical "
    "transforms (Fourier, Laplace, Taylor, Hilbert, Z-transform), and visualize "
    "the result. You can view either the curve or the coefficients a_i vs index i."
)

_TRANSFORM_HELP_INPUT = (
    "Use x as the independent variable. Example: sin(x), exp(-a*x)\n\n"
    "Parameters: name=value, comma-separated (e.g. a=1.0, omega=2).\n\n"
    "Range: x_min and x_max define the domain for sampling."
)

_TRANSFORM_HELP_TRANSFORMS = (
    "Original (f(x)) — Plot the function as-is.\n"
    "Fourier (FFT) — Discrete Fourier transform magnitude spectrum.\n"
    "Laplace (real axis) — Laplace transform L(s) = ∫f(t)e^{-st}dt over the range.\n"
    "Taylor series — Polynomial expansion around a center point (order, center).\n"
    "Hilbert (discrete) — Discrete Hilbert transform H[f](x).\n"
    "Z-transform (discrete) — Magnitude spectrum (DFT on unit circle)."
)

_TRANSFORM_HELP_DISPLAY = (
    "Curve (f vs x) — Plot the transformed function or spectrum vs its domain.\n\n"
    "Coefficients (a_i vs i) — Plot coefficients vs index:\n"
    "    Taylor: a_i = f^(i)(x0)/i!\n"
    "    Fourier: |F[k]| vs k\n"
    "    Laplace: L(s_i) vs i\n"
    "    Hilbert: |H[k]| vs k\n"
    "    Z-transform: x[n] vs n"
)

_TRANSFORM_HELP_EXPORT = (
    "Export the currently displayed data (curve or coefficients) to CSV. "
    "Use the Export CSV button to save the points to a file."
)

_TRANSFORM_HELP_SECTIONS: list[tuple[str, str]] = [
    ("About", _TRANSFORM_HELP_ABOUT),
    ("Function Input", _TRANSFORM_HELP_INPUT),
    ("Transformations", _TRANSFORM_HELP_TRANSFORMS),
    ("Display Mode", _TRANSFORM_HELP_DISPLAY),
    ("Export", _TRANSFORM_HELP_EXPORT),
]


class _TransformHelpDialog:
    """Help window for the Transforms section, with collapsible sections."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.win = tk.Toplevel(parent)
        self.win.title("Transforms — Help")
        self._on_close = on_close

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        # Take grab so collapsibles receive clicks (parent had grab from make_modal)
        parent.grab_release()
        self.win.grab_set()

        self._body_labels: list[ttk.Label] = []

        def _do_close() -> None:
            if self._on_close:
                self._on_close()
            self.win.destroy()

        self._do_close = _do_close
        self._build_ui()

        from frontend.window_utils import fit_and_center

        fit_and_center(self.win, min_width=780, min_height=520)
        self.win.transient(parent)
        self.win.protocol("WM_DELETE_WINDOW", self._do_close)

    def _build_ui(self) -> None:
        pad: int = get_env_from_schema("UI_PADDING")
        bg: str = get_env_from_schema("UI_BACKGROUND")

        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=pad, pady=pad)

        btn_close = ttk.Button(
            btn_frame, text="Close", style="Cancel.TButton",
            command=self._do_close,
        )
        btn_close.pack()

        ttk.Separator(self.win, orient=tk.HORIZONTAL).pack(
            side=tk.BOTTOM, fill=tk.X,
        )

        setup_arrow_enter_navigation([[btn_close]])
        btn_close.focus_set()

        self._scroll = ScrollableFrame(self.win)
        self._scroll.apply_bg(bg)
        self._scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        inner = self._scroll.inner
        inner.configure(padding=pad)

        ttk.Label(
            inner,
            text="Transforms — Help",
            style="Title.TLabel",
        ).pack(anchor=tk.W, pady=(0, pad))

        first = True
        for title, body in _TRANSFORM_HELP_SECTIONS:
            self._add_section(inner, title, body, expanded=first)
            first = False

        self._scroll.bind_new_children()

        def _update_wraplength(_e: tk.Event | None = None) -> None:  # type: ignore[type-arg]
            w = inner.winfo_width()
            if w > 100:
                wrap = max(400, w - 48)
                for lbl in self._body_labels:
                    lbl.configure(wraplength=wrap)

        inner.bind("<Configure>", _update_wraplength)

    def _add_section(
        self,
        parent: ttk.Frame,
        title: str,
        body: str,
        *,
        expanded: bool = False,
    ) -> None:
        pad: int = get_env_from_schema("UI_PADDING")
        section = CollapsibleSection(
            parent, self._scroll, title, expanded=expanded, pad=pad,
        )
        body_lbl = ttk.Label(
            section.content, text=body, justify=tk.LEFT, wraplength=720,
        )
        body_lbl.pack(anchor=tk.W, fill=tk.X)
        self._body_labels.append(body_lbl)
