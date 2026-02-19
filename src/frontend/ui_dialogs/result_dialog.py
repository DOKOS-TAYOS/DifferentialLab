"""Result dialog — left panel (magnitudes, stats, info) + right panel (plots)."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

from matplotlib.figure import Figure

from config.env import get_env_from_schema
from config.theme import get_font
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.window_utils import center_window, make_modal
from plotting.plot_utils import embed_plot_in_tk
from utils.logger import get_logger

logger = get_logger(__name__)

_MAGNITUDE_KEYS = {"mean", "rms", "std", "integral"}

_LEFT_MIN_WIDTH = 480


class ResultDialog:
    """Window showing the ODE solution, statistics, and file paths.

    The layout places text information on the left and plots on the right
    so that both are visible at the same time.

    Args:
        parent: Parent window.
        fig: Matplotlib figure with the y(x) plot.
        phase_fig: Optional phase portrait figure.
        statistics: Computed statistics dict.
        metadata: Solver metadata dict.
        csv_path: Path to the exported CSV.
        json_path: Path to the exported JSON.
        plot_path: Path to the exported plot image.
    """

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        fig: Figure,
        phase_fig: Figure | None,
        statistics: dict[str, Any],
        metadata: dict[str, Any],
        csv_path: Path,
        json_path: Path,
        plot_path: Path,
    ) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title(f"Results — {metadata.get('equation_name', 'ODE')}")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self._build_ui(fig, phase_fig, statistics, metadata,
                       csv_path, json_path, plot_path)

        self.win.update_idletasks()
        screen_w = self.win.winfo_screenwidth()
        screen_h = self.win.winfo_screenheight()
        win_w = min(int(screen_w * 0.92), max(1400, _LEFT_MIN_WIDTH + 700))
        win_h = min(int(screen_h * 0.88), 900)
        center_window(self.win, win_w, win_h)
        make_modal(self.win, parent)
        logger.info("Result dialog displayed")

    def _build_ui(
        self,
        fig: Figure,
        phase_fig: Figure | None,
        statistics: dict[str, Any],
        metadata: dict[str, Any],
        csv_path: Path,
        json_path: Path,
        plot_path: Path,
    ) -> None:
        pad: int = get_env_from_schema("UI_PADDING")

        # ── Fixed bottom button bar ──
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=pad, pady=pad)

        btn_close = ttk.Button(
            btn_frame, text="Close", style="Cancel.TButton",
            command=self.win.destroy,
        )
        btn_close.pack()

        setup_arrow_enter_navigation([[btn_close]])
        btn_close.focus_set()

        ttk.Separator(self.win, orient=tk.HORIZONTAL).pack(
            side=tk.BOTTOM, fill=tk.X,
        )

        # ── Main content area (grid: left info | right plot) ──
        content = ttk.Frame(self.win)
        content.pack(fill=tk.BOTH, expand=True, padx=pad, pady=pad)

        content.columnconfigure(0, weight=0, minsize=_LEFT_MIN_WIDTH)
        content.columnconfigure(1, weight=1, minsize=400)
        content.rowconfigure(0, weight=1)

        # ── LEFT: scrollable info panel ──
        left_frame = ttk.Frame(content, width=_LEFT_MIN_WIDTH)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, pad))
        left_frame.grid_propagate(False)

        left_scroll = ScrollableFrame(left_frame)
        left_scroll.apply_bg(get_env_from_schema("UI_BACKGROUND"))
        left_scroll.pack(fill=tk.BOTH, expand=True)
        left_inner = left_scroll.inner
        left_inner.configure(padding=pad)

        magnitudes = {k: v for k, v in statistics.items() if k in _MAGNITUDE_KEYS}
        other_stats = {k: v for k, v in statistics.items() if k not in _MAGNITUDE_KEYS}

        if magnitudes:
            mag_lf = ttk.LabelFrame(left_inner, text="Magnitudes", padding=pad)
            mag_lf.pack(fill=tk.X, pady=(0, pad))
            for key, val in magnitudes.items():
                row = ttk.Frame(mag_lf)
                row.pack(fill=tk.X, pady=1)
                ttk.Label(row, text=f"{key}:", width=16, anchor=tk.W).pack(side=tk.LEFT)
                ttk.Label(row, text=self._format_stat(val),
                          style="Small.TLabel").pack(side=tk.LEFT)

        if other_stats:
            stat_lf = ttk.LabelFrame(left_inner, text="Statistics", padding=pad)
            stat_lf.pack(fill=tk.X, pady=(0, pad))
            for key, val in other_stats.items():
                row = ttk.Frame(stat_lf)
                row.pack(fill=tk.X, pady=1)
                ttk.Label(row, text=f"{key}:", width=16, anchor=tk.W).pack(side=tk.LEFT)
                ttk.Label(row, text=self._format_stat(val),
                          style="Small.TLabel").pack(side=tk.LEFT)

        # Solver info
        info_lf = ttk.LabelFrame(left_inner, text="Solver Info", padding=pad)
        info_lf.pack(fill=tk.X, pady=(0, pad))

        info_items = [
            ("Method", metadata.get("method", "?")),
            ("Success", "Yes" if metadata.get("solver_success") else "No"),
            ("Evaluations", metadata.get("n_evaluations", "?")),
            ("Points", metadata.get("num_points", "?")),
        ]
        for label, value in info_items:
            row = ttk.Frame(info_lf)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{label}:", width=16, anchor=tk.W).pack(side=tk.LEFT)
            ttk.Label(row, text=str(value), style="Small.TLabel").pack(side=tk.LEFT)

        # File paths
        files_lf = ttk.LabelFrame(left_inner, text="Output Files", padding=pad)
        files_lf.pack(fill=tk.X, pady=(0, pad))

        for label, path in [("CSV", csv_path), ("JSON", json_path), ("Plot", plot_path)]:
            row = ttk.Frame(files_lf)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{label}:", width=6, anchor=tk.W).pack(side=tk.LEFT)
            ttk.Label(row, text=str(path), style="Small.TLabel",
                      wraplength=_LEFT_MIN_WIDTH - 80).pack(
                side=tk.LEFT, fill=tk.X, expand=True,
            )

        left_scroll.bind_new_children()

        # ── RIGHT: plots ──
        right_frame = ttk.Frame(content)
        right_frame.grid(row=0, column=1, sticky="nsew")

        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        plot_tab = ttk.Frame(notebook)
        notebook.add(plot_tab, text="  Solution y(x)  ")
        embed_plot_in_tk(fig, plot_tab)

        if phase_fig is not None:
            phase_tab = ttk.Frame(notebook)
            notebook.add(phase_tab, text="  Phase Portrait  ")
            embed_plot_in_tk(phase_fig, phase_tab)

    @staticmethod
    def _format_stat(value: Any) -> str:
        """Format a statistic value for display."""
        if value is None:
            return "N/A"
        if isinstance(value, dict):
            parts = [f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in value.items()]
            return ", ".join(parts)
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)
