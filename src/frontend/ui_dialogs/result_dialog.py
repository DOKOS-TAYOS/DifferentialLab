"""Result dialog — display plot, statistics, and export paths."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

from matplotlib.figure import Figure

from config.env import get_env_from_schema
from config.theme import get_font
from frontend.window_utils import center_window, make_modal
from plotting.plot_utils import embed_plot_in_tk
from utils.logger import get_logger

logger = get_logger(__name__)


class ResultDialog:
    """Window showing the ODE solution, statistics, and file paths.

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
        center_window(self.win, 960, 750)
        make_modal(self.win, parent)

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self._build_ui(fig, phase_fig, statistics, metadata, csv_path, json_path, plot_path)
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

        # Notebook for plots
        notebook = ttk.Notebook(self.win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=pad, pady=pad)

        # Solution plot tab
        plot_frame = ttk.Frame(notebook)
        notebook.add(plot_frame, text="  Solution y(x)  ")
        embed_plot_in_tk(fig, plot_frame)

        # Phase portrait tab (if 2nd order+)
        if phase_fig is not None:
            phase_frame = ttk.Frame(notebook)
            notebook.add(phase_frame, text="  Phase Portrait  ")
            embed_plot_in_tk(phase_fig, phase_frame)

        # Bottom pane: stats + info
        bottom = ttk.Frame(self.win, padding=pad)
        bottom.pack(fill=tk.BOTH, padx=pad, pady=(0, pad))

        # Statistics table
        stats_lf = ttk.LabelFrame(bottom, text="Statistics & Magnitudes", padding=pad)
        stats_lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(stats_lf, columns=("value",), show="headings", height=8)
        tree.heading("value", text="Value")
        tree.column("value", width=300)

        for key, val in statistics.items():
            display_val = self._format_stat(val)
            tree.insert("", tk.END, values=(f"{key}: {display_val}",))

        tree_scroll = ttk.Scrollbar(stats_lf, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=tree_scroll.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Solver info
        info_lf = ttk.LabelFrame(bottom, text="Solver Info", padding=pad)
        info_lf.pack(side=tk.LEFT, fill=tk.BOTH, padx=(pad, 0))

        info_items = [
            ("Method", metadata.get("method", "?")),
            ("Success", "Yes" if metadata.get("solver_success") else "No"),
            ("Evaluations", metadata.get("n_evaluations", "?")),
            ("Points", metadata.get("num_points", "?")),
        ]
        for label, value in info_items:
            row = ttk.Frame(info_lf)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{label}:", width=14, anchor=tk.W).pack(side=tk.LEFT)
            ttk.Label(row, text=str(value), style="Small.TLabel").pack(side=tk.LEFT)

        # File paths
        files_lf = ttk.LabelFrame(self.win, text="Output Files", padding=pad)
        files_lf.pack(fill=tk.X, padx=pad, pady=(0, pad))

        for label, path in [("CSV", csv_path), ("JSON", json_path), ("Plot", plot_path)]:
            row = ttk.Frame(files_lf)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{label}:", width=6, anchor=tk.W).pack(side=tk.LEFT)
            ttk.Label(row, text=str(path), style="Small.TLabel").pack(
                side=tk.LEFT, fill=tk.X, expand=True,
            )

        # Close button
        ttk.Button(self.win, text="Close", style="Cancel.TButton",
                   command=self.win.destroy).pack(pady=(0, pad))

    @staticmethod
    def _format_stat(value: Any) -> str:
        """Format a statistic value for display.

        Args:
            value: The value (may be a dict, float, int, or None).

        Returns:
            Formatted string.
        """
        if value is None:
            return "N/A"
        if isinstance(value, dict):
            parts = [f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in value.items()]
            return ", ".join(parts)
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)
