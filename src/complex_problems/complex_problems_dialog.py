"""Dialog for selecting which complex problem to solve."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from complex_problems.problem_registry import PROBLEM_REGISTRY, open_problem_dialog
from config import get_env_from_schema
from frontend.ui_dialogs import ToolTip, setup_arrow_enter_navigation
from frontend.window_utils import bind_wraplength, fit_and_center, make_modal
from utils import get_logger

logger = get_logger(__name__)


class ComplexProblemsDialog:
    """Window for selecting a complex problem to solve.

    Args:
        parent: Parent window.
    """

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Complex Problems")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self._build_ui()

        fit_and_center(self.win, min_width=640, min_height=680, resizable=True)
        make_modal(self.win, parent)
        logger.info("Complex problems dialog opened")

    def _build_ui(self) -> None:
        """Construct the dialog layout."""
        pad: int = get_env_from_schema("UI_PADDING")

        main_frame = ttk.Frame(self.win, padding=pad * 2)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main_frame,
            text="Complex Problems",
            style="Title.TLabel",
        ).pack(pady=(0, pad))

        desc = ttk.Label(
            main_frame,
            text=(
                "Select a special problem to solve. Each problem has\n"
                "custom parameters, statistics, and visualizations."
            ),
            style="Small.TLabel",
            justify=tk.CENTER,
        )
        desc.pack(pady=(0, pad * 2))
        bind_wraplength(main_frame, desc, pad=4 * pad, min_wrap=200)

        # Buttons for each problem
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.BOTH, expand=True)

        for i, (prob_id, descriptor) in enumerate(PROBLEM_REGISTRY.items()):
            btn = ttk.Button(
                btn_frame,
                text=descriptor.name,
                command=self._make_open_callback(prob_id),
            )
            btn.pack(fill=tk.X, padx=pad, pady=pad // 2)
            ToolTip(btn, descriptor.description)

        btn_close = ttk.Button(
            main_frame,
            text="Close",
            style="Cancel.TButton",
            command=self.win.destroy,
        )
        btn_close.pack(pady=(pad * 2, 0))

        setup_arrow_enter_navigation([[btn_close]])
        btn_close.focus_set()

    def _make_open_callback(self, problem_id: str):
        """Create a callback that opens the problem dialog and closes this one."""

        def _on_click() -> None:
            self.win.destroy()
            open_problem_dialog(problem_id, self.parent)

        return _on_click
