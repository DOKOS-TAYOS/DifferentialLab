"""Help / Information dialog with collapsible sections."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from config import (
    APP_NAME,
    APP_VERSION,
    AVAILABLE_STATISTICS,
    SOLVER_METHODS,
    SOLVER_METHOD_DESCRIPTIONS,
    get_env_from_schema,
)
from frontend.ui_dialogs.collapsible_section import CollapsibleSection
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.window_utils import center_window, make_modal
from utils import get_logger

logger = get_logger(__name__)

# ── Section content (human-readable) ─────────────────────────────────

_ABOUT = (
    f"Welcome to {APP_NAME} v{APP_VERSION}!\n\n"
    f"{APP_NAME} is a graphical tool for solving ordinary differential "
    "equations (ODEs) numerically. Whether you are studying physics, "
    "engineering or mathematics, it lets you explore how systems evolve "
    "over time — from simple exponential growth to complex oscillators.\n\n"
    "Under the hood it uses SciPy's robust integration engine (solve_ivp) "
    "and supports equations of any order."
)

_HOW_TO_USE = (
    "1.  Click  Solve  in the main menu.\n"
    "2.  Pick a predefined equation (and tweak its parameters) or switch to "
    "the Custom tab and write your own expression.\n"
    "3.  Set the domain [x_min, x_max], initial conditions, number of "
    "evaluation points, solver method, and which statistics to compute.\n"
    "4.  Press  Solve  to run the computation.\n"
    "5.  A results window will appear with the solution plot, phase portrait "
    "(for 2nd-order+ equations), statistics, and links to the exported files."
)

_CUSTOM_EXPRESSIONS = (
    "Use Python / NumPy syntax. The independent variable is  x .\n"
    "The state vector is  y :\n"
    "    y[0] = y        (the function itself)\n"
    "    y[1] = y'       (first derivative)\n"
    "    y[2] = y''      (second derivative)  ...\n\n"
    "Available math functions:\n"
    "    sin, cos, tan, exp, log, log10, sqrt, abs,\n"
    "    sinh, cosh, tanh, arcsin, arccos, arctan,\n"
    "    floor, ceil, sign, heaviside, pi, e\n\n"
    "Example — damped oscillator (order 2):\n"
    "    Expression:   -2*gamma*y[1] - omega**2*y[0]\n"
    "    Parameters:   omega=1.0, gamma=0.1"
)

_PREDEFINED_EQUATIONS = (
    "\u2022 Simple Harmonic Oscillator — classic undamped oscillation\n"
    "\u2022 Damped Oscillator — oscillation with energy loss\n"
    "\u2022 Exponential Growth / Decay — constant-rate change\n"
    "\u2022 Logistic Equation — growth with carrying capacity\n"
    "\u2022 Van der Pol Oscillator — nonlinear self-sustained oscillation\n"
    "\u2022 Simple Pendulum — large-angle pendulum motion\n"
    "\u2022 RC Circuit (Discharge) — capacitor voltage over time\n"
    "\u2022 Free Fall with Drag — motion against air resistance"
)

_OUTPUT_FILES = (
    "Every time you solve an equation three files are created inside the "
    "output/ folder (configurable):\n\n"
    "\u2022 CSV  — tabular x, y data columns ready for spreadsheets.\n"
    "\u2022 JSON — full metadata and all computed statistics.\n"
    "\u2022 Plot — image of the solution curve (PNG, JPG or PDF per your "
    "configuration)."
)

_CONFIGURATION = (
    "You can customise almost every visual and numerical aspect of "
    f"{APP_NAME}. Open  Configuration  from the main menu or edit the "
    ".env  file directly.\n\n"
    "Changes are saved to the  .env  file and the application restarts "
    "automatically so they take effect immediately."
)


def _solver_methods_text() -> str:
    lines: list[str] = []
    for m in SOLVER_METHODS:
        lines.append(f"\u2022 {m}  —  {SOLVER_METHOD_DESCRIPTIONS[m]}")
    return "\n".join(lines)


def _statistics_text() -> str:
    lines: list[str] = []
    for key, desc in AVAILABLE_STATISTICS.items():
        lines.append(f"\u2022 {key}  —  {desc}")
    return "\n".join(lines)


_SECTIONS: list[tuple[str, str]] = [
    ("About", _ABOUT),
    ("How to Use", _HOW_TO_USE),
    ("Writing Custom Expressions", _CUSTOM_EXPRESSIONS),
    ("Predefined Equations", _PREDEFINED_EQUATIONS),
    ("Available Statistics", _statistics_text()),
    ("Solver Methods", _solver_methods_text()),
    ("Output Files", _OUTPUT_FILES),
    ("Configuration", _CONFIGURATION),
]


class HelpDialog:
    """Information window with collapsible sections.

    Args:
        parent: Parent window.
    """

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.win = tk.Toplevel(parent)
        self.win.title(f"{APP_NAME} — Information")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self._body_labels: list[ttk.Label] = []
        self._build_ui()

        self.win.update_idletasks()
        req_width = self.win.winfo_reqwidth()
        req_height = self.win.winfo_reqheight()
        
        screen_w = self.win.winfo_screenwidth()
        screen_h = self.win.winfo_screenheight()
        
        win_w = min(max(req_width + 40, 900), int(screen_w * 0.9))
        win_h = min(max(req_height + 40, 750), int(screen_h * 0.9))
        
        center_window(self.win, win_w, win_h)
        make_modal(self.win, parent)

    def _build_ui(self) -> None:
        pad: int = get_env_from_schema("UI_PADDING")
        bg: str = get_env_from_schema("UI_BACKGROUND")

        # Fixed bottom button bar
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=pad, pady=pad)

        btn_close = ttk.Button(
            btn_frame, text="Close", style="Cancel.TButton",
            command=self.win.destroy,
        )
        btn_close.pack()

        ttk.Separator(self.win, orient=tk.HORIZONTAL).pack(
            side=tk.BOTTOM, fill=tk.X,
        )

        setup_arrow_enter_navigation([[btn_close]])
        btn_close.focus_set()

        # Scrollable content
        self._scroll = ScrollableFrame(self.win)
        self._scroll.apply_bg(bg)
        self._scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        inner = self._scroll.inner
        inner.configure(padding=pad)

        ttk.Label(
            inner,
            text=f"{APP_NAME} — Information",
            style="Title.TLabel",
        ).pack(anchor=tk.W, pady=(0, pad))

        first = True
        for title, body in _SECTIONS:
            self._add_section(inner, title, body, expanded=first)
            first = False

        self._scroll.bind_new_children()

        def _update_wraplength(_e: tk.Event | None = None) -> None:  # type: ignore[type-arg]
            w = inner.winfo_width()
            if w > 100:
                wrap = max(200, w - 48)
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
        """Add a collapsible section (header + body) wrapped in a container."""
        pad: int = get_env_from_schema("UI_PADDING")
        section = CollapsibleSection(
            parent, self._scroll, title, expanded=expanded, pad=pad,
        )
        body_lbl = ttk.Label(
            section.content, text=body, justify=tk.LEFT, wraplength=620,
        )
        body_lbl.pack(anchor=tk.W, fill=tk.X)
        self._body_labels.append(body_lbl)
