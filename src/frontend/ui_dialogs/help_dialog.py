"""Help / Information dialog with collapsible sections."""

from __future__ import annotations

import tkinter as tk
import webbrowser
from tkinter import ttk

from config import (
    APP_NAME,
    APP_VERSION,
    AVAILABLE_STATISTICS,
    SOLVER_METHOD_DESCRIPTIONS,
    SOLVER_METHODS,
    get_env_from_schema,
)
from frontend.ui_dialogs.collapsible_section import CollapsibleSection
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.window_utils import fit_and_center, make_modal
from utils import get_logger

YOUTUBE_CHANNEL_URL = "https://www.youtube.com/@whenphysics"

logger = get_logger(__name__)

# ── Section content (human-readable) ─────────────────────────────────

_ABOUT = (
    f"Welcome to {APP_NAME} v{APP_VERSION}!\n\n"
    f"{APP_NAME} is a graphical tool for solving differential equations "
    "and recurrence relations numerically. It supports:\n\n"
    "\u2022 ODEs — scalar ordinary differential equations (any order)\n"
    "\u2022 Vector ODEs — coupled systems (Lorenz, Lotka-Volterra, etc.)\n"
    "\u2022 Difference equations — recurrence relations (Fibonacci, logistic map)\n"
    "\u2022 PDEs — 2D elliptic equations (Poisson, Laplace, general operator-based)\n"
    "\u2022 Function transforms — Fourier, Laplace, Taylor, Hilbert, Z-transform\n\n"
    "The application uses SciPy's integration engine (solve_ivp) for ODEs "
    "and finite-difference methods for PDEs. It enables users to explore "
    "how systems evolve over time in physics, engineering, and mathematics."
)

_HOW_TO_USE = (
    "Main menu: Solve, Function Transform, Information, Configuration, Quit.\n\n"
    "Solving an equation:\n"
    "1.  Click  Solve Differential Equation  in the main menu.\n"
    "2.  Select a predefined equation (ODE, Vector ODE, Difference, or PDE) "
    "or switch to the Custom tab and define a custom expression.\n"
    "3.  Set the domain, initial conditions, evaluation points, solver method "
    "(ODEs), and which statistics to compute.\n"
    "4.  Press  Solve  to run the computation.\n"
    "5.  The results window shows interactive plots (solution, phase space, "
    "3D trajectory, animation for vector ODEs), statistics, and export buttons.\n\n"
    "Function transforms:\n"
    "1.  Click  Function Transform  in the main menu.\n"
    "2.  Enter a scalar function f(x) and select a transform (Fourier, Laplace, "
    "Taylor, Hilbert, Z-transform).\n"
    "3.  View the result and export data to CSV or save the plot as PNG."
)

_CUSTOM_EXPRESSIONS = (
    "Use Python / NumPy syntax.\n\n"
    "ODEs: Independent variable  x . State vector  f  notation:\n"
    "    f[0] = f,  f[1] = f',  f[2] = f'',  ...\n\n"
    "Vector ODEs: Use  f[i, k]  where i = component, k = derivative:\n"
    "    f[0,0] = position of component 0, f[0,1] = its velocity, etc.\n\n"
    "Difference equations: Use  n  for the index, f[0] = f_n, f[1] = f_{n+1}, etc.\n\n"
    "PDEs: Use  x, y  (and optionally more) as independent variables.\n\n"
    "Available mathematical functions:\n"
    "    sin, cos, tan, exp, log, log10, sqrt, abs,\n"
    "    sinh, cosh, tanh, arcsin, arccos, arctan,\n"
    "    floor, ceil, sign, heaviside, pi, e\n\n"
    "Example — damped oscillator (order 2):\n"
    "    Expression:   -2*gamma*f[1] - omega**2*f[0]\n"
    "    Parameters:   omega, gamma"
)

_PREDEFINED_EQUATIONS = (
    "ODE: Simple Harmonic Oscillator, Damped Oscillator, Exponential Growth/Decay, "
    "Logistic, Van der Pol, Pendulum, RC Circuit, Free Fall with Drag.\n\n"
    "Vector ODE: Lorenz System, Lotka-Volterra, Duffing, Rigid Body (Euler), Bloch "
    "Equations, Schrödinger, Coupled Harmonic Oscillators, Double Pendulum, "
    "Damped Coupled System.\n\n"
    "Difference: Geometric Growth, Logistic Map, Fibonacci, Second-Order Linear "
    "Recurrence, Discrete Logistic (Cobweb).\n\n"
    "PDE: Poisson 2D, Laplace 2D, and general operator-based PDEs with "
    "configurable derivative terms (f\u2093\u2093, f\u1d67\u1d67, f\u2093\u1d67, etc.)."
)

_OUTPUT_FILES = (
    "Files are saved on demand from the Results dialog:\n\n"
    "\u2022 Save CSV... — tabular x, f, f′, f₀, f′₀, ... columns for spreadsheets.\n"
    "\u2022 Save JSON... — full metadata and all computed statistics.\n"
    "\u2022 Matplotlib toolbar — save the plot as PNG, JPG, or PDF.\n"
    "\u2022 MP4 export — for vector ODE animations (via the Animation tab).\n\n"
    "The output directory (default output/) is configurable in Configuration."
)

_FUNCTION_TRANSFORMS = (
    "Open  Function Transform  from the main menu to apply mathematical "
    "transforms to scalar functions f(x).\n\n"
    "Transforms: Original (f(x)), Fourier (FFT), Laplace (real axis), Taylor "
    "series, Hilbert (discrete), Z-transform (discrete).\n\n"
    "Display mode: switch between  Curve (f vs x)  and  Coefficients (a\u1d62 vs i) "
    "for Taylor, Fourier, Laplace, Hilbert, or Z-transform views.\n\n"
    "Export data to CSV or save the plot as PNG from the transform dialog."
)

_CONFIGURATION = (
    "Almost every visual and numerical aspect of "
    f"{APP_NAME} can be customised. Open  Configuration  from the main menu or edit the "
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
    ("Function Transforms", _FUNCTION_TRANSFORMS),
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

        fit_and_center(self.win, min_width=1000, min_height=750)
        make_modal(self.win, parent)

    def _build_ui(self) -> None:
        pad: int = get_env_from_schema("UI_PADDING")
        bg: str = get_env_from_schema("UI_BACKGROUND")

        # Fixed bottom button bar
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=pad, pady=pad)

        btn_youtube = ttk.Button(
            btn_frame,
            text="If you want to support",
            command=lambda: webbrowser.open(YOUTUBE_CHANNEL_URL),
            padding=(14,8)
        )
        btn_youtube.pack(side=tk.LEFT, padx=(0, pad))

        btn_close = ttk.Button(
            btn_frame,
            text="Close",
            style="Cancel.TButton",
            command=self.win.destroy,
        )
        btn_close.pack(side=tk.LEFT)

        setup_arrow_enter_navigation([[btn_youtube, btn_close]])
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
        self.win.after(50, lambda: _update_wraplength(None))

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
            parent,
            self._scroll,
            title,
            expanded=expanded,
            pad=pad,
        )
        body_lbl = ttk.Label(
            section.content,
            text=body,
            justify=tk.LEFT,
        )
        body_lbl.pack(anchor=tk.W, fill=tk.X)
        self._body_labels.append(body_lbl)
