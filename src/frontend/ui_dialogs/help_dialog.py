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
    f"{APP_NAME} is a graphical tool for solving and visualising differential "
    "equations, recurrence relations, and mathematical transforms. "
    "It supports:\n\n"
    "\u2022 Scalar ODEs — ordinary differential equations of any order\n"
    "\u2022 Vector ODEs — coupled systems (Lorenz attractor, Lotka-Volterra, "
    "double pendulum, and more)\n"
    "\u2022 Difference equations — recurrence relations (Fibonacci, logistic map, ...)\n"
    "\u2022 PDEs — 2-D elliptic equations solved with finite differences "
    "(Poisson, Laplace, general operator-based)\n"
    "\u2022 Function transforms — Fourier (FFT), Laplace, Taylor series, "
    "Hilbert, and Z-transform\n\n"
    "Under the hood the application relies on SciPy's solve_ivp integrator "
    "for ODEs and finite-difference discretisation for PDEs.\n\n"
    "Tip: hover over any button or field to see a tooltip with a short description."
)

_HOW_TO_USE = (
    "The main menu has five buttons: Solve Differential Equation, "
    "Function Transform, Information, Configuration, and Quit.\n\n"
    "Solving an equation step by step:\n"
    "1.  Click  Solve Differential Equation.\n"
    "2.  Pick a predefined equation from the list, or switch to the  Custom  "
    "tab and write your own expression.\n"
    "3.  Choose the equation type: ODE, Vector ODE, Difference, or PDE.\n"
    "4.  Click  Next  to open the parameters screen.\n"
    "5.  Set the domain, initial/boundary conditions, evaluation points, "
    "solver method (for ODEs), and statistics to compute.\n"
    "6.  Press  Solve  to run the computation.\n"
    "7.  Explore the results: interactive plots (solution curve, phase space, "
    "3-D trajectory, animation), computed statistics, and export options "
    "(CSV, JSON, PNG, MP4).\n\n"
    "Function transforms step by step:\n"
    "1.  Click  Function Transform.\n"
    "2.  Enter a scalar function f(x) — for example sin(x) or exp(-a*x).\n"
    "3.  (Optional) Define parameters such as a=1.0, omega=2.\n"
    "4.  Select a transform: Fourier, Laplace, Taylor, Hilbert, or Z-transform.\n"
    "5.  View the result and export the data to CSV or save the plot."
)

_CUSTOM_EXPRESSIONS = (
    "Expressions use Python / NumPy syntax. Below is the notation for each "
    "equation type.\n\n"
    "Scalar ODE:\n"
    "    Independent variable: x\n"
    "    f[0] = f(x),  f[1] = f\u2032(x),  f[2] = f\u2033(x), ...\n"
    "    Write the highest derivative in terms of the lower ones.\n\n"
    "Vector ODE:\n"
    "    f[i, k]  where  i = component index,  k = derivative order.\n"
    "    f[0,0] = component 0,  f[0,1] = its first derivative, etc.\n\n"
    "Difference equation:\n"
    "    Index variable: n\n"
    "    f[0] = f\u2099,  f[1] = f\u2099\u208A\u2081,  f[2] = f\u2099\u208A\u2082, ...\n"
    "    Write f\u2099\u208Aorder as an expression of previous terms.\n\n"
    "PDE (2-D elliptic):\n"
    "    Spatial variables: x, y.  Select the LHS operator from the dropdown "
    "and write the RHS expression.\n\n"
    "Available math functions:\n"
    "    sin  cos  tan  exp  log  log10  sqrt  abs\n"
    "    sinh  cosh  tanh  arcsin  arccos  arctan\n"
    "    floor  ceil  sign  heaviside  pi  e\n\n"
    "Example — damped harmonic oscillator (ODE, order 2):\n"
    "    Expression:   -2*gamma*f[1] - omega**2*f[0]\n"
    "    Parameters:   omega, gamma\n"
    "    Meaning:  f\u2033 = -2\u03b3 f\u2032 - \u03c9\u00b2 f"
)

_PREDEFINED_EQUATIONS = (
    "The app ships with a library of classic equations ready to explore.\n\n"
    "ODE (8 equations):\n"
    "    Simple Harmonic Oscillator \u00b7 Damped Oscillator \u00b7 Exponential "
    "Growth / Decay \u00b7 Logistic \u00b7 Van der Pol \u00b7 Pendulum \u00b7 "
    "RC Circuit \u00b7 Free Fall with Drag\n\n"
    "Vector ODE (9 systems):\n"
    "    Lorenz System \u00b7 Lotka-Volterra \u00b7 Duffing \u00b7 Rigid Body (Euler) "
    "\u00b7 Bloch Equations \u00b7 Schr\u00f6dinger \u00b7 Coupled Harmonic "
    "Oscillators \u00b7 Double Pendulum \u00b7 Damped Coupled System\n\n"
    "Difference (5 recurrences):\n"
    "    Geometric Growth \u00b7 Logistic Map \u00b7 Fibonacci \u00b7 "
    "Second-Order Linear Recurrence \u00b7 Discrete Logistic (Cobweb)\n\n"
    "PDE (elliptic 2-D):\n"
    "    Poisson 2-D \u00b7 Laplace 2-D \u00b7 General operator-based PDEs "
    "with configurable derivative terms (f\u2093\u2093, f\u1d67\u1d67, f\u2093\u1d67, \u2026)"
)

_OUTPUT_FILES = (
    "All files are saved on demand from the Results or Transform dialog.\n\n"
    "\u2022 Save CSV  \u2014  tabular data (x, f, f\u2032, f\u2080, f\u2032\u2080, \u2026) "
    "ready for spreadsheets or further analysis.\n"
    "\u2022 Save JSON  \u2014  full metadata, equation definition, and all computed "
    "statistics in a structured format.\n"
    "\u2022 Matplotlib toolbar  \u2014  use the floppy-disk icon below any plot to "
    "save it as PNG, JPG, SVG, or PDF.\n"
    "\u2022 MP4 animation  \u2014  export a video of the time evolution for vector "
    "ODE systems (requires ffmpeg in your PATH).\n\n"
    "The default output directory is  output/  and can be changed in "
    "Configuration \u2192 File Paths."
)

_FUNCTION_TRANSFORMS = (
    "Open  Function Transform  from the main menu to apply mathematical "
    "transforms to any scalar function f(x).\n\n"
    "Available transforms:\n"
    "\u2022 Original  \u2014  plot f(x) over the chosen range.\n"
    "\u2022 Fourier (FFT)  \u2014  magnitude spectrum |\u0046(\u03c9)|.\n"
    "\u2022 Laplace  \u2014  L(s) evaluated along the real axis.\n"
    "\u2022 Taylor series  \u2014  polynomial expansion around a centre point "
    "(order 1\u201315).\n"
    "\u2022 Hilbert  \u2014  discrete Hilbert transform H[f](x).\n"
    "\u2022 Z-transform  \u2014  magnitude spectrum on the unit circle.\n\n"
    "Display mode: switch between  Curve  (function vs domain) and  "
    "Coefficients  (a\u1d62 vs index) to inspect individual terms.\n\n"
    "Use  Export CSV  or the Matplotlib toolbar to save results."
)

_CONFIGURATION = (
    f"Almost every visual and numerical aspect of {APP_NAME} can be customised: "
    "UI colours and fonts, plot styling (colours, line width, markers, DPI), "
    "solver defaults (method, tolerances, step size), output paths, and "
    "logging verbosity.\n\n"
    "Open  Configuration  from the main menu to edit settings in a graphical "
    "form, or edit the  .env  file directly with any text editor.\n\n"
    "After saving, the application restarts automatically so changes take "
    "effect immediately."
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


_KEYBOARD_SHORTCUTS = (
    "Navigate the interface with the keyboard as well as the mouse.\n\n"
    "\u2022 Arrow keys  \u2014  move focus between buttons and fields.\n"
    "\u2022 Enter / Space  \u2014  activate the focused button.\n"
    "\u2022 Tab / Shift+Tab  \u2014  cycle through input fields."
)

_SECTIONS: list[tuple[str, str]] = [
    ("About", _ABOUT),
    ("How to Use", _HOW_TO_USE),
    ("Writing Custom Expressions", _CUSTOM_EXPRESSIONS),
    ("Predefined Equations", _PREDEFINED_EQUATIONS),
    ("Function Transforms", _FUNCTION_TRANSFORMS),
    ("Solver Methods", _solver_methods_text()),
    ("Available Statistics", _statistics_text()),
    ("Output Files", _OUTPUT_FILES),
    ("Configuration", _CONFIGURATION),
    ("Keyboard Shortcuts", _KEYBOARD_SHORTCUTS),
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
            text="Support us on YouTube",
            command=lambda: webbrowser.open(YOUTUBE_CHANNEL_URL),
            padding=(14, 8),
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
