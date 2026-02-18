"""Help / Information dialog."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from config.constants import APP_NAME, APP_VERSION, SOLVER_METHODS, SOLVER_METHOD_DESCRIPTIONS
from config.env import get_env_from_schema
from frontend.window_utils import center_window, make_modal
from utils.logger import get_logger

logger = get_logger(__name__)

_HELP_TEXT = f"""\
{APP_NAME} v{APP_VERSION}
{'=' * 40}

DifferentialLab — numerical ODE solver with a graphical interface.

WHAT DOES IT SOLVE?
-------------------
This application solves ordinary differential equations (ODEs) of the form:

  y'   = f(x, y)          (first order)
  y''  = f(x, y, y')      (second order)
  y^(n) = f(x, y, y', …)  (n-th order)

It converts higher-order ODEs into first-order systems and uses SciPy's
robust integrators (solve_ivp) to compute the solution numerically.


HOW TO USE
----------
1. Click "Solve" in the main menu.
2. Choose a predefined equation (and adjust its parameters) or write
   a custom expression in the "Custom" tab.
3. Set the domain [x_min, x_max], initial conditions, grid resolution,
   solver method, and which statistics to compute.
4. Click "Solve" to run the computation.
5. View the plot and statistics in the result window.
   Output files (CSV, JSON, plot image) are saved to the output/ folder.


WRITING CUSTOM EXPRESSIONS
--------------------------
Use Python/NumPy syntax. The independent variable is "x".
The state vector is "y":
  y[0] = y        (the function itself)
  y[1] = y'       (first derivative)
  y[2] = y''      (second derivative)
  ...

Available math functions:
  sin, cos, tan, exp, log, log10, sqrt, abs,
  sinh, cosh, tanh, arcsin, arccos, arctan,
  floor, ceil, sign, heaviside, pi, e

Example — damped oscillator:
  Order: 2
  Expression: -2*gamma*y[1] - omega**2*y[0]
  Parameters: omega=1.0, gamma=0.1


PREDEFINED EQUATIONS
--------------------
- Simple Harmonic Oscillator
- Damped Oscillator
- Exponential Growth / Decay
- Logistic Equation
- Van der Pol Oscillator
- Simple Pendulum
- RC Circuit (Discharge)
- Free Fall with Drag


AVAILABLE STATISTICS
--------------------
- Mean value               - RMS (root mean square)
- Standard deviation       - Maximum (value + location)
- Minimum (value + loc.)   - Integral (area under curve)
- Zero crossings count     - Period estimate
- Amplitude estimate       - Energy estimate (2nd order)


SOLVER METHODS
--------------
"""

for _m in SOLVER_METHODS:
    _HELP_TEXT += f"  {_m:10s}  {SOLVER_METHOD_DESCRIPTIONS[_m]}\n"

_HELP_TEXT += """

OUTPUT FILES
------------
Each solve generates three files in output/:
  - CSV  — x, y data columns
  - JSON — metadata + all computed statistics
  - Plot — image (PNG/JPG/PDF per configuration)


CONFIGURATION
-------------
Edit settings via the "Configuration" button or directly in the .env file.
Changes take effect on the next application launch.
"""


class HelpDialog:
    """Scrollable help/information window.

    Args:
        parent: Parent window.
    """

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.win = tk.Toplevel(parent)
        self.win.title(f"{APP_NAME} — Information")
        center_window(self.win, 680, 580)
        make_modal(self.win, parent)

        bg: str = get_env_from_schema("UI_BACKGROUND")
        fg: str = get_env_from_schema("UI_FOREGROUND")
        self.win.configure(bg=bg)

        pad: int = get_env_from_schema("UI_PADDING")

        frame = ttk.Frame(self.win, padding=pad)
        frame.pack(fill=tk.BOTH, expand=True)

        text = tk.Text(
            frame,
            wrap=tk.WORD,
            bg=get_env_from_schema("UI_BUTTON_BG"),
            fg=fg,
            font=(get_env_from_schema("UI_FONT_FAMILY"), max(10, get_env_from_schema("UI_FONT_SIZE") - 4)),
            padx=12,
            pady=12,
            insertbackground=fg,
            selectbackground=get_env_from_schema("UI_TEXT_SELECT_BG"),
            relief=tk.FLAT,
        )
        text.insert("1.0", _HELP_TEXT)
        text.configure(state=tk.DISABLED)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)

        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(self.win, text="Close", style="Cancel.TButton",
                   command=self.win.destroy).pack(pady=pad)
