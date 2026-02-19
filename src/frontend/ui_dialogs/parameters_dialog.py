"""Parameters dialog — configure domain, ICs, method, and statistics."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from config import (
    AVAILABLE_STATISTICS,
    SOLVER_METHODS,
    SOLVER_METHOD_DESCRIPTIONS,
    get_env_from_schema,
)

from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import center_window, make_modal
from utils import DifferentialLabError, get_logger

logger = get_logger(__name__)


class ParametersDialog:
    """Dialog for configuring solver parameters, ICs, and statistics.

    Args:
        parent: Parent window.
        expression: ODE expression string.
        order: ODE order.
        parameters: Parameter name-value mapping.
        equation_name: Display name.
        default_y0: Default initial conditions.
        default_domain: Default ``[x_min, x_max]``.
    """

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        expression: str,
        order: int,
        parameters: dict[str, float],
        equation_name: str,
        default_y0: list[float],
        default_domain: list[float],
    ) -> None:
        self.parent = parent
        self.expression = expression
        self.order = order
        self.parameters = parameters
        self.equation_name = equation_name

        self.win = tk.Toplevel(parent)
        self.win.title(f"Parameters — {equation_name}")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self._y0_vars: list[tk.StringVar] = []
        self._stat_vars: dict[str, tk.BooleanVar] = {}

        self._build_ui(default_y0, default_domain)

        self.win.update_idletasks()
        req_width = self.win.winfo_reqwidth()
        req_height = self.win.winfo_reqheight()
        
        screen_w = self.win.winfo_screenwidth()
        screen_h = self.win.winfo_screenheight()
        
        win_w = min(max(req_width + 40, 740), int(screen_w * 0.9))
        win_h = min(max(req_height + 40, 700), int(screen_h * 0.9))
        
        center_window(self.win, win_w, win_h)
        make_modal(self.win, parent)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self, default_y0: list[float], default_domain: list[float]) -> None:
        pad: int = get_env_from_schema("UI_PADDING")

        # ── Fixed bottom button bar ──
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=pad, pady=pad)

        btn_inner = ttk.Frame(btn_frame)
        btn_inner.pack()

        btn_solve = ttk.Button(btn_inner, text="Solve", command=self._on_solve)
        btn_solve.pack(side=tk.LEFT, padx=pad)

        btn_cancel = ttk.Button(
            btn_inner, text="Cancel", style="Cancel.TButton",
            command=self.win.destroy,
        )
        btn_cancel.pack(side=tk.LEFT, padx=pad)

        setup_arrow_enter_navigation([[btn_solve, btn_cancel]])

        ttk.Separator(self.win, orient=tk.HORIZONTAL).pack(
            side=tk.BOTTOM, fill=tk.X,
        )

        # ── Scrollable content ──
        scroll = ScrollableFrame(self.win)
        scroll.apply_bg(get_env_from_schema("UI_BACKGROUND"))
        scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        scroll_frame = scroll.inner
        scroll_frame.configure(padding=pad)

        # Equation summary
        ttk.Label(scroll_frame, text=f"Equation: {self.equation_name}",
                  style="Subtitle.TLabel").pack(anchor=tk.W, pady=(0, pad))
        ttk.Label(scroll_frame, text=f"Expression: {self.expression}",
                  style="Small.TLabel").pack(anchor=tk.W, pady=(0, pad))

        # Domain
        domain_frame = ttk.LabelFrame(scroll_frame, text="Domain", padding=pad)
        domain_frame.pack(fill=tk.X, pady=(0, pad))

        row_d = ttk.Frame(domain_frame)
        row_d.pack(fill=tk.X)
        ttk.Label(row_d, text="x_min:").pack(side=tk.LEFT)
        self.xmin_var = tk.StringVar(value=str(default_domain[0]))
        ttk.Entry(row_d, textvariable=self.xmin_var, width=12).pack(side=tk.LEFT, padx=pad)
        ttk.Label(row_d, text="x_max:").pack(side=tk.LEFT)
        self.xmax_var = tk.StringVar(value=str(default_domain[1]))
        ttk.Entry(row_d, textvariable=self.xmax_var, width=12).pack(side=tk.LEFT, padx=pad)

        row_n = ttk.Frame(domain_frame)
        row_n.pack(fill=tk.X, pady=(pad, 0))
        ttk.Label(row_n, text="Evaluation points:").pack(side=tk.LEFT)
        self.npoints_var = tk.StringVar(value=str(get_env_from_schema("SOLVER_NUM_POINTS")))
        ttk.Spinbox(row_n, from_=10, to=1000000, width=10,
                     textvariable=self.npoints_var).pack(side=tk.LEFT, padx=pad)

        # Initial conditions
        ic_frame = ttk.LabelFrame(scroll_frame, text="Initial Conditions", padding=pad)
        ic_frame.pack(fill=tk.X, pady=(0, pad))

        ic_labels = self._ic_labels()
        for i in range(self.order):
            row = ttk.Frame(ic_frame)
            row.pack(fill=tk.X, pady=2)
            default_val = default_y0[i] if i < len(default_y0) else 0.0
            ttk.Label(row, text=f"{ic_labels[i]} =", width=14).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default_val))
            ttk.Entry(row, textvariable=var, width=12).pack(side=tk.LEFT, padx=pad)
            self._y0_vars.append(var)

        # Solver method
        method_frame = ttk.LabelFrame(scroll_frame, text="Solver Method", padding=pad)
        method_frame.pack(fill=tk.X, pady=(0, pad))

        self.method_var = tk.StringVar(value=get_env_from_schema("SOLVER_DEFAULT_METHOD"))
        combo = ttk.Combobox(method_frame, textvariable=self.method_var,
                              values=list(SOLVER_METHODS), state="readonly", width=15)
        combo.pack(side=tk.LEFT)
        self.method_desc = ttk.Label(method_frame, text="", style="Small.TLabel")
        self.method_desc.pack(side=tk.LEFT, padx=(pad, 0))
        combo.bind("<<ComboboxSelected>>", self._on_method_change)
        self._on_method_change(None)

        # Statistics checkboxes
        stats_frame = ttk.LabelFrame(scroll_frame, text="Statistics & Magnitudes", padding=pad)
        stats_frame.pack(fill=tk.X, pady=(0, pad))

        for key, desc in AVAILABLE_STATISTICS.items():
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(stats_frame, text=key, variable=var)
            cb.pack(anchor=tk.W)
            ToolTip(cb, desc)
            self._stat_vars[key] = var

        scroll.bind_new_children()
        btn_solve.focus_set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ic_labels(self) -> list[str]:
        labels = ["y(x0)"]
        for i in range(1, self.order):
            primes = "'" * i
            labels.append(f"y{primes}(x0)")
        return labels

    def _on_method_change(self, _event: Any) -> None:
        method = self.method_var.get()
        desc = SOLVER_METHOD_DESCRIPTIONS.get(method, "")
        self.method_desc.config(text=desc)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def _on_solve(self) -> None:
        """Parse inputs, run the solver pipeline, and open the result dialog."""
        try:
            x_min = float(self.xmin_var.get())
            x_max = float(self.xmax_var.get())
        except ValueError:
            messagebox.showerror("Invalid Domain",
                                 "x_min and x_max must be numbers.",
                                 parent=self.win)
            return

        try:
            n_points = int(self.npoints_var.get())
        except ValueError:
            messagebox.showerror("Invalid Grid",
                                 "Number of points must be an integer.",
                                 parent=self.win)
            return

        y0: list[float] = []
        for i, var in enumerate(self._y0_vars):
            try:
                y0.append(float(var.get()))
            except ValueError:
                messagebox.showerror("Invalid IC",
                                     f"Initial condition {i} must be a number.",
                                     parent=self.win)
                return

        method = self.method_var.get()
        selected_stats = {k for k, v in self._stat_vars.items() if v.get()}

        try:
            from pipeline import run_solver_pipeline

            result = run_solver_pipeline(
                expression=self.expression,
                order=self.order,
                parameters=self.parameters,
                equation_name=self.equation_name,
                x_min=x_min,
                x_max=x_max,
                y0=y0,
                n_points=n_points,
                method=method,
                selected_stats=selected_stats,
            )
        except DifferentialLabError as exc:
            messagebox.showerror("Error", str(exc), parent=self.win)
            return

        self.win.destroy()

        from frontend.ui_dialogs.result_dialog import ResultDialog

        ResultDialog(
            self.parent,
            fig=result.fig,
            phase_fig=result.phase_fig,
            statistics=result.statistics,
            metadata=result.metadata,
            csv_path=result.csv_path,
            json_path=result.json_path,
            plot_path=result.plot_path,
        )
