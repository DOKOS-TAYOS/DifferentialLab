"""Parameters dialog — configure domain, ICs, method, and statistics."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

import numpy as np

from config.constants import AVAILABLE_STATISTICS, SOLVER_METHODS, SOLVER_METHOD_DESCRIPTIONS
from config.env import get_env_from_schema
from config.paths import generate_output_basename, get_csv_path, get_json_path, get_plot_path
from config.theme import get_font
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import center_window, make_modal
from plotting.plot_utils import create_phase_plot, create_solution_plot, save_plot
from solver.equation_parser import parse_expression
from solver.ode_solver import solve_ode
from solver.statistics import compute_statistics
from solver.validators import validate_all_inputs
from utils.exceptions import DifferentialLabError, EquationParseError
from utils.export import export_all_results
from utils.logger import get_logger

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

        center_window(self.win, 740, 700)
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
        """Parse inputs, run the solver, and open the result dialog."""
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

        errors = validate_all_inputs(
            self.expression, self.order, x_min, x_max,
            y0, n_points, method, self.parameters,
        )
        if errors:
            messagebox.showerror("Validation Error",
                                 "\n".join(errors), parent=self.win)
            return

        selected_stats = {k for k, v in self._stat_vars.items() if v.get()}

        try:
            ode_func = parse_expression(self.expression, self.order, self.parameters)
        except EquationParseError as exc:
            messagebox.showerror("Parse Error", str(exc), parent=self.win)
            return

        t_eval = np.linspace(x_min, x_max, n_points)

        try:
            solution = solve_ode(ode_func, (x_min, x_max), y0,
                                 method=method, t_eval=t_eval)
        except DifferentialLabError as exc:
            messagebox.showerror("Solver Error", str(exc), parent=self.win)
            return

        stats = compute_statistics(solution.x, solution.y, selected_stats)

        basename = generate_output_basename()
        csv_path = get_csv_path(basename)
        json_path = get_json_path(basename)
        plot_path = get_plot_path(basename)

        metadata = {
            "equation_name": self.equation_name,
            "expression": self.expression,
            "order": self.order,
            "parameters": self.parameters,
            "domain": [x_min, x_max],
            "initial_conditions": y0,
            "method": method,
            "num_points": n_points,
            "solver_success": solution.success,
            "solver_message": solution.message,
            "n_evaluations": solution.n_eval,
        }

        fig = create_solution_plot(
            solution.x, solution.y,
            title=self.equation_name,
            xlabel="x", ylabel="y",
        )
        save_plot(fig, plot_path)

        phase_fig = None
        if self.order >= 2:
            phase_fig = create_phase_plot(solution.y, title=f"{self.equation_name} — Phase")

        export_all_results(solution.x, solution.y, stats, metadata, csv_path, json_path)

        self.win.destroy()

        from frontend.ui_dialogs.result_dialog import ResultDialog

        ResultDialog(
            self.parent,
            fig=fig,
            phase_fig=phase_fig,
            statistics=stats,
            metadata=metadata,
            csv_path=csv_path,
            json_path=json_path,
            plot_path=plot_path,
        )
