"""Parameters dialog — configure domain, ICs, method, and statistics."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from config import (
    AVAILABLE_STATISTICS,
    SOLVER_METHOD_DESCRIPTIONS,
    SOLVER_METHODS,
    get_env_from_schema,
)
from frontend.theme import get_font
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.window_utils import fit_and_center, make_modal
from utils import DifferentialLabError, get_logger

logger = get_logger(__name__)


class ParametersDialog:
    """Dialog for configuring solver parameters, ICs, and statistics.

    Args:
        parent: Parent window.
        expression: ODE expression string (optional).
        function_name: Name of function in config.equations (optional).
        order: ODE order.
        parameters: Parameter name-value mapping.
        equation_name: Display name.
        default_y0: Default initial conditions.
        default_domain: Default ``[x_min, x_max]``.
    """

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        *,
        expression: str | None = None,
        function_name: str | None = None,
        order: int,
        parameters: dict[str, float],
        equation_name: str,
        default_y0: list[float],
        default_domain: list[float],
        selected_derivatives: list[int] | None = None,
        display_formula: str | None = None,
        equation_type: str = "ode",
    ) -> None:
        self.parent = parent
        self.expression = expression
        self.function_name = function_name
        self.order = order
        self.parameters = parameters
        self.equation_name = equation_name
        self.display_formula = (
            display_formula
            if display_formula is not None
            else (expression or f"<function:{function_name}>")
        )
        self.selected_derivatives = (
            selected_derivatives if selected_derivatives else list(range(order))
        )
        self.equation_type = equation_type

        self.win = tk.Toplevel(parent)
        self.win.title(f"Parameters — {equation_name}")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self._y0_vars: list[tk.StringVar] = []
        self._x0_vars: list[tk.StringVar] = []
        self._stat_keys: list[str] = []

        self._build_ui(default_y0, default_domain)

        fit_and_center(self.win, min_width=920, min_height=700)
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
        ttk.Label(scroll_frame, text=self.display_formula,
                  style="Small.TLabel", wraplength=600, justify=tk.LEFT).pack(
            anchor=tk.W, pady=(0, pad)
        )

        # Domain
        domain_label = "Domain (n_min, n_max)" if self.equation_type == "difference" else "Domain"
        domain_frame = ttk.LabelFrame(scroll_frame, text=domain_label, padding=pad)
        domain_frame.pack(fill=tk.X, pady=(0, pad))

        x_min_label = "n_min:" if self.equation_type == "difference" else "x_min:"
        x_max_label = "n_max:" if self.equation_type == "difference" else "x_max:"
        row_d = ttk.Frame(domain_frame)
        row_d.pack(fill=tk.X)
        ttk.Label(row_d, text=x_min_label).pack(side=tk.LEFT)
        self.xmin_var = tk.StringVar(value=str(int(default_domain[0]) if self.equation_type == "difference" else default_domain[0]))
        ttk.Entry(row_d, textvariable=self.xmin_var, width=12, font=get_font()).pack(
            side=tk.LEFT, padx=pad
        )
        ttk.Label(row_d, text=x_max_label).pack(side=tk.LEFT)
        self.xmax_var = tk.StringVar(value=str(int(default_domain[1]) if self.equation_type == "difference" else default_domain[1]))
        ttk.Entry(row_d, textvariable=self.xmax_var, width=12, font=get_font()).pack(
            side=tk.LEFT, padx=pad
        )

        if self.equation_type != "difference":
            row_n = ttk.Frame(domain_frame)
            row_n.pack(fill=tk.X, pady=(pad, 0))
            ttk.Label(row_n, text="Evaluation points:").pack(side=tk.LEFT)
            self.npoints_var = tk.StringVar(value=str(get_env_from_schema("SOLVER_NUM_POINTS")))
            npoints_entry = ttk.Entry(row_n, textvariable=self.npoints_var, width=10, font=get_font())
            npoints_entry.pack(side=tk.LEFT, padx=pad)
            btn_decrease = ttk.Button(row_n, text="−", width=3, style="Small.TButton",
                                      command=lambda: self._change_npoints(0.1))
            btn_decrease.pack(side=tk.LEFT, padx=(0, 2))
            btn_increase = ttk.Button(row_n, text="+", width=3, style="Small.TButton",
                                      command=lambda: self._change_npoints(10))
            btn_increase.pack(side=tk.LEFT)

        # Initial conditions
        ic_frame = ttk.LabelFrame(scroll_frame, text="Initial Conditions", padding=pad)
        ic_frame.pack(fill=tk.X, pady=(0, pad))

        _subscripts = "₀₁₂₃₄₅₆₇₈₉"
        ic_labels = self._ic_labels()
        default_x0_val = str(int(default_domain[0]) if self.equation_type == "difference" else default_domain[0])
        for i in range(self.order):
            row = ttk.Frame(ic_frame)
            row.pack(fill=tk.X, pady=2)
            default_val = default_y0[i] if i < len(default_y0) else 1.0
            sub = _subscripts[i] if i < len(_subscripts) else str(i)

            ttk.Label(row, text=f"{ic_labels[i]} =", width=14).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default_val))
            ttk.Entry(row, textvariable=var, width=10, font=get_font()).pack(
                side=tk.LEFT, padx=(pad, pad * 2),
            )

            if self.equation_type != "difference":
                ttk.Label(row, text=f"x{sub} =").pack(side=tk.LEFT)
                x_var = tk.StringVar(value=default_x0_val)
                ttk.Entry(row, textvariable=x_var, width=10, font=get_font()).pack(
                    side=tk.LEFT, padx=pad,
                )
                self._x0_vars.append(x_var)
            else:
                self._x0_vars.append(tk.StringVar(value=default_x0_val))

            self._y0_vars.append(var)

        # Solver method (ODE only)
        self.method_frame = ttk.LabelFrame(scroll_frame, text="Solver Method", padding=pad)
        self.method_frame.pack(fill=tk.X, pady=(0, pad))

        self.method_var = tk.StringVar(value=get_env_from_schema("SOLVER_DEFAULT_METHOD"))
        combo = ttk.Combobox(self.method_frame, textvariable=self.method_var,
                              values=list(SOLVER_METHODS), state="readonly", width=15,
                              font=get_font())
        combo.pack(anchor=tk.W)
        self.method_desc = ttk.Label(self.method_frame, text="", style="Small.TLabel",
                                     wraplength=600, justify=tk.LEFT)
        self.method_desc.pack(anchor=tk.W, pady=(2, 0))
        combo.bind("<<ComboboxSelected>>", self._on_method_change)
        self._on_method_change(None)
        if self.equation_type == "difference":
            self.method_frame.pack_forget()

        # Statistics listbox (extended selection)
        stats_frame = ttk.LabelFrame(scroll_frame, text="Statistics & Magnitudes", padding=pad)
        stats_frame.pack(fill=tk.X, pady=(0, pad))

        self._stat_keys: list[str] = list(AVAILABLE_STATISTICS.keys())

        stats_list_frame = ttk.Frame(stats_frame)
        stats_list_frame.pack(fill=tk.X)

        btn_bg: str = get_env_from_schema("UI_BUTTON_BG")
        fg: str = get_env_from_schema("UI_FOREGROUND")
        stats_scrollbar = ttk.Scrollbar(stats_list_frame, orient=tk.VERTICAL)
        self._stats_listbox = tk.Listbox(
            stats_list_frame,
            selectmode=tk.EXTENDED,
            height=min(len(self._stat_keys), 6),
            bg=btn_bg,
            fg=fg,
            font=get_font(),
            exportselection=False,
            yscrollcommand=stats_scrollbar.set,
        )
        stats_scrollbar.config(command=self._stats_listbox.yview)
        self._stats_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for key in self._stat_keys:
            self._stats_listbox.insert(tk.END, key)
        self._stats_listbox.select_set(0, tk.END)

        self._stats_desc_label = ttk.Label(stats_frame, text="", style="Small.TLabel",
                                            wraplength=600, justify=tk.LEFT)
        self._stats_desc_label.pack(anchor=tk.W, pady=(4, 0))
        self._stats_listbox.bind("<<ListboxSelect>>", self._on_stats_select)

        scroll.bind_new_children()
        btn_solve.focus_set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ic_labels(self) -> list[str]:
        subscripts = "₀₁₂₃₄₅₆₇₈₉"
        if self.equation_type == "difference":
            return [f"y{subscripts[i] if i < len(subscripts) else str(i)}" for i in range(self.order)]
        labels = [f"y(x{subscripts[0]})"]
        for i in range(1, self.order):
            primes = "'" * i
            sub = subscripts[i] if i < len(subscripts) else str(i)
            labels.append(f"y{primes}(x{sub})")
        return labels

    def _change_npoints(self, factor: float) -> None:
        """Change evaluation points by an order of magnitude.

        Args:
            factor: Multiplication factor (10 to increase, 0.1 to decrease).
        """
        try:
            current = int(self.npoints_var.get())
            new_value = max(10, int(current * factor))
            self.npoints_var.set(str(new_value))
        except ValueError:
            # If invalid, reset to default
            self.npoints_var.set(str(get_env_from_schema("SOLVER_NUM_POINTS")))

    def _on_method_change(self, _event: Any) -> None:
        method = self.method_var.get()
        desc = SOLVER_METHOD_DESCRIPTIONS.get(method, "")
        self.method_desc.config(text=desc)

    def _on_stats_select(self, _event: Any) -> None:
        indices = self._stats_listbox.curselection()
        if not indices:
            self._stats_desc_label.config(text="")
            return
        last_key = self._stat_keys[indices[-1]]
        desc = AVAILABLE_STATISTICS.get(last_key, "")
        self._stats_desc_label.config(text=desc)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def _on_solve(self) -> None:
        """Parse inputs, run the solver pipeline, and open the result dialog."""
        try:
            x_min = float(self.xmin_var.get())
            x_max = float(self.xmax_var.get())
        except ValueError:
            domain_name = "n_min and n_max" if self.equation_type == "difference" else "x_min and x_max"
            messagebox.showerror("Invalid Domain",
                                 f"{domain_name} must be numbers.",
                                 parent=self.win)
            return

        if self.equation_type == "difference":
            n_points = int(x_max) - int(x_min) + 1
            x0_list = None
            method = "iteration"
        else:
            try:
                n_points = int(self.npoints_var.get())
            except ValueError:
                messagebox.showerror("Invalid Grid",
                                     "Number of points must be an integer.",
                                     parent=self.win)
                return
            subscripts = "₀₁₂₃₄₅₆₇₈₉"
            x0_list = []
            for i, x_var in enumerate(self._x0_vars):
                sub = subscripts[i] if i < len(subscripts) else str(i)
                try:
                    x0_list.append(float(x_var.get()))
                except ValueError:
                    messagebox.showerror(
                        "Invalid IC Point",
                        f"x{sub} must be a number.",
                        parent=self.win,
                    )
                    return
            method = self.method_var.get()

        y0: list[float] = []
        for i, var in enumerate(self._y0_vars):
            try:
                y0.append(float(var.get()))
            except ValueError:
                messagebox.showerror(
                    "Invalid IC",
                    f"Initial condition {i} must be a number.",
                    parent=self.win,
                )
                return

        selected_indices = self._stats_listbox.curselection()
        selected_stats = {self._stat_keys[i] for i in selected_indices}

        try:
            from pipeline import run_solver_pipeline

            result = run_solver_pipeline(
                expression=self.expression,
                function_name=self.function_name,
                order=self.order,
                parameters=self.parameters,
                equation_name=self.equation_name,
                x_min=x_min,
                x_max=x_max,
                y0=y0,
                n_points=n_points,
                method=method,
                selected_stats=selected_stats,
                selected_derivatives=self.selected_derivatives,
                x0_list=x0_list,
                equation_type=self.equation_type,
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
