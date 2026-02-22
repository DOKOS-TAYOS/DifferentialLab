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
from frontend.ui_dialogs.tooltip import ToolTip
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
        parameters_schema: dict[str, dict[str, Any]] | None = None,
        display_formula: str | None = None,
        equation_type: str = "ode",
        variables: list[str] | None = None,
        vector_expressions: list[str] | None = None,
        vector_components: int = 1,
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
        self.parameters_schema = parameters_schema or {}
        n_derivs = vector_components if (
            (vector_expressions and len(vector_expressions) > 0) or equation_type == "vector_ode"
        ) else order
        self.selected_derivatives = (
            selected_derivatives if selected_derivatives is not None else list(range(n_derivs))
        )
        self.equation_type = equation_type
        self.variables = variables if variables else ["x"]
        self.vector_expressions = vector_expressions
        self.vector_components = vector_components
        self.is_vector = (
            (vector_expressions is not None and len(vector_expressions) > 0)
            or equation_type == "vector_ode"
        )
        self.is_pde = equation_type == "pde" or len(self.variables) > 1

        self.win = tk.Toplevel(parent)
        self.win.title(f"Parameters — {equation_name}")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self._y0_vars: list[tk.StringVar] = []
        self._x0_vars: list[tk.StringVar] = []
        self._eq_param_vars: dict[str, tk.StringVar] = {}
        self._derivatives_listbox: tk.Listbox | None = None
        self._stat_keys: list[str] = []

        self._build_ui(default_y0, default_domain)

        fit_and_center(self.win, min_width=1050, min_height=700)
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


        # ── Scrollable content ──
        scroll = ScrollableFrame(self.win)
        scroll.apply_bg(get_env_from_schema("UI_BACKGROUND"))
        scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        scroll_frame = scroll.inner
        scroll_frame.configure(padding=pad)

        # Equation summary
        ttk.Label(scroll_frame, text=f"Equation: {self.equation_name}",
                  style="Subtitle.TLabel").pack(anchor=tk.W, pady=(0, pad))
        formula_lbl = ttk.Label(
            scroll_frame, text=self.display_formula,
            style="Small.TLabel", justify=tk.LEFT,
        )
        formula_lbl.pack(anchor=tk.W, pady=(0, pad))

        # Two-column layout: left = domain + ICs, right = solver + statistics
        columns_frame = ttk.Frame(scroll_frame)
        columns_frame.pack(fill=tk.BOTH, expand=True, pady=(0, pad))

        left_col = ttk.Frame(columns_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, pad))
        right_col = ttk.Frame(columns_frame)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Domain (left column)
        domain_label = "Domain (n_min, n_max)" if self.equation_type == "difference" else "Domain"
        domain_frame = ttk.LabelFrame(left_col, text=domain_label, padding=pad)
        domain_frame.pack(fill=tk.X, pady=(0, pad))

        x_min_label = "n_min:" if self.equation_type == "difference" else "x_min:"
        x_max_label = "n_max:" if self.equation_type == "difference" else "x_max:"
        row_d = ttk.Frame(domain_frame)
        row_d.pack(fill=tk.X)
        ttk.Label(row_d, text=x_min_label).pack(side=tk.LEFT)
        is_diff = self.equation_type == "difference"
        xmin_val = int(default_domain[0]) if is_diff else default_domain[0]
        self.xmin_var = tk.StringVar(value=str(xmin_val))
        ttk.Entry(row_d, textvariable=self.xmin_var, width=12, font=get_font()).pack(
            side=tk.LEFT, padx=pad
        )
        ttk.Label(row_d, text=x_max_label).pack(side=tk.LEFT)
        xmax_val = int(default_domain[1]) if is_diff else default_domain[1]
        self.xmax_var = tk.StringVar(value=str(xmax_val))
        ttk.Entry(row_d, textvariable=self.xmax_var, width=12, font=get_font()).pack(
            side=tk.LEFT, padx=pad
        )

        self.ymin_var: tk.StringVar | None = None
        self.ymax_var: tk.StringVar | None = None
        self.npoints_y_var: tk.StringVar | None = None
        self.plot_3d_var: tk.BooleanVar | None = None
        if self.is_pde and len(default_domain) >= 4:
            row_y = ttk.Frame(domain_frame)
            row_y.pack(fill=tk.X, pady=(pad, 0))
            ttk.Label(row_y, text="y_min:").pack(side=tk.LEFT)
            self.ymin_var = tk.StringVar(value=str(default_domain[2]))
            ttk.Entry(row_y, textvariable=self.ymin_var, width=12, font=get_font()).pack(
                side=tk.LEFT, padx=pad
            )
            ttk.Label(row_y, text="y_max:").pack(side=tk.LEFT)
            self.ymax_var = tk.StringVar(value=str(default_domain[3]))
            ttk.Entry(row_y, textvariable=self.ymax_var, width=12, font=get_font()).pack(
                side=tk.LEFT, padx=pad
            )
            row_ny = ttk.Frame(domain_frame)
            row_ny.pack(fill=tk.X, pady=(pad, 0))
            ttk.Label(row_ny, text="Grid points (y):").pack(side=tk.LEFT)
            self.npoints_y_var = tk.StringVar(value="1000")
            ttk.Entry(row_ny, textvariable=self.npoints_y_var, width=10, font=get_font()).pack(
                side=tk.LEFT, padx=pad
            )
            plot_frame = ttk.Frame(domain_frame)
            plot_frame.pack(fill=tk.X, pady=(pad, 0))
            self.plot_3d_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                plot_frame, text="3D surface plot (uncheck for 2D contour)",
                variable=self.plot_3d_var,
            ).pack(anchor=tk.W)

        # Equation parameters (ω, γ, etc.) — left column
        if self.parameters:
            eq_params_frame = ttk.LabelFrame(left_col, text="Equation Parameters", padding=pad)
            eq_params_frame.pack(fill=tk.X, pady=(0, pad))
            for pname, val in self.parameters.items():
                row = ttk.Frame(eq_params_frame)
                row.pack(fill=tk.X, pady=2)
                ttk.Label(row, text=f"{pname}:", width=12).pack(side=tk.LEFT)
                var = tk.StringVar(value=str(val))
                entry = ttk.Entry(row, textvariable=var, width=12, font=get_font())
                entry.pack(side=tk.LEFT, padx=(pad, 0))
                self._eq_param_vars[pname] = var
                pinfo = self.parameters_schema.get(pname, {})
                ToolTip(entry, pinfo.get("description", ""))

        # Derivatives to plot — right column (skip for PDE)
        if not self.is_pde:
            deriv_frame = ttk.LabelFrame(right_col, text="Derivatives to Plot", padding=pad)
            deriv_frame.pack(fill=tk.X, pady=(0, pad))
            vec_exprs = self.vector_expressions
            is_vec = (vec_exprs is not None and len(vec_exprs) > 0) or self.equation_type == "vector_ode"
            if is_vec and self.vector_components > 1:
                subscripts = "₀₁₂₃₄₅₆₇₈₉"
                def _sub(i: int) -> str:
                    return subscripts[i] if i < len(subscripts) else str(i)
                derivative_labels = [f"f_{_sub(i)}" for i in range(self.vector_components)]
            else:
                derivative_labels = ["y"] if self.order == 1 else [f"y[{i}]" for i in range(self.order)]
            deriv_list_frame = ttk.Frame(deriv_frame)
            deriv_list_frame.pack(fill=tk.X)
            deriv_scrollbar = ttk.Scrollbar(deriv_list_frame, orient=tk.VERTICAL)
            _btn_bg: str = get_env_from_schema("UI_BUTTON_BG")
            _fg: str = get_env_from_schema("UI_FOREGROUND")
            _select_bg: str = get_env_from_schema("UI_BUTTON_FG")
            _select_fg: str = "#000000"
            self._derivatives_listbox = tk.Listbox(
                deriv_list_frame,
                selectmode=tk.EXTENDED,
                height=min(len(derivative_labels), 6),
                bg=_btn_bg,
                fg=_fg,
                selectbackground=_select_bg,
                selectforeground=_select_fg,
                font=get_font(),
                exportselection=False,
                yscrollcommand=deriv_scrollbar.set,
            )
            deriv_scrollbar.config(command=self._derivatives_listbox.yview)
            self._derivatives_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
            deriv_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            for label in derivative_labels:
                self._derivatives_listbox.insert(tk.END, label)
            for i in self.selected_derivatives:
                if 0 <= i < len(derivative_labels):
                    self._derivatives_listbox.select_set(i)

        if self.equation_type != "difference" and not self.is_pde:
            row_n = ttk.Frame(domain_frame)
            row_n.pack(fill=tk.X, pady=(pad, 0))
            ttk.Label(row_n, text="Evaluation points:").pack(side=tk.LEFT)
            self.npoints_var = tk.StringVar(value=str(get_env_from_schema("SOLVER_NUM_POINTS")))
            npoints_entry = ttk.Entry(
                row_n, textvariable=self.npoints_var, width=10, font=get_font()
            )
            npoints_entry.pack(side=tk.LEFT, padx=pad)
            btn_decrease = ttk.Button(row_n, text="−", width=3, style="Small.TButton",
                                      command=lambda: self._change_npoints(0.1))
            btn_decrease.pack(side=tk.LEFT, padx=(0, 2))
            btn_increase = ttk.Button(row_n, text="+", width=3, style="Small.TButton",
                                      command=lambda: self._change_npoints(10))
            btn_increase.pack(side=tk.LEFT)
        elif self.is_pde:
            row_n = ttk.Frame(domain_frame)
            row_n.pack(fill=tk.X, pady=(pad, 0))
            ttk.Label(row_n, text="Grid points (x):").pack(side=tk.LEFT)
            self.npoints_var = tk.StringVar(value="1000")
            ttk.Entry(row_n, textvariable=self.npoints_var, width=10, font=get_font()).pack(
                side=tk.LEFT, padx=pad
            )

        # Initial conditions (skip for PDE) — left column
        if not self.is_pde:
            ic_frame = ttk.LabelFrame(left_col, text="Initial Conditions", padding=pad)
            ic_frame.pack(fill=tk.X, pady=(0, pad))

            _subscripts = "₀₁₂₃₄₅₆₇₈₉"
            ic_labels = self._ic_labels()
            n_ic = self.order * self.vector_components if self.is_vector else self.order
            x0_val = int(default_domain[0]) if is_diff else default_domain[0]
            default_x0_val = str(x0_val)
            for i in range(n_ic):
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

        # Solver method (ODE only) — right column
        self.method_frame = ttk.LabelFrame(right_col, text="Solver Method", padding=pad)
        self.method_frame.pack(fill=tk.X, pady=(0, pad))

        self.method_var = tk.StringVar(value=get_env_from_schema("SOLVER_DEFAULT_METHOD"))
        combo = ttk.Combobox(self.method_frame, textvariable=self.method_var,
                              values=list(SOLVER_METHODS), state="readonly", width=15,
                              font=get_font())
        combo.pack(anchor=tk.W)
        self.method_desc = ttk.Label(
            self.method_frame, text="", style="Small.TLabel", justify=tk.LEFT,
        )
        self.method_desc.pack(anchor=tk.W, pady=(2, 0))
        combo.bind("<<ComboboxSelected>>", self._on_method_change)
        self._on_method_change(None)
        if self.equation_type == "difference" or self.is_pde:
            self.method_frame.pack_forget()

        # Statistics listbox (extended selection) — right column
        stats_frame = ttk.LabelFrame(right_col, text="Statistics & Magnitudes", padding=pad)
        stats_frame.pack(fill=tk.X, pady=(0, pad))

        self._stat_keys: list[str] = list(AVAILABLE_STATISTICS.keys())

        stats_list_frame = ttk.Frame(stats_frame)
        stats_list_frame.pack(fill=tk.X)

        btn_bg: str = get_env_from_schema("UI_BUTTON_BG")
        fg: str = get_env_from_schema("UI_FOREGROUND")
        stats_select_bg: str = get_env_from_schema("UI_BUTTON_FG")
        stats_select_fg: str = "#000000"
        stats_scrollbar = ttk.Scrollbar(stats_list_frame, orient=tk.VERTICAL)
        self._stats_listbox = tk.Listbox(
            stats_list_frame,
            selectmode=tk.EXTENDED,
            height=min(len(self._stat_keys), 6),
            bg=btn_bg,
            fg=fg,
            selectbackground=stats_select_bg,
            selectforeground=stats_select_fg,
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

        self._stats_desc_label = ttk.Label(
            stats_frame, text="", style="Small.TLabel", justify=tk.LEFT,
        )
        self._stats_desc_label.pack(anchor=tk.W, pady=(4, 0))
        self._stats_listbox.bind("<<ListboxSelect>>", self._on_stats_select)

        def _update_wraplength(_e: tk.Event | None = None) -> None:  # type: ignore[type-arg]
            w = scroll_frame.winfo_width()
            if w > 100:
                formula_wrap = max(200, w - 2 * pad)
                formula_lbl.configure(wraplength=formula_wrap)
            # Use actual width of stats_frame for solver text (always visible)
            mw = stats_frame.winfo_width()
            if mw > 100:
                col_wrap = max(150, mw - 2 * pad)
                self.method_desc.configure(wraplength=col_wrap)
                self._stats_desc_label.configure(wraplength=col_wrap)

        scroll_frame.bind("<Configure>", _update_wraplength)
        stats_frame.bind("<Configure>", _update_wraplength)
        scroll.bind_new_children()
        self.win.after(50, _update_wraplength)
        self.win.after(150, _update_wraplength)
        btn_solve.focus_set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ic_labels(self) -> list[str]:
        subscripts = "₀₁₂₃₄₅₆₇₈₉"
        if self.equation_type == "difference":
            def _sub(i: int) -> str:
                return subscripts[i] if i < len(subscripts) else str(i)

            return [f"y{_sub(i)}" for i in range(self.order)]
        if self.is_vector:
            labels: list[str] = []
            for c in range(self.vector_components):
                comp_sub = subscripts[c] if c < len(subscripts) else str(c)
                for k in range(self.order):
                    if k == 0:
                        labels.append(f"f_{comp_sub}")
                    else:
                        primes = "'" * k
                        labels.append(f"f_{comp_sub}{primes}")
            return labels
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
        # Equation parameters
        if self._eq_param_vars:
            params: dict[str, float] = {}
            for pname, var in self._eq_param_vars.items():
                try:
                    params[pname] = float(var.get())
                except ValueError:
                    messagebox.showerror(
                        "Invalid Parameter",
                        f"Parameter '{pname}' must be a number.",
                        parent=self.win,
                    )
                    return
            self.parameters = params

        # Derivatives to plot
        if self._derivatives_listbox is not None:
            selected = list(self._derivatives_listbox.curselection())
            if not selected:
                messagebox.showwarning(
                    "No Derivatives Selected",
                    "Please select at least one derivative to plot.",
                    parent=self.win,
                )
                return
            self.selected_derivatives = selected

        try:
            x_min = float(self.xmin_var.get())
            x_max = float(self.xmax_var.get())
        except ValueError:
            domain_name = (
                "n_min and n_max" if self.equation_type == "difference" else "x_min and x_max"
            )
            messagebox.showerror("Invalid Domain",
                                 f"{domain_name} must be numbers.",
                                 parent=self.win)
            return

        if self.is_pde:
            if self.ymin_var is None or self.ymax_var is None:
                messagebox.showerror("Invalid PDE", "y_min and y_max required.",
                                     parent=self.win)
                return
            try:
                y_min = float(self.ymin_var.get())
                y_max = float(self.ymax_var.get())
            except ValueError:
                messagebox.showerror("Invalid Domain", "y_min and y_max must be numbers.",
                                     parent=self.win)
                return
            try:
                n_points = int(self.npoints_var.get())
                n_points_y = int(self.npoints_y_var.get()) if self.npoints_y_var else n_points
            except (ValueError, AttributeError):
                messagebox.showerror("Invalid Grid", "Grid points must be integers.",
                                     parent=self.win)
                return
            _max_pde_grid = 1000
            if n_points > _max_pde_grid or n_points_y > _max_pde_grid:
                messagebox.showerror(
                    "Grid too large",
                    f"PDE grid is limited to {_max_pde_grid} points per axis to avoid "
                    f"excessive memory use. You entered {n_points}×{n_points_y}.",
                    parent=self.win,
                )
                return
            y0 = []
            x0_list = None
            method = "fdm"
            plot_3d = self.plot_3d_var.get() if self.plot_3d_var else True
        elif self.equation_type == "difference":
            n_points = int(x_max) - int(x_min) + 1
            x0_list = None
            method = "iteration"
            y_min = None
            y_max = None
            n_points_y = None
            plot_3d = True
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
            y_min = None
            y_max = None
            n_points_y = None
            plot_3d = True

        y0_list: list[float] = []
        if not self.is_pde:
            for i, var in enumerate(self._y0_vars):
                try:
                    y0_list.append(float(var.get()))
                except ValueError:
                    messagebox.showerror(
                        "Invalid IC",
                        f"Initial condition {i} must be a number.",
                        parent=self.win,
                    )
                    return
        y0 = y0_list if not self.is_pde else []

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
                variables=self.variables,
                y_min=y_min,
                y_max=y_max,
                n_points_y=n_points_y,
                plot_3d=plot_3d,
                vector_expressions=self.vector_expressions,
                vector_components=self.vector_components,
            )
        except DifferentialLabError as exc:
            messagebox.showerror("Error", str(exc), parent=self.win)
            return
        except (MemoryError, OSError) as exc:
            messagebox.showerror(
                "Memory Error",
                f"Not enough memory to solve: {exc}\n\n"
                "Try reducing the grid size (points per axis).",
                parent=self.win,
            )
            return
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self.win)
            return

        self.win.destroy()

        from frontend.ui_dialogs.result_dialog import ResultDialog

        animation_data: dict | None = None
        if getattr(result, "is_vector", False) and getattr(result, "animation_fig", None):
            animation_data = {
                "x": result.x,
                "y": result.y,
                "order": getattr(result, "vector_order", 2),
                "vector_components": getattr(result, "vector_components", 1),
                "title": result.metadata.get("equation_name", "ODE") + " — f_i(x)",
            }
        ResultDialog(
            self.parent,
            fig=result.fig,
            phase_fig=result.phase_fig,
            statistics=result.statistics,
            metadata=result.metadata,
            x=result.x,
            y=result.y,
            y_grid=getattr(result, "y_grid", None),
            animation_fig=getattr(result, "animation_fig", None),
            animation_3d_fig=getattr(result, "animation_3d_fig", None),
            animation_data=animation_data,
        )
