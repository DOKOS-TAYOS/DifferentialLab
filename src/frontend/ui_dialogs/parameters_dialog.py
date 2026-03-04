"""Parameters dialog — configure domain, ICs, method, and statistics."""

from __future__ import annotations

import queue
import re
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from config import (
    AVAILABLE_STATISTICS,
    SOLVER_METHOD_DESCRIPTIONS,
    SOLVER_METHODS,
    get_default_solver_method,
    get_env_from_schema,
)
from frontend.theme import get_contrast_foreground, get_font
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import bind_wraplength, fit_and_center, make_modal
from utils import DifferentialLabError, get_logger

logger = get_logger(__name__)

_MAX_PDE_GRID = 1000


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
        parameters_schema: dict[str, dict[str, Any]] | None = None,
        display_formula: str | None = None,
        equation_type: str = "ode",
        variables: list[str] | None = None,
        vector_expressions: list[str] | None = None,
        vector_components: int = 1,
        pde_operator: str = "neg_laplacian",
        component_orders: tuple[int, ...] | None = None,
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
        self.equation_type = equation_type
        self.variables = variables if variables else ["x"]
        self.vector_expressions = vector_expressions
        self.vector_components = vector_components
        self.is_vector = (
            vector_expressions is not None and len(vector_expressions) > 0
        ) or equation_type == "vector_ode"
        self.pde_operator = pde_operator
        self.is_pde = equation_type == "pde" or len(self.variables) > 1
        self.component_orders = component_orders

        self.win = tk.Toplevel(parent)
        self.win.title(f"Parameters — {equation_name}")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self._y0_vars: list[tk.StringVar] = []
        self._x0_vars: list[tk.StringVar] = []
        self._eq_param_vars: dict[str, tk.StringVar] = {}

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
            btn_inner,
            text="Cancel",
            style="Cancel.TButton",
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
        ttk.Label(
            scroll_frame, text=f"Equation: {self.equation_name}", style="Subtitle.TLabel"
        ).pack(anchor=tk.W, pady=(0, pad))
        formula_lbl = ttk.Label(
            scroll_frame,
            text=self.display_formula,
            style="Small.TLabel",
            justify=tk.LEFT,
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
        domain_label = (
            "Domain (n\u2098\u1d62\u2099, n\u2098\u2090\u2093)"  # n_min, n_max
            if self.equation_type == "difference"
            else "Domain"
        )
        domain_frame = ttk.LabelFrame(left_col, text=domain_label, padding=pad)
        domain_frame.pack(fill=tk.X, pady=(0, pad))

        # n_min/x_min, n_max/x_max
        var0 = self.variables[0] if self.variables else "x"
        if self.equation_type == "difference":
            x_min_label = "n\u2098\u1d62\u2099:"
            x_max_label = "n\u2098\u2090\u2093:"
        elif self.is_pde:
            x_min_label = "x[0]\u2098\u1d62\u2099:"
            x_max_label = "x[0]\u2098\u2090\u2093:"
        else:
            x_min_label = f"{var0}\u2098\u1d62\u2099:"
            x_max_label = f"{var0}\u2098\u2090\u2093:"
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
        if self.is_pde and len(default_domain) >= 4:
            pde_label_1 = "x[1]"
            row_y = ttk.Frame(domain_frame)
            row_y.pack(fill=tk.X, pady=(pad, 0))
            ttk.Label(row_y, text=f"{pde_label_1}\u2098\u1d62\u2099:").pack(side=tk.LEFT)
            self.ymin_var = tk.StringVar(value=str(default_domain[2]))
            ttk.Entry(row_y, textvariable=self.ymin_var, width=12, font=get_font()).pack(
                side=tk.LEFT, padx=pad
            )
            ttk.Label(row_y, text=f"{pde_label_1}\u2098\u2090\u2093:").pack(side=tk.LEFT)
            self.ymax_var = tk.StringVar(value=str(default_domain[3]))
            ttk.Entry(row_y, textvariable=self.ymax_var, width=12, font=get_font()).pack(
                side=tk.LEFT, padx=pad
            )
            row_ny = ttk.Frame(domain_frame)
            row_ny.pack(fill=tk.X, pady=(pad, 0))
            ttk.Label(row_ny, text=f"Grid points ({pde_label_1}):").pack(side=tk.LEFT)
            self.npoints_y_var = tk.StringVar(value="1000")
            ttk.Entry(row_ny, textvariable=self.npoints_y_var, width=10, font=get_font()).pack(
                side=tk.LEFT, padx=pad
            )

        # Equation parameters (ω, γ, etc.) — left column, 2 per row
        if self.parameters:
            _sub_digits = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"

            def _subscript_n(n: int) -> str:
                """Return subscript digits for integer n (e.g. 12 → '₁₂')."""
                return "".join(_sub_digits[int(d)] if d.isdigit() else d for d in str(n))

            eq_params_frame = ttk.LabelFrame(left_col, text="Equation Parameters", padding=pad)
            eq_params_frame.pack(fill=tk.X, pady=(0, pad))
            params_items = list(self.parameters.items())

            def _display_name_for(pname: str) -> str:
                pinfo = self.parameters_schema.get(pname, {})
                m = re.match(r"^(.+)\[(\d+)\]$", pname)
                if m:
                    return f"{m.group(1)}{_subscript_n(int(m.group(2)))}"
                return pinfo.get("display", pname)

            label_width = max(len(_display_name_for(p)) for p in self.parameters) + 1

            for i in range(0, len(params_items), 2):
                row = ttk.Frame(eq_params_frame)
                row.pack(fill=tk.X, pady=2)
                for pname, val in params_items[i : i + 2]:
                    pinfo = self.parameters_schema.get(pname, {})
                    display_name = _display_name_for(pname)

                    # Detect list parameter
                    is_list_param = isinstance(val, list)
                    if is_list_param:
                        ttk.Label(row, text=f"{display_name}:", width=label_width).pack(
                            side=tk.LEFT,
                        )
                        default_csv = ", ".join(str(v) for v in val)
                        var = tk.StringVar(value=default_csv)
                        entry = ttk.Entry(row, textvariable=var, width=20, font=get_font())
                        entry.pack(side=tk.LEFT, padx=(pad, pad * 2))
                        self._eq_param_vars[pname] = var
                        m = re.match(r"^(.+)\[(\d+)\]$", pname)
                        n = int(m.group(2)) if m else len(val)
                        ToolTip(
                            entry,
                            pinfo.get("description", "")
                            or f"Comma-separated values ({n} components)",
                        )
                    else:
                        ttk.Label(row, text=f"{display_name}:", width=label_width).pack(
                            side=tk.LEFT,
                        )
                        var = tk.StringVar(value=str(val))
                        entry = ttk.Entry(row, textvariable=var, width=12, font=get_font())
                        entry.pack(side=tk.LEFT, padx=(pad, pad * 2))
                        self._eq_param_vars[pname] = var
                        ToolTip(entry, pinfo.get("description", ""))

        if self.equation_type != "difference" and not self.is_pde:
            row_n = ttk.Frame(domain_frame)
            row_n.pack(fill=tk.X, pady=(pad, 0))
            ttk.Label(row_n, text="Evaluation points:").pack(side=tk.LEFT)
            self.npoints_var = tk.StringVar(value=str(get_env_from_schema("SOLVER_NUM_POINTS")))
            npoints_entry = ttk.Entry(
                row_n, textvariable=self.npoints_var, width=10, font=get_font()
            )
            npoints_entry.pack(side=tk.LEFT, padx=pad)
            btn_decrease = ttk.Button(
                row_n,
                text="−",
                width=3,
                style="Small.TButton",
                command=lambda: self._change_npoints(0.1),
            )
            btn_decrease.pack(side=tk.LEFT, padx=(0, 2))
            btn_increase = ttk.Button(
                row_n,
                text="+",
                width=3,
                style="Small.TButton",
                command=lambda: self._change_npoints(10),
            )
            btn_increase.pack(side=tk.LEFT)
        elif self.is_pde:
            row_n = ttk.Frame(domain_frame)
            row_n.pack(fill=tk.X, pady=(pad, 0))
            ttk.Label(row_n, text="Grid points (x[0]):").pack(side=tk.LEFT)
            self.npoints_var = tk.StringVar(value="1000")
            ttk.Entry(row_n, textvariable=self.npoints_var, width=10, font=get_font()).pack(
                side=tk.LEFT, padx=pad
            )

        # Initial conditions (skip for PDE) — left column
        self._bc_vars: list[tk.StringVar] = []
        self._bc_type_vars: list[tk.StringVar] = []
        self._domain_shape_var: tk.StringVar | None = None
        self._mask_expr_var: tk.StringVar | None = None
        self._contour_bc_expr_var: tk.StringVar | None = None
        self._contour_bc_type_var: tk.StringVar | None = None
        self._rect_bc_frame: ttk.LabelFrame | None = None
        self._contour_bc_frame: ttk.LabelFrame | None = None
        if self.is_pde:
            # Domain shape selector
            shape_frame = ttk.LabelFrame(left_col, text="Domain Shape", padding=pad)
            shape_frame.pack(fill=tk.X, pady=(0, pad))

            row_shape = ttk.Frame(shape_frame)
            row_shape.pack(fill=tk.X)
            ttk.Label(row_shape, text="Shape:").pack(side=tk.LEFT)
            self._domain_shape_var = tk.StringVar(value="Rectangle")
            shape_combo = ttk.Combobox(
                row_shape,
                textvariable=self._domain_shape_var,
                values=["Rectangle", "Custom contour"],
                state="readonly",
                width=18,
                font=get_font(),
            )
            shape_combo.pack(side=tk.LEFT, padx=pad)
            shape_combo.bind("<<ComboboxSelected>>", self._on_domain_shape_change)

            # Mask expression entry (hidden by default)
            self._mask_row = ttk.Frame(shape_frame)
            ttk.Label(self._mask_row, text="Mask expr:").pack(side=tk.LEFT)
            self._mask_expr_var = tk.StringVar(value="x**2 + y**2 <= 1")
            mask_entry = ttk.Entry(
                self._mask_row,
                textvariable=self._mask_expr_var,
                width=30,
                font=get_font(),
            )
            mask_entry.pack(side=tk.LEFT, padx=pad)
            ToolTip(
                mask_entry,
                "Boolean expression defining the domain, e.g. x**2 + y**2 <= 1",
            )

            # Rectangular boundary conditions
            self._rect_bc_frame = ttk.LabelFrame(
                left_col,
                text="Boundary Conditions",
                padding=pad,
            )
            self._rect_bc_frame.pack(fill=tk.X, pady=(0, pad))

            idx0 = "x[0]"
            idx1 = "x[1]"
            boundaries = [
                (f"{idx1} = {idx1}\u2098\u1d62\u2099 (bottom)", idx0),
                (f"{idx1} = {idx1}\u2098\u2090\u2093 (top)", idx0),
                (f"{idx0} = {idx0}\u2098\u1d62\u2099 (left)", idx1),
                (f"{idx0} = {idx0}\u2098\u2090\u2093 (right)", idx1),
            ]
            for label_text, free_var in boundaries:
                row = ttk.Frame(self._rect_bc_frame)
                row.pack(fill=tk.X, pady=1)
                ttk.Label(row, text=f"{label_text}:", width=24).pack(side=tk.LEFT)
                bc_type_var = tk.StringVar(value="Dirichlet")
                bc_type_combo = ttk.Combobox(
                    row,
                    textvariable=bc_type_var,
                    values=["Dirichlet", "Neumann"],
                    state="readonly",
                    width=10,
                    font=get_font(),
                )
                bc_type_combo.pack(side=tk.LEFT, padx=(pad, 2))
                self._bc_type_vars.append(bc_type_var)
                bc_var = tk.StringVar(value="0")
                bc_entry = ttk.Entry(
                    row,
                    textvariable=bc_var,
                    width=16,
                    font=get_font(),
                )
                bc_entry.pack(side=tk.LEFT, padx=(2, 0))
                ToolTip(
                    bc_entry,
                    f"Expression as a function of {free_var}, e.g. sin(pi*{free_var}). "
                    "For Dirichlet: value. For Neumann: normal derivative.",
                )
                self._bc_vars.append(bc_var)

            # Contour boundary conditions (hidden by default)
            self._contour_bc_frame = ttk.LabelFrame(
                left_col,
                text="Contour Boundary Conditions",
                padding=pad,
            )
            row_cbc_type = ttk.Frame(self._contour_bc_frame)
            row_cbc_type.pack(fill=tk.X, pady=1)
            ttk.Label(row_cbc_type, text="BC type:").pack(side=tk.LEFT)
            self._contour_bc_type_var = tk.StringVar(value="Dirichlet")
            ttk.Combobox(
                row_cbc_type,
                textvariable=self._contour_bc_type_var,
                values=["Dirichlet", "Neumann"],
                state="readonly",
                width=10,
                font=get_font(),
            ).pack(side=tk.LEFT, padx=pad)

            row_cbc_expr = ttk.Frame(self._contour_bc_frame)
            row_cbc_expr.pack(fill=tk.X, pady=1)
            ttk.Label(row_cbc_expr, text="Value:").pack(side=tk.LEFT)
            self._contour_bc_expr_var = tk.StringVar(value="0")
            contour_bc_entry = ttk.Entry(
                row_cbc_expr,
                textvariable=self._contour_bc_expr_var,
                width=25,
                font=get_font(),
            )
            contour_bc_entry.pack(side=tk.LEFT, padx=pad)
            ToolTip(
                contour_bc_entry,
                "Expression for boundary value (Dirichlet) or normal derivative (Neumann), "
                "as a function of x and y.",
            )
        else:
            ic_frame = ttk.LabelFrame(left_col, text="Initial Conditions", padding=pad)
            ic_frame.pack(fill=tk.X, pady=(0, pad))

            _subscripts = "₀₁₂₃₄₅₆₇₈₉"
            ic_labels = self._ic_labels()
            if self.component_orders:
                n_ic = sum(self.component_orders)
            elif self.is_vector:
                n_ic = self.order * self.vector_components
            else:
                n_ic = self.order
            ic_label_width = max(len(label) for label in ic_labels) + 1
            x0_label_width = (
                max(
                    len(f"x{_subscripts[i] if i < len(_subscripts) else str(i)} =")
                    for i in range(n_ic)
                )
                + 1
            )
            x0_val = int(default_domain[0]) if is_diff else default_domain[0]
            default_x0_val = str(x0_val)
            for i in range(n_ic):
                row = ttk.Frame(ic_frame)
                row.pack(fill=tk.X, pady=2)
                default_val = default_y0[i] if i < len(default_y0) else 1.0
                sub = _subscripts[i] if i < len(_subscripts) else str(i)

                ttk.Label(row, text=f"{ic_labels[i]} =", width=ic_label_width).pack(side=tk.LEFT)
                var = tk.StringVar(value=str(default_val))
                ttk.Entry(row, textvariable=var, width=10, font=get_font()).pack(
                    side=tk.LEFT,
                    padx=(pad, pad * 2),
                )

                if self.equation_type != "difference":
                    ttk.Label(row, text=f"x{sub} =", width=x0_label_width).pack(side=tk.LEFT)
                    x_var = tk.StringVar(value=default_x0_val)
                    ttk.Entry(row, textvariable=x_var, width=10, font=get_font()).pack(
                        side=tk.LEFT,
                        padx=pad,
                    )
                    self._x0_vars.append(x_var)
                else:
                    self._x0_vars.append(tk.StringVar(value=default_x0_val))

                self._y0_vars.append(var)

        # Solver method (ODE only) — right column
        self.method_frame = ttk.LabelFrame(right_col, text="Solver Method", padding=pad)
        self.method_frame.pack(fill=tk.X, pady=(0, pad))

        self.method_var = tk.StringVar(value=get_default_solver_method())
        combo = ttk.Combobox(
            self.method_frame,
            textvariable=self.method_var,
            values=list(SOLVER_METHODS),
            state="readonly",
            width=15,
            font=get_font(),
        )
        combo.pack(anchor=tk.W)
        self.method_desc = ttk.Label(
            self.method_frame,
            text="",
            style="Small.TLabel",
            justify=tk.LEFT,
        )
        self.method_desc.pack(anchor=tk.W, pady=(2, 0))
        combo.bind("<<ComboboxSelected>>", self._on_method_change)
        self._on_method_change(None)
        if self.equation_type == "difference" or self.is_pde:
            self.method_frame.pack_forget()

        # Statistics listbox (extended selection) — right column
        stats_frame = ttk.LabelFrame(right_col, text="Statistics & Magnitudes", padding=pad)
        stats_frame.pack(fill=tk.X, pady=(0, pad))

        self._stat_keys = list(AVAILABLE_STATISTICS.keys())

        stats_list_frame = ttk.Frame(stats_frame)
        stats_list_frame.pack(fill=tk.X)

        btn_bg: str = get_env_from_schema("UI_BUTTON_BG")
        fg: str = get_env_from_schema("UI_FOREGROUND")
        stats_select_bg: str = get_env_from_schema("UI_BUTTON_FG")
        stats_select_fg: str = get_contrast_foreground(stats_select_bg)
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
            stats_frame,
            text="",
            style="Small.TLabel",
            justify=tk.LEFT,
        )
        self._stats_desc_label.pack(anchor=tk.W, pady=(4, 0))
        self._stats_listbox.bind("<<ListboxSelect>>", self._on_stats_select)

        bind_wraplength(scroll_frame, formula_lbl, pad=2 * pad, min_wrap=200)
        bind_wraplength(
            stats_frame,
            [self.method_desc, self._stats_desc_label],
            pad=2 * pad,
            min_wrap=150,
        )
        scroll.bind_new_children()
        btn_solve.focus_set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ic_labels(self) -> list[str]:
        subscripts = "₀₁₂₃₄₅₆₇₈₉"

        def _sub(i: int) -> str:
            return subscripts[i] if i < len(subscripts) else str(i)

        if self.equation_type == "difference":
            return [f"f{_sub(i)}" for i in range(self.order)]
        if self.is_vector:
            labels: list[str] = []
            orders = self.component_orders or tuple(
                self.order for _ in range(self.vector_components)
            )
            for c, comp_order in enumerate(orders):
                comp_sub = _sub(c)
                for k in range(comp_order):
                    if k == 0:
                        labels.append(f"f{comp_sub}")
                    else:
                        primes = "\u2032" * k
                        labels.append(f"f{primes}{comp_sub}")
            return labels
        labels = [f"f(x{subscripts[0]})"]
        for i in range(1, self.order):
            primes = "\u2032" * i
            sub = subscripts[i] if i < len(subscripts) else str(i)
            labels.append(f"f{primes}(x{sub})")
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

    def _on_domain_shape_change(self, _event: Any) -> None:
        """Toggle visibility between rectangular and custom contour BC sections."""
        is_custom = self._domain_shape_var and self._domain_shape_var.get() == "Custom contour"
        if is_custom:
            self._mask_row.pack(fill=tk.X, pady=(4, 0))
            if self._rect_bc_frame:
                self._rect_bc_frame.pack_forget()
            if self._contour_bc_frame:
                self._contour_bc_frame.pack(fill=tk.X, pady=(0, 4))
            # Suggest symmetric domain for custom contours (e.g. circles)
            if self.xmin_var and float(self.xmin_var.get() or 0) >= 0:
                self.xmin_var.set("-1.0")
                self.xmax_var.set("1.0")
            if self.ymin_var and float(self.ymin_var.get() or 0) >= 0:
                self.ymin_var.set("-1.0")
            if self.ymax_var:
                self.ymax_var.set("1.0")
        else:
            self._mask_row.pack_forget()
            if self._contour_bc_frame:
                self._contour_bc_frame.pack_forget()
            if self._rect_bc_frame:
                self._rect_bc_frame.pack(fill=tk.X, pady=(0, 4))

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
            import numpy as _np

            params: dict[str, Any] = {}
            for pname, var in self._eq_param_vars.items():
                raw = var.get().strip()
                # Detect list parameter: name[n]
                m = re.match(r"^(.+)\[(\d+)\]$", pname)
                if m:
                    base_name = m.group(1)
                    try:
                        values = [float(v.strip()) for v in raw.split(",")]
                    except ValueError:
                        messagebox.showerror(
                            "Invalid Parameter",
                            f"Parameter '{pname}' must be comma-separated numbers.",
                            parent=self.win,
                        )
                        return
                    # Store under base name so name[i] works in expressions
                    params[base_name] = _np.array(values)
                else:
                    try:
                        params[pname] = float(raw)
                    except ValueError:
                        messagebox.showerror(
                            "Invalid Parameter",
                            f"Parameter '{pname}' must be a number.",
                            parent=self.win,
                        )
                        return
            self.parameters = params

        try:
            x_min = float(self.xmin_var.get())
            x_max = float(self.xmax_var.get())
        except ValueError:
            domain_name = (
                "n\u2098\u1d62\u2099 and n\u2098\u2090\u2093"
                if self.equation_type == "difference"
                else "x\u2098\u1d62\u2099 and x\u2098\u2090\u2093"
            )
            messagebox.showerror(
                "Invalid Domain", f"{domain_name} must be numbers.", parent=self.win
            )
            return

        if self.is_pde:
            if self.ymin_var is None or self.ymax_var is None:
                messagebox.showerror(
                    "Invalid PDE",
                    "y\u2098\u1d62\u2099 and y\u2098\u2090\u2093 required.",
                    parent=self.win,
                )
                return
            try:
                y_min = float(self.ymin_var.get())
                y_max = float(self.ymax_var.get())
            except ValueError:
                messagebox.showerror(
                    "Invalid Domain",
                    "y\u2098\u1d62\u2099 and y\u2098\u2090\u2093 must be numbers.",
                    parent=self.win,
                )
                return
            try:
                n_points = int(self.npoints_var.get())
                n_points_y = int(self.npoints_y_var.get()) if self.npoints_y_var else n_points
            except (ValueError, AttributeError):
                messagebox.showerror(
                    "Invalid Grid", "Grid points must be integers.", parent=self.win
                )
                return
            if n_points > _MAX_PDE_GRID or n_points_y > _MAX_PDE_GRID:
                messagebox.showerror(
                    "Grid too large",
                    f"PDE grid is limited to {_MAX_PDE_GRID} points per axis to avoid "
                    f"excessive memory use. You entered {n_points}×{n_points_y}.",
                    parent=self.win,
                )
                return
            y0 = []
            x0_list = None
            method = "fdm"
        elif self.equation_type == "difference":
            n_points = int(x_max) - int(x_min) + 1
            x0_list = None
            method = "iteration"
            y_min = None
            y_max = None
            n_points_y = None
        else:
            try:
                n_points = int(self.npoints_var.get())
            except ValueError:
                messagebox.showerror(
                    "Invalid Grid", "Number of points must be an integer.", parent=self.win
                )
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

        # Collect PDE boundary condition expressions and new parameters
        bc_expressions: list[str] | None = None
        bc_types: list[str] | None = None
        mask_expression: str | None = None
        contour_bc_expression: str | None = None
        contour_bc_type: str | None = None

        if self.is_pde:
            is_custom_contour = (
                self._domain_shape_var is not None
                and self._domain_shape_var.get() == "Custom contour"
            )

            if is_custom_contour:
                mask_expression = self._mask_expr_var.get().strip() if self._mask_expr_var else None
                if not mask_expression:
                    messagebox.showerror(
                        "Missing Mask",
                        "Custom contour requires a mask expression.",
                        parent=self.win,
                    )
                    return
                contour_bc_type = (
                    self._contour_bc_type_var.get().strip().lower()
                    if self._contour_bc_type_var
                    else "dirichlet"
                )
                contour_bc_expression = (
                    self._contour_bc_expr_var.get().strip() or "0"
                    if self._contour_bc_expr_var
                    else "0"
                )
            else:
                # Rectangular: collect per-edge expressions and types
                if self._bc_vars:
                    bc_expressions = [var.get().strip() or "0" for var in self._bc_vars]
                if self._bc_type_vars:
                    bc_types = [var.get().strip().lower() for var in self._bc_type_vars]

        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        def _run_solver() -> None:
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
                    x0_list=x0_list,
                    equation_type=self.equation_type,
                    variables=self.variables,
                    y_min=y_min,
                    y_max=y_max,
                    n_points_y=n_points_y,
                    vector_expressions=self.vector_expressions,
                    vector_components=self.vector_components,
                    pde_operator=self.pde_operator,
                    component_orders=self.component_orders,
                    bc_expressions=bc_expressions,
                    bc_types=bc_types,
                    mask_expression=mask_expression,
                    contour_bc_expression=contour_bc_expression,
                    contour_bc_type=contour_bc_type,
                )
                result_queue.put(("success", result))
            except DifferentialLabError as exc:
                logger.warning("Solver pipeline failed (user-facing): %s", exc)
                result_queue.put(("error", ("DifferentialLabError", str(exc))))
            except (MemoryError, OSError) as exc:
                logger.error("Solver pipeline: memory/system error: %s", exc, exc_info=True)
                result_queue.put(
                    (
                        "error",
                        (
                            "Memory Error",
                            f"Not enough memory to solve: {exc}\n\n"
                            "Try reducing the grid size (points per axis).",
                        ),
                    )
                )
            except Exception as exc:
                logger.exception("Solver pipeline: unexpected error")
                result_queue.put(("error", ("Error", str(exc))))

        from frontend.ui_dialogs.loading_dialog import LoadingDialog

        loading = LoadingDialog(self.parent, message="Solving...")
        self.win.destroy()

        # Keep ParametersDialog alive until _check_result runs on the main thread.
        # Otherwise GC may collect it from the worker thread when the solver finishes,
        # causing "RuntimeError: main thread is not in main loop" in Variable.__del__.
        dialog_ref = self

        threading.Thread(target=_run_solver, daemon=True).start()

        def _release_tk_vars() -> None:
            """Clear all tk.StringVar references so GC doesn't call __del__ off-thread.

            Variables are collected into a list and dropped in a deferred after(0)
            callback. This ensures they are garbage-collected on the main thread,
            avoiding "RuntimeError: main thread is not in main loop" in Variable.__del__
            when the worker thread triggers GC.
            """
            d = dialog_ref
            vars_to_discard: list[tk.StringVar] = []
            vars_to_discard.extend(d._y0_vars)
            vars_to_discard.extend(d._x0_vars)
            vars_to_discard.extend(d._eq_param_vars.values())
            vars_to_discard.extend(d._bc_vars)
            vars_to_discard.extend(d._bc_type_vars)
            for attr in (
                "xmin_var",
                "xmax_var",
                "ymin_var",
                "ymax_var",
                "npoints_var",
                "npoints_y_var",
                "method_var",
                "_domain_shape_var",
                "_mask_expr_var",
                "_contour_bc_expr_var",
                "_contour_bc_type_var",
            ):
                if hasattr(d, attr):
                    v = getattr(d, attr)
                    if v is not None:
                        vars_to_discard.append(v)
                    setattr(d, attr, None)
            d._y0_vars.clear()
            d._x0_vars.clear()
            d._eq_param_vars.clear()
            d._bc_vars.clear()
            d._bc_type_vars.clear()

            def _discard_on_main() -> None:
                vars_to_discard.clear()

            d.parent.after(0, _discard_on_main)

        def _check_result() -> None:
            _ = dialog_ref  # Closure keeps dialog_ref alive until this callback runs
            try:
                status, data = result_queue.get_nowait()
                try:
                    loading.destroy()
                except tk.TclError:
                    pass
                _release_tk_vars()
                if not self.parent.winfo_exists():
                    return
                if status == "success":
                    from frontend.ui_dialogs.result_dialog import ResultDialog

                    ResultDialog(self.parent, result=data)
                else:
                    title, msg = data
                    messagebox.showerror(title, msg, parent=self.parent)
            except queue.Empty:
                if self.parent.winfo_exists():
                    self.parent.after(100, _check_result)

        self.parent.after(100, _check_result)
