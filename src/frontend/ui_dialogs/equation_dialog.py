"""Equation selection dialog — choose predefined or write custom ODE."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from config import get_env_from_schema
from frontend.theme import get_font, get_select_colors
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import fit_and_center, make_modal
from solver import load_predefined_equations
from utils import get_logger

logger = get_logger(__name__)


class EquationDialog:
    """Dialog for selecting or entering an ODE.

    Args:
        parent: Parent window.
    """

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Select Equation")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self.equations = load_predefined_equations()
        self._equation_keys: list[str] = list(self.equations.keys())
        self._filtered_keys: list[str] = list(self.equations.keys())
        self._selected_key: str | None = None
        self._param_vars: dict[str, tk.StringVar] = {}
        self._derivative_vars: list[tk.BooleanVar] = []
        self._equation_type_var = tk.StringVar(value="ode")

        self._build_ui()

        fit_and_center(self.win, min_width=1060, min_height=650)
        make_modal(self.win, parent)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        pad: int = get_env_from_schema("UI_PADDING")

        # ── Fixed bottom button bar ──
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=pad, pady=pad)

        btn_inner = ttk.Frame(btn_frame)
        btn_inner.pack()

        self._btn_next = ttk.Button(
            btn_inner, text="Next \u2192", command=self._on_next,
        )
        self._btn_next.pack(side=tk.LEFT, padx=pad)

        btn_cancel = ttk.Button(
            btn_inner, text="Cancel", style="Cancel.TButton",
            command=self.win.destroy,
        )
        btn_cancel.pack(side=tk.LEFT, padx=pad)

        setup_arrow_enter_navigation([[self._btn_next, btn_cancel]])

        ttk.Separator(self.win, orient=tk.HORIZONTAL).pack(
            side=tk.BOTTOM, fill=tk.X,
        )

        # ── Equation type selector ──
        type_frame = ttk.Frame(self.win)
        type_frame.pack(fill=tk.X, padx=pad, pady=(pad, 0))
        ttk.Label(type_frame, text="Equation type:", style="Subtitle.TLabel").pack(
            side=tk.LEFT, padx=(0, pad)
        )
        ttk.Radiobutton(
            type_frame,
            text="Differential (ODE)",
            variable=self._equation_type_var,
            value="ode",
            command=self._on_type_change,
        ).pack(side=tk.LEFT, padx=pad)
        ttk.Radiobutton(
            type_frame,
            text="Difference (recurrence)",
            variable=self._equation_type_var,
            value="difference",
            command=self._on_type_change,
        ).pack(side=tk.LEFT, padx=pad)
        ttk.Radiobutton(
            type_frame,
            text="PDE (multivariate)",
            variable=self._equation_type_var,
            value="pde",
            command=self._on_type_change,
        ).pack(side=tk.LEFT, padx=pad)

        # ── Notebook ──
        self._notebook = ttk.Notebook(self.win)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=pad, pady=pad)

        # --- Tab 1: Predefined ---
        predef_frame = ttk.Frame(self._notebook, padding=pad)
        self._notebook.add(predef_frame, text="  Predefined  ")

        left = ttk.Frame(predef_frame)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, pad))

        ttk.Label(left, text="Equations:", style="Subtitle.TLabel").pack(anchor=tk.W)

        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        btn_bg: str = get_env_from_schema("UI_BUTTON_BG")
        fg: str = get_env_from_schema("UI_FOREGROUND")
        select_bg, select_fg = get_select_colors(element_bg=btn_bg, text_fg=fg)
        self.eq_listbox = tk.Listbox(
            list_frame,
            width=28,
            height=18,
            bg=btn_bg,
            fg=fg,
            selectbackground=select_bg,
            selectforeground=select_fg,
            font=get_font(),
            yscrollcommand=scrollbar.set,
            exportselection=False,
        )
        scrollbar.config(command=self.eq_listbox.yview)
        self.eq_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.eq_listbox.bind("<<ListboxSelect>>", self._on_select_equation)

        right = ttk.Frame(predef_frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.desc_label = ttk.Label(
            right, text="", style="Small.TLabel", justify=tk.LEFT, wraplength=0,
        )
        self.desc_label.pack(anchor=tk.W, pady=(0, pad), fill=tk.X)

        self.params_frame = ttk.LabelFrame(right, text="Parameters", padding=pad)
        self.params_frame.pack(fill=tk.X, pady=(0, pad))

        self.derivatives_frame = ttk.LabelFrame(right, text="Derivatives to Plot", padding=pad)
        self.derivatives_frame.pack(fill=tk.X, pady=(0, pad))

        self._populate_equation_list()

        # --- Tab 2: Custom (scrollable) ---
        custom_frame = ttk.Frame(self._notebook)
        self._notebook.add(custom_frame, text="  Custom  ")

        custom_scroll = ScrollableFrame(custom_frame)
        custom_scroll.apply_bg(get_env_from_schema("UI_BACKGROUND"))
        custom_scroll.pack(fill=tk.BOTH, expand=True)
        ci = custom_scroll.inner
        ci.configure(padding=pad)

        self.custom_hint_label = ttk.Label(
            ci,
            text="Write the highest derivative as a Python expression.",
            style="Subtitle.TLabel",
        )
        self.custom_hint_label.pack(anchor=tk.W, pady=(0, pad))

        self.custom_hint_text = tk.StringVar(
            value=(
                "Use y[0] for y, y[1] for y', y[2] for y'', etc.\n"
                "Use x for the independent variable.\n"
                "Available: sin, cos, tan, exp, log, sqrt, pi, e, abs, \u2026\n"
                "Example (harmonic oscillator):  -\u03c9**2 * y[0]"
            )
        )
        self.custom_hint_detail = ttk.Label(
            ci, textvariable=self.custom_hint_text, style="Small.TLabel",
            justify=tk.LEFT,
        )
        self.custom_hint_detail.pack(anchor=tk.W, pady=(0, pad))

        unicode_frame = ttk.LabelFrame(
            ci, text="Unicode symbols — select and copy", padding=pad
        )
        unicode_frame.pack(fill=tk.X, pady=(0, pad))

        # Each entry shows the escape code (copyable) followed by '=' and the rendered character.
        _unicode_hint_display = (
            "\\u03B1=\u03B1  \\u03B2=\u03B2  \\u03B3=\u03B3  \\u03B4=\u03B4  \\u03B5=\u03B5\n"
            "\\u03B6=\u03B6  \\u03B7=\u03B7  \\u03B8=\u03B8  \\u03BB=\u03BB  \\u03BC=\u03BC\n"
            "\\u03BE=\u03BE  \\u03C0=\u03C0  \\u03C1=\u03C1  \\u03C3=\u03C3  \\u03C6=\u03C6\n"
            "\\u03C9=\u03C9  \\u0394=\u0394  \\u03A3=\u03A3  \\u03A6=\u03A6  \\u03A9=\u03A9"
        )
        _btn_bg: str = get_env_from_schema("UI_BUTTON_BG")
        _fg: str = get_env_from_schema("UI_FOREGROUND")
        unicode_text = tk.Text(
            unicode_frame,
            height=4,
            bg=_btn_bg,
            fg=_fg,
            font=get_font(),
            borderwidth=0,
            highlightthickness=0,
            wrap="none",
        )
        unicode_text.insert("1.0", _unicode_hint_display)
        unicode_text.config(state="disabled")
        unicode_text.pack(fill=tk.X)

        row_order = ttk.Frame(ci)
        row_order.pack(fill=tk.X, pady=(pad, pad))
        self.custom_order_label = ttk.Label(row_order, text="Order:")
        self.custom_order_label.pack(side=tk.LEFT)
        self.custom_order_var = tk.StringVar(value="2")
        _font = get_font()
        spinbox = ttk.Spinbox(row_order, from_=1, to=10, width=5,
                     textvariable=self.custom_order_var, font=_font)
        spinbox.pack(side=tk.LEFT, padx=(pad, 0))

        self.custom_expr_label = ttk.Label(ci, text="Expression for highest derivative:")
        self.custom_expr_label.pack(anchor=tk.W)
        self.custom_expr = tk.Text(
            ci, height=3, width=60,
            bg=_btn_bg,
            fg=_fg,
            insertbackground=_fg,
            font=get_font(),
        )
        self.custom_expr.pack(fill=tk.X, pady=(4, pad))

        ttk.Label(ci, text="Parameters (name=value, comma-separated):").pack(anchor=tk.W)
        self.custom_params = ttk.Entry(ci, width=50, font=get_font())
        self.custom_params.pack(fill=tk.X, pady=(4, pad))
        ToolTip(self.custom_params, "E.g.: \u03c9=1.0, \u03b3=0.1")

        self.custom_derivatives_frame = ttk.LabelFrame(
            ci, text="Derivatives to Plot", padding=pad
        )
        self.custom_derivatives_frame.pack(fill=tk.X, pady=(pad, 0))
        self._custom_deriv_listbox: tk.Listbox | None = None
        self.custom_order_var.trace_add("write", self._update_custom_derivatives)
        self._update_custom_derivatives()

        custom_scroll.bind_new_children()
        self.eq_listbox.focus_set()
        self._update_custom_hints()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _populate_equation_list(self) -> None:
        """Populate the listbox with equations matching the current type."""
        eq_type = self._equation_type_var.get()
        self._filtered_keys = [
            k for k, eq in self.equations.items()
            if getattr(eq, "equation_type", "ode") == eq_type
        ]
        self.eq_listbox.delete(0, tk.END)
        for key in self._filtered_keys:
            self.eq_listbox.insert(tk.END, self.equations[key].name)
        self._selected_key = None
        self.desc_label.config(text="")

    def _on_type_change(self) -> None:
        """When equation type changes, refresh the list and custom hints."""
        self._populate_equation_list()
        self._update_custom_hints()

    def _update_custom_hints(self) -> None:
        """Update custom tab hints based on equation type."""
        if self._equation_type_var.get() == "difference":
            self.custom_hint_label.config(
                text="Write y_{n+order} as a Python expression."
            )
            self.custom_hint_text.set(
                "Use n for the index, y[0] for y_n, y[1] for y_{n+1}, etc.\n"
                "Available: sin, cos, tan, exp, log, sqrt, pi, e, abs, \u2026\n"
                "Example (geometric growth):  r * y[0]"
            )
            self.custom_expr_label.config(text="Expression for y_{n+order}:")
        else:
            self.custom_hint_label.config(
                text="Write the highest derivative as a Python expression."
            )
            self.custom_hint_text.set(
                "Use y[0] for y, y[1] for y', y[2] for y'', etc.\n"
                "Use x for the independent variable.\n"
                "Available: sin, cos, tan, exp, log, sqrt, pi, e, abs, \u2026\n"
                "Example (harmonic oscillator):  -\u03c9**2 * y[0]"
            )
            self.custom_expr_label.config(text="Expression for highest derivative:")

    def _update_custom_derivatives(self, *_args: Any) -> None:
        """Rebuild the derivatives listbox when the ODE order changes."""
        try:
            order = int(self.custom_order_var.get())
        except ValueError:
            order = 1

        for child in self.custom_derivatives_frame.winfo_children():
            child.destroy()

        derivative_labels = ["y"] if order == 1 else [f"y[{i}]" for i in range(order)]
        btn_bg: str = get_env_from_schema("UI_BUTTON_BG")
        fg: str = get_env_from_schema("UI_FOREGROUND")
        lb = tk.Listbox(
            self.custom_derivatives_frame,
            selectmode=tk.EXTENDED,
            height=min(len(derivative_labels), 6),
            bg=btn_bg,
            fg=fg,
            font=get_font(),
            exportselection=False,
        )
        for label in derivative_labels:
            lb.insert(tk.END, label)
        lb.select_set(0, tk.END)
        lb.pack(fill=tk.X)
        self._custom_deriv_listbox = lb

    def _on_next(self) -> None:
        """Route to predefined or custom handler based on active tab."""
        idx = self._notebook.index(self._notebook.select())
        if idx == 0:
            self._on_next_predefined()
        else:
            self._on_next_custom()

    def _on_select_equation(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        sel = self.eq_listbox.curselection()
        if not sel:
            self.desc_label.config(text="")
            return
        idx = sel[0]
        key = self._filtered_keys[idx]
        eq = self.equations[key]
        self._selected_key = key

        self.desc_label.config(text=eq.description)

        for child in self.params_frame.winfo_children():
            child.destroy()
        self._param_vars.clear()

        pad: int = get_env_from_schema("UI_PADDING")
        for pname, pinfo in eq.parameters.items():
            row = ttk.Frame(self.params_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{pname}:", width=12).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(pinfo["default"]))
            entry = ttk.Entry(row, textvariable=var, width=12, font=get_font())
            entry.pack(side=tk.LEFT, padx=(pad, 0))
            self._param_vars[pname] = var
            ToolTip(entry, pinfo.get("description", ""))

        for child in self.derivatives_frame.winfo_children():
            child.destroy()
        self._derivative_vars.clear()

        derivative_labels = ["y"] if eq.order == 1 else [f"y[{i}]" for i in range(eq.order)]
        for i, label in enumerate(derivative_labels):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(self.derivatives_frame, text=label, variable=var)
            cb.pack(anchor=tk.W)
            self._derivative_vars.append(var)


    def _on_next_predefined(self) -> None:
        if self._selected_key is None:
            messagebox.showwarning("No Selection", "Please select an equation.",
                                   parent=self.win)
            return

        eq = self.equations[self._selected_key]
        params: dict[str, float] = {}
        for pname, var in self._param_vars.items():
            try:
                params[pname] = float(var.get())
            except ValueError:
                messagebox.showerror(
                    "Invalid Parameter",
                    f"Parameter '{pname}' must be a number.",
                    parent=self.win,
                )
                return

        selected_derivatives = [i for i, var in enumerate(self._derivative_vars) if var.get()]
        eq_type = getattr(eq, "equation_type", "ode")
        if not selected_derivatives and eq_type != "pde":
            messagebox.showwarning("No Derivatives Selected",
                                   "Please select at least one derivative to plot.",
                                   parent=self.win)
            return

        self.win.destroy()
        from frontend.ui_dialogs.parameters_dialog import ParametersDialog

        eq_type: str = getattr(eq, "equation_type", "ode")
        variables: list[str] = getattr(eq, "variables", ["x"])
        ParametersDialog(
            self.parent,
            expression=eq.expression,
            function_name=eq.function_name,
            order=eq.order,
            parameters=params,
            equation_name=eq.name,
            default_y0=eq.default_initial_conditions,
            default_domain=eq.default_domain,
            selected_derivatives=selected_derivatives,
            display_formula=eq.formula,
            equation_type=eq_type,
            variables=variables,
        )

    def _on_next_custom(self) -> None:
        from solver.equation_parser import normalize_unicode_escapes

        expr = normalize_unicode_escapes(self.custom_expr.get("1.0", tk.END).strip())
        if not expr:
            messagebox.showwarning("Empty Expression",
                                   "Please enter an ODE expression.",
                                   parent=self.win)
            return

        try:
            order = int(self.custom_order_var.get())
        except ValueError:
            messagebox.showerror("Invalid Order",
                                 "ODE order must be a positive integer.",
                                 parent=self.win)
            return

        params: dict[str, float] = {}
        raw_params = self.custom_params.get().strip()
        if raw_params:
            for pair in raw_params.split(","):
                pair = pair.strip()
                if "=" not in pair:
                    messagebox.showerror(
                        "Invalid Parameters",
                        f"Cannot parse '{pair}'. Use name=value format.",
                        parent=self.win,
                    )
                    return
                name, val_str = pair.split("=", 1)
                normalized_name = normalize_unicode_escapes(name.strip())
                try:
                    params[normalized_name] = float(val_str.strip())
                except ValueError:
                    messagebox.showerror(
                        "Invalid Parameters",
                        f"Value for '{normalized_name}' is not a number.",
                        parent=self.win,
                    )
                    return

        selected_derivatives = (
            list(self._custom_deriv_listbox.curselection())
            if self._custom_deriv_listbox is not None
            else []
        )
        if not selected_derivatives:
            messagebox.showwarning("No Derivatives Selected",
                                   "Please select at least one derivative to plot.",
                                   parent=self.win)
            return

        eq_type = self._equation_type_var.get()
        self.win.destroy()
        from frontend.ui_dialogs.parameters_dialog import ParametersDialog

        default_domain = [0, 50] if eq_type == "difference" else [0.0, 10.0]
        ParametersDialog(
            self.parent,
            expression=expr,
            function_name=None,
            order=order,
            parameters=params,
            equation_name="Custom difference" if eq_type == "difference" else "Custom ODE",
            default_y0=[1.0] * order,
            default_domain=default_domain,
            selected_derivatives=selected_derivatives,
            equation_type=eq_type,
        )
