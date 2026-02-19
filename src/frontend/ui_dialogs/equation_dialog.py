"""Equation selection dialog — choose predefined or write custom ODE."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from config import get_env_from_schema, get_font
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import center_window, make_modal
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
        self._selected_key: str | None = None
        self._param_vars: dict[str, tk.StringVar] = {}

        self._build_ui()

        center_window(self.win, 820, 650)
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
        self.eq_listbox = tk.Listbox(
            list_frame,
            width=28,
            height=18,
            bg=get_env_from_schema("UI_BUTTON_BG"),
            fg=get_env_from_schema("UI_FOREGROUND"),
            selectbackground=get_env_from_schema("UI_TEXT_SELECT_BG"),
            font=get_font(),
            yscrollcommand=scrollbar.set,
            exportselection=False,
        )
        scrollbar.config(command=self.eq_listbox.yview)
        self.eq_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for key, eq in self.equations.items():
            self.eq_listbox.insert(tk.END, eq.name)

        self.eq_listbox.bind("<<ListboxSelect>>", self._on_select_equation)

        right = ttk.Frame(predef_frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.desc_label = ttk.Label(right, text="", style="Small.TLabel",
                                     wraplength=380, justify=tk.LEFT)
        self.desc_label.pack(anchor=tk.W, pady=(0, pad))

        self.params_frame = ttk.LabelFrame(right, text="Parameters", padding=pad)
        self.params_frame.pack(fill=tk.X, pady=(0, pad))

        self.defaults_label = ttk.Label(right, text="", style="Small.TLabel",
                                         wraplength=380, justify=tk.LEFT)
        self.defaults_label.pack(anchor=tk.W)

        # --- Tab 2: Custom ---
        custom_frame = ttk.Frame(self._notebook, padding=pad)
        self._notebook.add(custom_frame, text="  Custom  ")

        ttk.Label(
            custom_frame,
            text="Write the highest derivative as a Python expression.",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(0, pad))

        hint = (
            "Use y[0] for y, y[1] for y', y[2] for y'', etc.\n"
            "Use x for the independent variable.\n"
            "Available: sin, cos, tan, exp, log, sqrt, pi, e, abs, \u2026\n"
            "Example (harmonic oscillator):  -omega**2 * y[0]"
        )
        ttk.Label(custom_frame, text=hint, style="Small.TLabel",
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(0, pad))

        row_order = ttk.Frame(custom_frame)
        row_order.pack(fill=tk.X, pady=(0, pad))
        ttk.Label(row_order, text="ODE Order:").pack(side=tk.LEFT)
        self.custom_order_var = tk.StringVar(value="2")
        ttk.Spinbox(row_order, from_=1, to=10, width=5,
                     textvariable=self.custom_order_var).pack(side=tk.LEFT, padx=(pad, 0))

        ttk.Label(custom_frame, text="Expression for highest derivative:").pack(anchor=tk.W)
        self.custom_expr = tk.Text(
            custom_frame, height=3, width=60,
            bg=get_env_from_schema("UI_BUTTON_BG"),
            fg=get_env_from_schema("UI_FOREGROUND"),
            insertbackground=get_env_from_schema("UI_FOREGROUND"),
            font=get_font(),
        )
        self.custom_expr.pack(fill=tk.X, pady=(4, pad))

        ttk.Label(custom_frame, text="Parameters (name=value, comma-separated):").pack(anchor=tk.W)
        self.custom_params = ttk.Entry(custom_frame, width=50)
        self.custom_params.pack(fill=tk.X, pady=(4, pad))
        ToolTip(self.custom_params, "E.g.: omega=1.0, gamma=0.1")

        self.eq_listbox.focus_set()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

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
            return
        idx = sel[0]
        key = list(self.equations.keys())[idx]
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
            entry = ttk.Entry(row, textvariable=var, width=12)
            entry.pack(side=tk.LEFT, padx=(pad, 0))
            self._param_vars[pname] = var
            ToolTip(entry, pinfo.get("description", ""))

        ic_text = f"Default ICs: {eq.default_initial_conditions}"
        domain_text = f"Default domain: [{eq.default_domain[0]}, {eq.default_domain[1]}]"
        self.defaults_label.config(text=f"{ic_text}\n{domain_text}")

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

        self.win.destroy()
        from frontend.ui_dialogs.parameters_dialog import ParametersDialog

        ParametersDialog(
            self.parent,
            expression=eq.expression,
            order=eq.order,
            parameters=params,
            equation_name=eq.name,
            default_y0=eq.default_initial_conditions,
            default_domain=eq.default_domain,
        )

    def _on_next_custom(self) -> None:
        expr = self.custom_expr.get("1.0", tk.END).strip()
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
                try:
                    params[name.strip()] = float(val_str.strip())
                except ValueError:
                    messagebox.showerror(
                        "Invalid Parameters",
                        f"Value for '{name.strip()}' is not a number.",
                        parent=self.win,
                    )
                    return

        self.win.destroy()
        from frontend.ui_dialogs.parameters_dialog import ParametersDialog

        ParametersDialog(
            self.parent,
            expression=expr,
            order=order,
            parameters=params,
            equation_name="Custom ODE",
            default_y0=[0.0] * order,
            default_domain=[0.0, 10.0],
        )
