"""Equation selection dialog — choose predefined or write custom ODE."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from config import get_env_from_schema
from frontend.theme import get_font, get_select_colors
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import bind_wraplength, fit_and_center, make_modal
from solver import load_predefined_equations


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
        self._filtered_keys: list[str] = []
        self._selected_category: str | None = None
        self._selected_key: str | None = None
        self._equation_type_var = tk.StringVar(value="ode")

        self._build_ui()

        fit_and_center(self.win, min_width=1200, min_height=650)
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
            btn_inner,
            text="Next \u2192",
            command=self._on_next,
        )
        self._btn_next.pack(side=tk.LEFT, padx=pad)

        btn_cancel = ttk.Button(
            btn_inner,
            text="Cancel",
            style="Cancel.TButton",
            command=self.win.destroy,
        )
        btn_cancel.pack(side=tk.LEFT, padx=pad)

        setup_arrow_enter_navigation([[self._btn_next, btn_cancel]])

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
            text="Vector ODE",
            variable=self._equation_type_var,
            value="vector_ode",
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

        btn_bg: str = get_env_from_schema("UI_BUTTON_BG")
        fg: str = get_env_from_schema("UI_FOREGROUND")
        select_bg, select_fg = get_select_colors(element_bg=btn_bg, text_fg=fg)

        predef_frame.columnconfigure(0, weight=1)
        predef_frame.columnconfigure(1, weight=1)
        predef_frame.rowconfigure(0, weight=3)
        predef_frame.rowconfigure(1, weight=1)

        # Left column: category list (full height, 50% width)
        left = ttk.Frame(predef_frame)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, pad))

        ttk.Label(left, text="Category:", style="Subtitle.TLabel").pack(anchor=tk.W)

        cat_list_frame = ttk.Frame(left)
        cat_list_frame.pack(fill=tk.BOTH, expand=True)

        cat_scrollbar = ttk.Scrollbar(cat_list_frame, orient=tk.VERTICAL)
        self.category_listbox = tk.Listbox(
            cat_list_frame,
            width=18,
            height=20,
            bg=btn_bg,
            fg=fg,
            selectbackground=select_bg,
            selectforeground=select_fg,
            font=get_font(),
            yscrollcommand=cat_scrollbar.set,
            exportselection=False,
        )
        cat_scrollbar.config(command=self.category_listbox.yview)
        self.category_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.category_listbox.bind("<<ListboxSelect>>", self._on_select_category)

        # Right column: equation list (3/4 height) + description (1/4 height), 50% width
        right_container = ttk.Frame(predef_frame)
        right_container.grid(row=0, column=1, rowspan=2, sticky="nsew")
        right_container.columnconfigure(0, weight=1)
        right_container.rowconfigure(0, weight=0)
        right_container.rowconfigure(1, weight=3)
        right_container.rowconfigure(2, weight=1)

        ttk.Label(right_container, text="Equation:", style="Subtitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )

        eq_list_frame = ttk.Frame(right_container)
        eq_list_frame.grid(row=1, column=0, sticky="nsew", pady=(pad, 0))
        eq_list_frame.columnconfigure(0, weight=1)
        eq_list_frame.rowconfigure(0, weight=1)

        eq_scrollbar = ttk.Scrollbar(eq_list_frame, orient=tk.VERTICAL)
        self.eq_listbox = tk.Listbox(
            eq_list_frame,
            width=40,
            height=9,
            bg=btn_bg,
            fg=fg,
            selectbackground=select_bg,
            selectforeground=select_fg,
            font=get_font(),
            yscrollcommand=eq_scrollbar.set,
            exportselection=False,
        )
        eq_scrollbar.config(command=self.eq_listbox.yview)
        self.eq_listbox.grid(row=0, column=0, sticky="nsew")
        eq_scrollbar.grid(row=0, column=1, sticky="ns")

        self.eq_listbox.bind("<<ListboxSelect>>", self._on_select_equation)

        desc_frame = ttk.LabelFrame(right_container, text="Information", padding=pad)
        desc_frame.grid(row=2, column=0, sticky="nsew", pady=(pad, 0))
        desc_frame.columnconfigure(0, weight=1)
        desc_frame.rowconfigure(0, weight=1)

        self.desc_label = ttk.Label(
            desc_frame,
            text="",
            style="Small.TLabel",
            justify=tk.LEFT,
        )
        self.desc_label.pack(anchor=tk.W, fill=tk.BOTH, expand=True)

        bind_wraplength(desc_frame, self.desc_label, pad=2 * pad)

        self._populate_category_list()

        # --- Tab 2: Custom ---
        self._custom_outer = ttk.Frame(self._notebook, padding=pad)
        self._notebook.add(self._custom_outer, text="  Custom  ")

        # The custom content is rebuilt dynamically when equation type changes.
        self._custom_inner: ttk.Frame | None = None
        self._vec_expr_widgets: list[tk.Text] = []

        self.category_listbox.focus_set()
        self._rebuild_custom_tab()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _get_categories_for_type(self) -> list[str]:
        """Return categories for equations matching the current equation type."""
        eq_type = self._equation_type_var.get()
        return sorted(
            {
                eq.category
                for eq in self.equations.values()
                if getattr(eq, "equation_type", "ode") == eq_type
            }
        )

    def _populate_category_list(self) -> None:
        """Populate the category listbox and select first category."""
        categories = self._get_categories_for_type()
        self.category_listbox.delete(0, tk.END)
        for cat in categories:
            self.category_listbox.insert(tk.END, cat)
        self._selected_category = None
        self._filtered_keys = []
        self.eq_listbox.delete(0, tk.END)
        self.desc_label.config(text="")
        if categories:
            self.category_listbox.selection_set(0)
            self._on_select_category(None)

    def _on_select_category(self, _event: tk.Event | None) -> None:  # type: ignore[type-arg]
        """When category changes, populate the equation list."""
        sel = self.category_listbox.curselection()
        if not sel:
            self._selected_category = None
            self._filtered_keys = []
            self.eq_listbox.delete(0, tk.END)
            self.desc_label.config(text="")
            return
        categories = self._get_categories_for_type()
        idx = sel[0]
        self._selected_category = categories[idx]
        eq_type = self._equation_type_var.get()
        self._filtered_keys = [
            k
            for k, eq in self.equations.items()
            if eq.category == self._selected_category
            and getattr(eq, "equation_type", "ode") == eq_type
        ]
        self.eq_listbox.delete(0, tk.END)
        for key in self._filtered_keys:
            self.eq_listbox.insert(tk.END, self.equations[key].name)
        self._selected_key = None
        self.desc_label.config(text="")
        if self._filtered_keys:
            self.eq_listbox.selection_set(0)
            self._on_select_equation(None)

    def _on_type_change(self) -> None:
        """When equation type changes, refresh predefined list and custom tab."""
        self._populate_category_list()
        self._rebuild_custom_tab()

    def _rebuild_custom_tab(self) -> None:
        """Destroy and recreate the custom tab contents for the current equation type."""
        if self._custom_inner is not None:
            self._custom_inner.destroy()

        pad: int = get_env_from_schema("UI_PADDING")
        _btn_bg: str = get_env_from_schema("UI_BUTTON_BG")
        _fg: str = get_env_from_schema("UI_FOREGROUND")
        _font = get_font()

        ci = ttk.Frame(self._custom_outer)
        ci.pack(fill=tk.BOTH, expand=True)
        self._custom_inner = ci
        self._vec_expr_widgets = []

        eq_type = self._equation_type_var.get()

        # -- Hint --
        if eq_type == "difference":
            hint_title = "Write f_{n+order} as a Python expression."
            hint_detail = (
                "Use n for the index, f[0] for f_n, f[1] for f_{n+1}, etc.\n"
                "Example (geometric growth):  r * f[0]"
            )
        elif eq_type == "vector_ode":
            hint_title = "Write the highest derivative for each component."
            hint_detail = (
                "Use f[i,k] where i = component index, k = derivative order.\n"
                "f[i,0] = component i, f[i,1] = its first derivative, etc.\n"
                "Example (coupled oscillators):  -\u03c9**2 * f[0,0] + k * (f[1,0] - f[0,0])"
            )
        elif eq_type == "pde":
            hint_title = "Select the LHS operator and write the RHS expression."
            hint_detail = (
                "Choose which derivative operator equals the expression.\n"
                "Spatial: x, y (or x[0], x[1]). Solution: f. Derivatives: "
                "f[0]=f_x, f[1]=f_y, f[0,0]=f_xx, f[0,1]=f_xy, f[1,1]=f_yy.\n"
                "Example: -\u2207\u00b2f = sin(pi*x)*sin(pi*y) or Helmholtz: expression = f"
            )
        else:
            hint_title = "Write the highest derivative as a Python expression."
            hint_detail = (
                "Use f[0] or f for the function, f[1] for f\u2032, f[2] for f\u2033, etc. "
                "Use x for the independent variable.\n"
                "Example (harmonic oscillator):  -\u03c9**2 * f[0]"
            )

        self.custom_hint_label = ttk.Label(ci, text=hint_title, style="Subtitle.TLabel")
        self.custom_hint_label.pack(anchor=tk.W, pady=(0, pad))

        self.custom_hint_text = tk.StringVar(value=hint_detail)
        self.custom_hint_detail = ttk.Label(
            ci, textvariable=self.custom_hint_text, style="Small.TLabel", justify=tk.LEFT
        )
        self.custom_hint_detail.pack(anchor=tk.W, pady=(0, pad), fill=tk.X)
        bind_wraplength(ci, self.custom_hint_detail, pad=2 * pad)

        # -- Unicode reference --
        unicode_frame = ttk.LabelFrame(
            ci, text="Unicode symbols — copy and paste directly", padding=pad
        )
        unicode_frame.pack(fill=tk.X, pady=(0, pad))
        _unicode_hint = (
            "\u03b1 \u03b2 \u03b3 \u03b4 \u03b5 \u03b6 \u03b7"
            "\u03b8 \u03bb \u03bc \u03be \u03c0 \u03c1 \u03c3"
            "\u03c6 \u03c9 \u0394 \u03a3 \u03a6 \u03a9"
        )
        _font_small = (_font[0], max(9, _font[1] - 4))
        unicode_text = tk.Text(
            unicode_frame,
            height=1,
            bg=_btn_bg,
            fg=_fg,
            font=_font_small,
            borderwidth=0,
            highlightthickness=0,
            wrap="none",
        )
        unicode_text.insert("1.0", _unicode_hint)
        unicode_text.config(state="disabled")
        unicode_text.pack(fill=tk.X)

        # -- Type-specific controls --
        if eq_type == "vector_ode":
            self._build_custom_vector_ode(ci, pad, _btn_bg, _fg, _font)
        elif eq_type == "pde":
            self._build_custom_pde(ci, pad, _btn_bg, _fg, _font)
        else:
            self._build_custom_scalar(ci, pad, _btn_bg, _fg, _font, eq_type)

    def _build_custom_scalar(
        self, ci: ttk.Frame, pad: int, btn_bg: str, fg: str, font: Any, eq_type: str
    ) -> None:
        """Build the custom tab for scalar ODE / difference."""
        row_order = ttk.Frame(ci)
        row_order.pack(fill=tk.X, pady=(pad, pad))
        ttk.Label(row_order, text="Order:").pack(side=tk.LEFT)
        self.custom_order_var = tk.StringVar(value="2")
        ttk.Spinbox(
            row_order,
            from_=1,
            to=10,
            width=5,
            textvariable=self.custom_order_var,
            font=font,
        ).pack(side=tk.LEFT, padx=(pad, 0))

        _primes = ["", "\u2032", "\u2033", "\u2034"]
        _superscript = "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
        _subscript = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"

        def _ode_derivative_label(order_val: int) -> str:
            if order_val < len(_primes):
                prime = _primes[order_val]
                return f"Expression for f{prime}(x) ="
            sup = "".join(_superscript[int(d)] for d in str(order_val))
            return f"Expression for f\u207d{sup}\u207e(x) ="

        def _difference_label(order_val: int) -> str:
            sub = "".join(_subscript[int(d)] for d in str(order_val))
            return f"Expression for f\u2099\u208a{sub} ="  # fₙ₊k

        def _update_expr_label(*_args: str) -> None:
            try:
                order_val = int(self.custom_order_var.get())
            except ValueError:
                order_val = 4
            order_val = max(1, min(10, order_val))
            if eq_type == "difference":
                text = _difference_label(order_val)
            else:
                text = _ode_derivative_label(order_val)
            self._custom_expr_label.config(text=text)

        expr_label_text = (
            _difference_label(2) if eq_type == "difference" else _ode_derivative_label(2)
        )
        self._custom_expr_label = ttk.Label(ci, text=expr_label_text)
        self._custom_expr_label.pack(anchor=tk.W)
        self.custom_order_var.trace_add("write", _update_expr_label)

        self.custom_expr = tk.Text(
            ci,
            height=3,
            width=60,
            bg=btn_bg,
            fg=fg,
            insertbackground=fg,
            font=font,
        )
        self.custom_expr.pack(fill=tk.X, pady=(4, pad))

        ttk.Label(ci, text="Parameter names (comma-separated):").pack(anchor=tk.W)
        self.custom_params = ttk.Entry(ci, width=50, font=font)
        self.custom_params.pack(fill=tk.X, pady=(4, pad))
        ToolTip(self.custom_params, "E.g.: \u03c9, \u03b3")

    def _build_custom_vector_ode(
        self, ci: ttk.Frame, pad: int, btn_bg: str, fg: str, font: Any
    ) -> None:
        """Build the custom tab for vector ODE (per-component expressions)."""
        top_row = ttk.Frame(ci)
        top_row.pack(fill=tk.X, pady=(pad, pad))

        ttk.Label(top_row, text="Components:").pack(side=tk.LEFT)
        self._vec_n_var = tk.StringVar(value="2")
        vec_spin = ttk.Spinbox(
            top_row,
            from_=2,
            to=100,
            width=5,
            textvariable=self._vec_n_var,
            font=font,
        )
        vec_spin.pack(side=tk.LEFT, padx=(pad, pad))
        self._vec_n_refresh_id: str | None = None
        self._vec_n_var.trace_add("write", self._on_vec_n_change)

        # Dummy order var for compatibility (actual orders come from per-component spinboxes)
        self.custom_order_var = tk.StringVar(value="2")

        # Mode: per-component boxes or bulk expression
        mode_frame = ttk.Frame(ci)
        mode_frame.pack(fill=tk.X, pady=(0, pad))
        self._vec_mode_var = tk.StringVar(value="per_component")
        ttk.Radiobutton(
            mode_frame,
            text="Per-component expressions",
            variable=self._vec_mode_var,
            value="per_component",
            command=self._on_vec_mode_change,
        ).pack(side=tk.LEFT, padx=(0, pad))
        ttk.Radiobutton(
            mode_frame,
            text="Bulk expression (use i as loop variable)",
            variable=self._vec_mode_var,
            value="bulk",
            command=self._on_vec_mode_change,
        ).pack(side=tk.LEFT)

        # Container that switches between per-component and bulk
        self._vec_content_frame = ttk.Frame(ci)
        self._vec_content_frame.pack(fill=tk.BOTH, expand=True)

        # Per-component order spinbox variables
        self._vec_order_vars: list[tk.StringVar] = []

        ttk.Label(ci, text="Parameter names (comma-separated):").pack(anchor=tk.W, pady=(pad, 0))
        self.custom_params = ttk.Entry(ci, width=50, font=font)
        self.custom_params.pack(fill=tk.X, pady=(4, pad))
        ToolTip(self.custom_params, "E.g.: \u03c9, k")

        self._refresh_vec_boxes()

    def _on_vec_n_change(self, *args: object) -> None:
        """Update expression boxes when number of components changes (debounced)."""
        if self._vec_n_refresh_id is not None:
            try:
                self.win.after_cancel(self._vec_n_refresh_id)
            except tk.TclError:
                pass
        self._vec_n_refresh_id = self.win.after(150, self._do_vec_n_refresh)

    def _do_vec_n_refresh(self) -> None:
        """Perform the actual refresh (called after debounce delay)."""
        self._vec_n_refresh_id = None
        self._refresh_vec_boxes()

    def _on_vec_mode_change(self) -> None:
        """Switch between per-component and bulk expression modes."""
        self._refresh_vec_boxes()

    def _refresh_vec_boxes(self) -> None:
        """Rebuild the per-component or bulk expression widgets."""
        for w in self._vec_content_frame.winfo_children():
            w.destroy()
        self._vec_expr_widgets = []
        self._vec_order_vars = []
        self._vec_label_refs: list[ttk.Label] = []

        _btn_bg: str = get_env_from_schema("UI_BUTTON_BG")
        _fg: str = get_env_from_schema("UI_FOREGROUND")
        _bg: str = get_env_from_schema("UI_BACKGROUND")
        _font = get_font()
        pad: int = get_env_from_schema("UI_PADDING")

        try:
            n = int(self._vec_n_var.get())
        except ValueError:
            n = 2

        mode = self._vec_mode_var.get()

        if mode == "bulk":
            # Bulk mode: single order spinbox + single expression template
            _primes_bulk = ["", "\u2032", "\u2033", "\u2034"]
            _superscript_bulk = "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"

            def _bulk_label(order_val: int) -> str:
                if order_val < len(_primes_bulk):
                    prime = _primes_bulk[order_val]
                    return f"Expression for f{prime}\u1d62(x) = (component i, use i as variable):"
                sup = "".join(_superscript_bulk[int(d)] for d in str(order_val))
                return (
                    f"Expression for f\u207d{sup}\u207e\u1d62(x) = "
                    "(component i, use i as variable):"
                )

            bulk_top = ttk.Frame(self._vec_content_frame)
            bulk_top.pack(fill=tk.X, pady=(0, pad))
            ttk.Label(bulk_top, text="Order per component:").pack(side=tk.LEFT)
            bulk_order_var = tk.StringVar(value="2")
            self._vec_order_vars.append(bulk_order_var)
            ttk.Spinbox(
                bulk_top,
                from_=1,
                to=10,
                width=5,
                textvariable=bulk_order_var,
                font=_font,
            ).pack(side=tk.LEFT, padx=(pad, 0))

            self._vec_bulk_expr_label = ttk.Label(
                self._vec_content_frame,
                text=_bulk_label(2),
            )
            self._vec_bulk_expr_label.pack(anchor=tk.W)

            def _on_bulk_order_change(*_args: str) -> None:
                try:
                    val = int(bulk_order_var.get())
                except ValueError:
                    val = 2
                val = max(1, min(10, val))
                self._vec_bulk_expr_label.config(text=_bulk_label(val))

            bulk_order_var.trace_add("write", _on_bulk_order_change)
            self._vec_bulk_expr = tk.Text(
                self._vec_content_frame,
                height=3,
                width=60,
                bg=_btn_bg,
                fg=_fg,
                insertbackground=_fg,
                font=_font,
            )
            self._vec_bulk_expr.pack(fill=tk.X, pady=(4, pad))
        else:
            # Per-component: each row has an order spinbox + label + expression box
            _primes = ["", "\u2032", "\u2033", "\u2034"]
            _superscript = "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"

            canvas = tk.Canvas(
                self._vec_content_frame,
                highlightthickness=0,
                height=150,
                bg=_bg,
            )
            scrollbar = ttk.Scrollbar(
                self._vec_content_frame, orient=tk.VERTICAL, command=canvas.yview
            )
            inner = tk.Frame(canvas, bg=_bg)
            canvas.create_window((0, 0), window=inner, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            _sub_digits = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"

            def _sub(idx: int) -> str:
                if 0 <= idx < len(_sub_digits):
                    return _sub_digits[idx]
                return "".join(_sub_digits[int(d)] if d.isdigit() else d for d in str(idx))

            def _make_label_text(comp_idx: int, order_val: int) -> str:
                if order_val < len(_primes):
                    prime = _primes[order_val]
                else:
                    sup = "".join(_superscript[int(d)] for d in str(order_val))
                    prime = f"\u207d{sup}\u207e"
                return f"f{prime}{_sub(comp_idx)} ="

            for i in range(n):
                row = tk.Frame(inner, bg=_bg)
                row.pack(fill=tk.X, pady=2)

                # Order spinbox for this component
                order_var = tk.StringVar(value="2")
                self._vec_order_vars.append(order_var)

                ttk.Label(row, text="ord:", width=4).pack(side=tk.LEFT)
                spin = ttk.Spinbox(
                    row,
                    from_=1,
                    to=10,
                    width=3,
                    textvariable=order_var,
                    font=_font,
                )
                spin.pack(side=tk.LEFT, padx=(0, pad))

                # Label showing f″₀ = etc., updates when order changes
                lbl = ttk.Label(row, text=_make_label_text(i, 2), width=10)
                lbl.pack(side=tk.LEFT)
                self._vec_label_refs.append(lbl)

                # Bind order spinbox change to update label
                def _on_order_change(
                    _var: str, _idx: str, _mode: str, comp=i, ov=order_var, lb=lbl
                ) -> None:
                    try:
                        val = int(ov.get())
                    except ValueError:
                        val = 1
                    lb.config(text=_make_label_text(comp, val))

                order_var.trace_add("write", _on_order_change)

                # Expression text box
                txt = tk.Text(
                    row,
                    height=1,
                    width=45,
                    bg=_btn_bg,
                    fg=_fg,
                    insertbackground=_fg,
                    font=_font,
                )
                txt.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(pad, 0))
                self._vec_expr_widgets.append(txt)

            inner.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))

            def _on_mousewheel(ev: tk.Event) -> str:  # type: ignore[type-arg]
                if canvas.winfo_exists():
                    if hasattr(ev, "delta") and ev.delta != 0:
                        canvas.yview_scroll(int(-1 * (ev.delta / 120)), "units")
                    elif getattr(ev, "num", 0) == 5:
                        canvas.yview_scroll(1, "units")
                    elif getattr(ev, "num", 0) == 4:
                        canvas.yview_scroll(-1, "units")
                return "break"

            def _bind_mousewheel(w: tk.Widget) -> None:
                w.bind("<MouseWheel>", _on_mousewheel)
                w.bind("<Button-4>", _on_mousewheel)
                w.bind("<Button-5>", _on_mousewheel)
                for child in w.winfo_children():
                    _bind_mousewheel(child)

            _bind_mousewheel(canvas)
            _bind_mousewheel(scrollbar)
            _bind_mousewheel(inner)

    def _build_custom_pde(self, ci: ttk.Frame, pad: int, btn_bg: str, fg: str, font: Any) -> None:
        """Build the custom tab for PDE."""
        top_row = ttk.Frame(ci)
        top_row.pack(fill=tk.X, pady=(pad, pad))

        ttk.Label(top_row, text="Independent variables:").pack(side=tk.LEFT)
        self._pde_nvars_var = tk.StringVar(value="2")
        nvars_spin = ttk.Spinbox(
            top_row,
            from_=2,
            to=2,
            width=5,
            textvariable=self._pde_nvars_var,
            font=font,
            state="readonly",
        )
        nvars_spin.pack(side=tk.LEFT, padx=(pad, 0))
        ToolTip(nvars_spin, "Number of independent variables (limited to 2)")
        self._pde_vars_label = ttk.Label(
            top_row,
            text="  x[0], x[1]",
            style="Small.TLabel",
        )
        self._pde_vars_label.pack(side=tk.LEFT, padx=(pad, 0))

        self.custom_order_var = tk.StringVar(value="2")

        # Operator selector (LHS of the PDE)
        op_row = ttk.Frame(ci)
        op_row.pack(fill=tk.X, pady=(0, pad))
        ttk.Label(op_row, text="Left-hand side operator:").pack(side=tk.LEFT)
        self._pde_op_var = tk.StringVar(value="-\u2207\u00b2f (Poisson)")
        _pde_operators = [
            "-\u2207\u00b2f (Poisson)",
            "\u2207\u00b2f (Laplacian)",
            "f\u2080\u2080",  # f_00 (second deriv wrt x[0])
            "f\u2081\u2081",  # f_11 (second deriv wrt x[1])
            "f\u2080\u2081",  # f_01 (mixed partial)
            "f\u2080",  # f_0  (first deriv wrt x[0])
            "f\u2081",  # f_1  (first deriv wrt x[1])
        ]
        ttk.Combobox(
            op_row,
            textvariable=self._pde_op_var,
            values=_pde_operators,
            state="readonly",
            width=22,
            font=font,
        ).pack(side=tk.LEFT, padx=(pad, 0))

        ttk.Label(
            ci,
            text="Right-hand side expression (use x[0], x[1] or x, y for variables):",
        ).pack(anchor=tk.W)
        self.custom_expr = tk.Text(
            ci,
            height=3,
            width=60,
            bg=btn_bg,
            fg=fg,
            insertbackground=fg,
            font=font,
        )
        self.custom_expr.pack(fill=tk.X, pady=(4, pad))

        ttk.Label(ci, text="Parameter names (comma-separated):").pack(anchor=tk.W)
        self.custom_params = ttk.Entry(ci, width=50, font=font)
        self.custom_params.pack(fill=tk.X, pady=(4, pad))
        ToolTip(self.custom_params, "E.g.: k, \u03b1")

    def _on_next(self) -> None:
        """Route to predefined or custom handler based on active tab."""
        idx = self._notebook.index(self._notebook.select())
        if idx == 0:
            self._on_next_predefined()
        else:
            self._on_next_custom()

    def _on_select_equation(self, _event: tk.Event | None) -> None:  # type: ignore[type-arg]
        sel = self.eq_listbox.curselection()
        if not sel:
            self.desc_label.config(text="")
            return
        idx = sel[0]
        key = self._filtered_keys[idx]
        eq = self.equations[key]
        self._selected_key = key

        self.desc_label.config(text=eq.description)

    def _on_next_predefined(self) -> None:
        if self._selected_key is None:
            messagebox.showwarning("No Selection", "Please select an equation.", parent=self.win)
            return

        eq = self.equations[self._selected_key]
        params: dict[str, float] = {
            pname: float(pinfo.get("default", 0.0)) for pname, pinfo in eq.parameters.items()
        }

        self.win.destroy()
        from frontend.ui_dialogs.parameters_dialog import ParametersDialog

        eq_type: str = getattr(eq, "equation_type", "ode")
        variables: list[str] = getattr(eq, "variables", ["x"])
        vector_expressions: list[str] | None = getattr(eq, "vector_expressions", None)
        vector_components: int = getattr(eq, "vector_components", 1)
        ParametersDialog(
            self.parent,
            expression=eq.expression,
            function_name=eq.function_name,
            order=eq.order,
            parameters=params,
            parameters_schema=eq.parameters,
            equation_name=eq.name,
            default_y0=eq.default_initial_conditions,
            default_domain=eq.default_domain,
            display_formula=eq.formula,
            equation_type=eq_type,
            variables=variables,
            vector_expressions=vector_expressions,
            vector_components=vector_components,
        )

    def _parse_custom_params(self) -> dict[str, float | list[float]] | None:
        """Parse custom parameter names from the entry.

        Values default to 0; the user sets them in the next dialog.
        Names of the form ``name[n]`` define a list parameter with *n* components.
        """
        import re as _re

        from utils import normalize_unicode_escapes

        params: dict[str, float | list[float]] = {}
        raw_params = self.custom_params.get().strip()
        if raw_params:
            for name in raw_params.split(","):
                name = name.strip()
                if not name:
                    continue
                normalized_name = normalize_unicode_escapes(name)
                # Detect list parameter pattern: name[n]
                m = _re.match(r"^(.+)\[(\d+)\]$", normalized_name)
                if m:
                    n = int(m.group(2))
                    params[normalized_name] = [0.0] * n
                else:
                    params[normalized_name] = 0.0
        return params

    def _on_next_custom(self) -> None:
        eq_type = self._equation_type_var.get()
        if eq_type == "vector_ode":
            self._on_next_custom_vector()
        elif eq_type == "pde":
            self._on_next_custom_pde()
        else:
            self._on_next_custom_scalar()

    def _on_next_custom_scalar(self) -> None:
        from utils import normalize_unicode_escapes

        expr = normalize_unicode_escapes(self.custom_expr.get("1.0", tk.END).strip())
        if not expr:
            messagebox.showwarning(
                "Empty Expression", "Please enter an expression.", parent=self.win
            )
            return

        try:
            order = int(self.custom_order_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid Order", "Order must be a positive integer.", parent=self.win
            )
            return

        params = self._parse_custom_params()
        if params is None:
            return

        eq_type = self._equation_type_var.get()
        self.win.destroy()
        from frontend.ui_dialogs.parameters_dialog import ParametersDialog

        default_domain: list[float] = [0.0, 50.0] if eq_type == "difference" else [0.0, 10.0]
        ParametersDialog(
            self.parent,
            expression=expr,
            function_name=None,
            order=order,
            parameters=params,
            equation_name="Custom Difference" if eq_type == "difference" else "Custom ODE",
            default_y0=[1.0] * order,
            default_domain=default_domain,
            equation_type=eq_type,
        )

    def _on_next_custom_vector(self) -> None:
        import re

        from utils import normalize_unicode_escapes

        try:
            n_components = int(self._vec_n_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid Input", "Number of components must be an integer.", parent=self.win
            )
            return

        mode = self._vec_mode_var.get()

        # Read per-component orders
        component_orders: list[int] = []
        for idx, ov in enumerate(self._vec_order_vars):
            try:
                component_orders.append(int(ov.get()))
            except ValueError:
                messagebox.showerror(
                    "Invalid Order",
                    f"Order for component {idx} must be a positive integer.",
                    parent=self.win,
                )
                return

        if mode == "bulk":
            # Bulk mode: single order applies to all components
            order = component_orders[0] if component_orders else 2
            component_orders = [order] * n_components

            bulk_expr = normalize_unicode_escapes(self._vec_bulk_expr.get("1.0", tk.END).strip())
            if not bulk_expr:
                messagebox.showwarning(
                    "Empty Expression", "Please enter a bulk expression.", parent=self.win
                )
                return
            # Expand bulk expression for each component index
            vector_expressions = []
            for i in range(n_components):
                expanded = re.sub(r"\bi\b", str(i), bulk_expr)
                vector_expressions.append(expanded)
        else:
            if len(self._vec_expr_widgets) != n_components:
                messagebox.showerror(
                    "Mismatch",
                    "Number of expression boxes doesn't match components. "
                    "Click 'Refresh component boxes'.",
                    parent=self.win,
                )
                return
            vector_expressions = []
            for idx, widget in enumerate(self._vec_expr_widgets):
                expr = normalize_unicode_escapes(widget.get("1.0", tk.END).strip())
                if not expr:
                    messagebox.showwarning(
                        "Empty Expression",
                        f"Expression for component {idx} is empty.",
                        parent=self.win,
                    )
                    return
                vector_expressions.append(expr)

        params = self._parse_custom_params()
        if params is None:
            return

        # Use max order for the pipeline (uniform assumption), pass component_orders for notation
        order = max(component_orders) if component_orders else 2
        all_same = len(set(component_orders)) == 1
        n_state = sum(component_orders)
        default_y0 = [0.0] * n_state

        self.win.destroy()
        from frontend.ui_dialogs.parameters_dialog import ParametersDialog

        ParametersDialog(
            self.parent,
            expression=None,
            function_name=None,
            order=order,
            parameters=params,
            equation_name="Custom Vector ODE",
            default_y0=default_y0,
            default_domain=[0.0, 10.0],
            equation_type="vector_ode",
            vector_expressions=vector_expressions,
            vector_components=n_components,
            component_orders=tuple(component_orders) if not all_same else None,
        )

    def _on_next_custom_pde(self) -> None:
        from utils import normalize_unicode_escapes

        expr = normalize_unicode_escapes(self.custom_expr.get("1.0", tk.END).strip())
        if not expr:
            messagebox.showwarning(
                "Empty Expression", "Please enter a PDE expression.", parent=self.win
            )
            return

        try:
            n_vars = int(self._pde_nvars_var.get())
        except ValueError:
            n_vars = 2
        variables = [f"x[{i}]" for i in range(n_vars)]

        params = self._parse_custom_params()
        if params is None:
            return

        # Map UI operator label to internal operator key
        op_label = self._pde_op_var.get()
        _op_map = {
            "-\u2207\u00b2f (Poisson)": "neg_laplacian",
            "\u2207\u00b2f (Laplacian)": "laplacian",
            "f\u2080\u2080": "fxx",
            "f\u2081\u2081": "fyy",
            "f\u2080\u2081": "fxy",
            "f\u2080": "fx",
            "f\u2081": "fy",
        }
        pde_operator = _op_map.get(op_label, "neg_laplacian")

        self.win.destroy()
        from frontend.ui_dialogs.parameters_dialog import ParametersDialog

        default_domain = [0.0, 1.0, 0.0, 1.0]
        ParametersDialog(
            self.parent,
            expression=expr,
            function_name=None,
            order=2,
            parameters=params,
            equation_name="Custom PDE",
            default_y0=[],
            default_domain=default_domain,
            equation_type="pde",
            variables=variables,
            pde_operator=pde_operator,
        )
