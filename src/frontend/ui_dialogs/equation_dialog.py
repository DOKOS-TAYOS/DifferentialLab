"""Equation selection dialog — choose predefined or write custom ODE."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from config import get_env_from_schema
from frontend.theme import get_font, get_select_colors
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.solve_session import (
    EquationSelection,
    SolveSession,
    choose_filtered_key,
    filter_equation_keys,
    matching_categories,
)
from frontend.ui_dialogs.symbol_palette import SymbolPalette, SymbolTargetTracker
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import (
    bind_wraplength,
    calculate_screen_aware_minsize,
    fit_and_center,
    make_modal,
)
from solver import load_predefined_equations


class EquationDialog:
    """Dialog for selecting or entering an equation.

    Args:
        parent: Parent window.
    """

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        *,
        session: SolveSession | None = None,
    ) -> None:
        self.parent = parent
        self.session = session or SolveSession()
        self.win = tk.Toplevel(parent)
        self.win.title("Choose an Equation")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self.equations = load_predefined_equations()
        self._filtered_keys: list[str] = []
        self._selected_category: str | None = None
        self._selected_key: str | None = None
        self._equation_type_var = tk.StringVar(value=self.session.current_family)
        self._active_family = self.session.current_family
        self._restoring_custom = False

        self._build_ui()

        min_width, min_height = calculate_screen_aware_minsize(
            self.win.winfo_screenwidth(),
            self.win.winfo_screenheight(),
            900,
            650,
        )
        self.win.minsize(min_width, min_height)
        fit_and_center(
            self.win,
            min_width=min_width,
            min_height=min_height,
            resizable=True,
        )
        make_modal(self.win, parent)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        pad: int = get_env_from_schema("UI_PADDING")

        # ── Fixed bottom button bar ──
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=pad, pady=pad)

        self._btn_cancel = ttk.Button(
            btn_frame,
            text="Cancel",
            style="Secondary.TButton",
            command=self.win.destroy,
        )
        self._btn_cancel.pack(side=tk.LEFT)

        self._btn_next = ttk.Button(
            btn_frame,
            text="Continue",
            style="Primary.TButton",
            command=self._on_next,
        )
        self._btn_next.pack(side=tk.RIGHT)

        setup_arrow_enter_navigation([[self._btn_cancel, self._btn_next]])

        # ── Header ──
        header = ttk.Frame(self.win, padding=(pad, pad, pad, 0))
        header.pack(fill=tk.X)
        title_row = ttk.Frame(header)
        title_row.pack(fill=tk.X)
        ttk.Label(title_row, text="Choose equation", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(title_row, text="Step 1 of 3", style="Small.TLabel").pack(side=tk.RIGHT)
        ttk.Label(
            header,
            text="Select a built-in equation or define a custom one.",
            style="Small.TLabel",
        ).pack(anchor=tk.W, pady=(2, pad))

        # ── Equation type selector ──
        type_frame = ttk.Frame(self.win)
        type_frame.pack(fill=tk.X, padx=pad, pady=(pad, 0))
        ttk.Label(type_frame, text="Equation family:", style="Subtitle.TLabel").pack(
            side=tk.LEFT, padx=(0, pad)
        )
        families = (
            ("ODE", "ode"),
            ("Recurrence", "difference"),
            ("Vector ODE", "vector_ode"),
            ("2D PDE", "pde"),
            ("3D PDE", "pde_3d"),
            ("Vector PDE", "vector_pde"),
        )
        for label, value in families:
            ttk.Radiobutton(
                type_frame,
                text=label,
                variable=self._equation_type_var,
                value=value,
                command=self._on_type_change,
            ).pack(side=tk.LEFT, padx=(0, 2 * pad))

        # ── Notebook ──
        self._notebook = ttk.Notebook(self.win)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=pad, pady=pad)
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # --- Tab 1: Predefined ---
        predef_frame = ttk.Frame(self._notebook, padding=pad)
        self._notebook.add(predef_frame, text="  Built-in  ")

        btn_bg: str = get_env_from_schema("UI_BUTTON_BG")
        fg: str = get_env_from_schema("UI_FOREGROUND")
        select_bg, select_fg = get_select_colors(element_bg=btn_bg, text_fg=fg)

        predef_frame.columnconfigure(0, weight=1)
        predef_frame.rowconfigure(2, weight=1)

        ttk.Label(predef_frame, text="Search", style="Subtitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self._search_var = tk.StringVar(value=self.session.family_state().search_query)
        self.search_entry = ttk.Entry(
            predef_frame,
            textvariable=self._search_var,
            font=get_font(),
        )
        self.search_entry.grid(row=1, column=0, sticky="ew", pady=(4, pad))
        self._search_var.trace_add("write", self._on_search_change)

        panes = ttk.PanedWindow(predef_frame, orient=tk.HORIZONTAL)
        panes.grid(row=2, column=0, sticky="nsew")

        left = ttk.Frame(panes, padding=(0, 0, pad, 0))
        middle = ttk.Frame(panes, padding=(pad, 0))
        details = ttk.LabelFrame(panes, text="Details", padding=pad)
        panes.add(left, weight=1)
        panes.add(middle, weight=2)
        panes.add(details, weight=3)

        ttk.Label(left, text="Categories", style="Subtitle.TLabel").pack(anchor=tk.W)

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

        ttk.Label(middle, text="Equations", style="Subtitle.TLabel").pack(anchor=tk.W)

        eq_list_frame = ttk.Frame(middle)
        eq_list_frame.pack(fill=tk.BOTH, expand=True)

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
        self.eq_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        eq_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.eq_listbox.bind("<<ListboxSelect>>", self._on_select_equation)
        self.eq_listbox.bind("<Return>", lambda _event: self._on_next())

        self.desc_label = ttk.Label(
            details,
            text="",
            style="Small.TLabel",
            justify=tk.LEFT,
            anchor=tk.NW,
        )
        self.desc_label.pack(anchor=tk.W, fill=tk.BOTH, expand=True)

        bind_wraplength(details, self.desc_label, pad=2 * pad)

        # --- Tab 2: Custom ---
        self._custom_scroll = ScrollableFrame(self._notebook, padding=pad)
        self._custom_scroll.apply_bg(bg)
        self._custom_outer = self._custom_scroll.inner
        self._notebook.add(self._custom_scroll, text="  Custom  ")

        # The custom content is rebuilt dynamically when equation type changes.
        self._custom_inner: ttk.Frame | None = None
        self._vec_expr_widgets: list[tk.Text] = []
        self._symbol_tracker = SymbolTargetTracker()

        family_state = self.session.family_state()
        self._notebook.select(0 if family_state.mode == "built_in" else 1)
        self._populate_category_list()
        self.category_listbox.focus_set()
        self._rebuild_custom_tab()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _get_categories_for_type(self) -> list[str]:
        """Return categories for equations matching the current equation type."""
        return matching_categories(
            self.equations,
            self._equation_type_var.get(),
            self._search_var.get(),
        )

    def _populate_category_list(self) -> None:
        """Populate non-empty categories and restore a valid family selection."""
        categories = self._get_categories_for_type()
        self._visible_categories = categories
        self.category_listbox.delete(0, tk.END)
        for cat in categories:
            self.category_listbox.insert(tk.END, cat)
        family_state = self.session.family_state()
        selected_category = family_state.category
        if selected_category not in categories:
            selected_category = categories[0] if categories else None
        self._selected_category = selected_category
        if selected_category is None:
            self._filtered_keys = []
            self.eq_listbox.delete(0, tk.END)
            self._selected_key = None
            self.desc_label.config(text="No built-in equations match this search.")
            return
        category_index = categories.index(selected_category)
        self.category_listbox.selection_set(category_index)
        self.category_listbox.see(category_index)
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
        idx = sel[0]
        if idx >= len(self._visible_categories):
            return
        self._selected_category = self._visible_categories[idx]
        eq_type = self._equation_type_var.get()
        family_state = self.session.family_state()
        if not self._search_var.get().strip():
            family_state.category = self._selected_category
        self._filtered_keys = filter_equation_keys(
            self.equations,
            eq_type,
            self._search_var.get(),
            self._selected_category,
        )
        self.eq_listbox.delete(0, tk.END)
        for key in self._filtered_keys:
            self.eq_listbox.insert(tk.END, self.equations[key].name)
        preferred_key = choose_filtered_key(self._filtered_keys, family_state.equation_key)
        self._selected_key = preferred_key
        if preferred_key is None:
            self.desc_label.config(text="No built-in equations match this search.")
            return
        equation_index = self._filtered_keys.index(preferred_key)
        self.eq_listbox.selection_set(equation_index)
        self.eq_listbox.see(equation_index)
        self._on_select_equation(None)

    def _on_search_change(self, *_args: object) -> None:
        """Apply a local, case-insensitive catalog filter for the active family."""
        if not hasattr(self, "category_listbox"):
            return
        self.session.family_state().search_query = self._search_var.get()
        self._populate_category_list()

    def _on_tab_changed(self, _event: tk.Event | None) -> None:  # type: ignore[type-arg]
        """Remember the Built-in/Custom mode independently for each family."""
        if not hasattr(self, "_notebook"):
            return
        selected_tab = self._notebook.select()
        if not selected_tab:
            return
        index = self._notebook.index(selected_tab)
        self.session.family_state().mode = "built_in" if index == 0 else "custom"

    def _on_type_change(self) -> None:
        """When equation type changes, refresh predefined list and custom tab."""
        previous_family = self._active_family
        self._capture_custom_state(previous_family)
        previous_state = self.session.family_state(previous_family)
        if not self._search_var.get().strip():
            previous_state.category = self._selected_category
            previous_state.equation_key = self._selected_key

        self._active_family = self._equation_type_var.get()
        self.session.current_family = self._active_family
        family_state = self.session.family_state()
        self._search_var.set(family_state.search_query)
        self._notebook.select(0 if family_state.mode == "built_in" else 1)
        self._populate_category_list()
        self._rebuild_custom_tab()

    def _rebuild_custom_tab(self) -> None:
        """Destroy and recreate the custom tab contents for the current equation type."""
        if self._custom_inner is not None:
            self._custom_inner.destroy()

        pad: int = get_env_from_schema("UI_PADDING")
        button_bg: str = get_env_from_schema("UI_BUTTON_BG")
        foreground: str = get_env_from_schema("UI_FOREGROUND")
        font = get_font()

        ci = ttk.Frame(self._custom_outer)
        ci.pack(fill=tk.BOTH, expand=True)
        self._custom_inner = ci
        self._vec_expr_widgets = []
        eq_type = self._equation_type_var.get()

        editor_panel = ttk.LabelFrame(ci, text="Equation definition", padding=pad)
        reference_panel = ttk.LabelFrame(ci, text="Reference / Symbols", padding=pad)
        self._custom_editor_panel = editor_panel
        self._custom_reference_panel = reference_panel
        self._custom_layout_mode = ""

        if eq_type == "vector_ode":
            self._build_custom_vector_ode(editor_panel, pad, button_bg, foreground, font)
        elif eq_type == "vector_pde":
            self._build_custom_vector_pde(editor_panel, pad, button_bg, foreground, font)
        elif eq_type == "pde_3d":
            self._build_custom_pde_3d(editor_panel, pad, button_bg, foreground, font)
        elif eq_type == "pde":
            self._build_custom_pde(editor_panel, pad, button_bg, foreground, font)
        else:
            self._build_custom_scalar(editor_panel, pad, button_bg, foreground, font, eq_type)

        self._build_custom_reference(reference_panel, eq_type, pad)
        ci.bind("<Configure>", self._on_custom_workspace_configure, add="+")
        self._layout_custom_workspace(1200)
        self._apply_custom_state(eq_type)
        self._refresh_custom_scroll()

    def _on_custom_workspace_configure(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Switch the Custom editor between side-by-side and stacked layouts."""
        self._layout_custom_workspace(event.width)

    def _refresh_custom_scroll(self) -> None:
        """Refresh scrolling after dynamic Custom-editor widgets are rebuilt."""
        self._custom_scroll.bind_new_children()
        self._custom_scroll.refresh_scroll_region()

    def _layout_custom_workspace(self, width: int) -> None:
        """Place editor and reference panels without requiring horizontal scrolling."""
        mode = "wide" if width >= 700 else "stacked"
        if mode == self._custom_layout_mode:
            return
        self._custom_layout_mode = mode
        pad: int = get_env_from_schema("UI_PADDING")
        editor = self._custom_editor_panel
        reference = self._custom_reference_panel
        container = self._custom_inner
        if container is None:
            return
        editor.grid_forget()
        reference.grid_forget()
        for column in (0, 1):
            container.columnconfigure(column, weight=0)
        for row in (0, 1):
            container.rowconfigure(row, weight=0)
        if mode == "wide":
            editor.grid(row=0, column=0, sticky="nsew", padx=(0, pad // 2))
            reference.grid(row=0, column=1, sticky="nsew", padx=(pad // 2, 0))
            container.columnconfigure(0, weight=3)
            container.columnconfigure(1, weight=2)
            container.rowconfigure(0, weight=1)
        else:
            editor.grid(row=0, column=0, sticky="nsew", pady=(0, pad // 2))
            reference.grid(row=1, column=0, sticky="nsew", pady=(pad // 2, 0))
            container.columnconfigure(0, weight=1)

    def _build_custom_reference(self, parent: ttk.LabelFrame, family: str, pad: int) -> None:
        """Build concise, family-specific syntax guidance and the shared palette."""
        sections = self._custom_reference_sections(family)
        for title, detail in sections:
            ttk.Label(parent, text=title, style="Subtitle.TLabel").pack(anchor=tk.W)
            label = ttk.Label(parent, text=detail, style="Small.TLabel", justify=tk.LEFT)
            label.pack(anchor=tk.W, fill=tk.X, pady=(1, pad // 2))
            bind_wraplength(parent, label, pad=2 * pad, min_wrap=180)
        self._symbol_palette = SymbolPalette(
            parent,
            tracker=self._symbol_tracker,
            columns=7,
        )
        self._symbol_palette.pack(fill=tk.X, pady=(pad // 2, 0))

    @staticmethod
    def _custom_reference_sections(family: str) -> tuple[tuple[str, str], ...]:
        """Return scannable syntax lines for one Custom equation family."""
        if family == "difference":
            return (
                ("Independent index", "n"),
                ("State notation", "f[0] = fₙ, f[1] = fₙ₊₁, and so on."),
                ("Example", "r * f[0]"),
            )
        if family == "vector_ode":
            return (
                ("Independent variable", "x"),
                ("State notation", "f[i,k]: component i, derivative order k."),
                ("Example", "-ω**2 * f[0,0] + k * (f[1,0] - f[0,0])"),
            )
        if family == "pde":
            return (
                ("Variables", "x, y (also accepted as x[0], x[1])"),
                ("Derivatives", "f[0], f[1], f[0,0], f[0,1], f[1,1]"),
                ("Example", "sin(pi*x) * sin(pi*y)"),
            )
        if family == "pde_3d":
            return (
                ("Variables", "x, y, z"),
                ("State notation", "f; first derivatives fx, fy, fz"),
                ("Derivatives", "fxx, fxy, fxz, fyy, fyz, fzz"),
                ("Example", "-fxx - fyy - fzz - 3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)"),
            )
        if family == "vector_pde":
            return (
                ("Variables", "x, y"),
                ("State notation", "f[i], fx[i], fy[i] for component i."),
                ("Derivatives", "fxx[i], fxy[i], fyy[i]"),
                ("Example", "-fxx[0] - fyy[0]"),
            )
        return (
            ("Independent variable", "x"),
            ("State notation", "f or f[0]; derivatives f[1], f[2], and so on."),
            ("Example", "-ω**2 * f[0]"),
        )

    def _register_symbol_target(self, target: tk.Text | ttk.Entry) -> None:
        """Register a static or freshly rebuilt mathematical editor."""
        self._symbol_tracker.register_target(target)

    @staticmethod
    def _text_value(widget: tk.Text | None) -> str:
        """Read a Text widget as plain user data."""
        if widget is None:
            return ""
        return widget.get("1.0", tk.END).rstrip("\n")

    @staticmethod
    def _set_text(widget: tk.Text | None, value: str) -> None:
        """Replace a Text widget's content with a plain string."""
        if widget is None:
            return
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)

    def _capture_custom_state(self, family: str) -> None:
        """Capture the active family's custom editor into pure session data."""
        if not hasattr(self, "custom_params"):
            return
        draft = self.session.family_state(family).custom_draft
        draft["order"] = self.custom_order_var.get()
        draft["parameters"] = self.custom_params.get()
        if family == "vector_ode":
            draft["components"] = self._vec_n_var.get()
            draft["mode"] = self._vec_mode_var.get()
            self._capture_vector_widget_state(draft)
        elif family == "vector_pde":
            draft["components"] = self._vec_n_var.get()
            draft["expressions"] = [self._text_value(widget) for widget in self._vec_expr_widgets]
        else:
            draft["expression"] = self._text_value(getattr(self, "custom_expr", None))
            if family == "pde":
                draft["operator"] = self._pde_op_var.get()
                draft["variables"] = self._pde_nvars_var.get()

    def _capture_vector_widget_state(self, draft: dict[str, Any]) -> None:
        """Retain the current Vector ODE mode's orders and expressions."""
        mode = getattr(self, "_active_vec_mode", self._vec_mode_var.get())
        if mode == "bulk":
            draft["bulk_order"] = self._vec_order_vars[0].get() if self._vec_order_vars else "2"
            draft["bulk_expression"] = self._text_value(getattr(self, "_vec_bulk_expr", None))
        else:
            draft["component_orders"] = [variable.get() for variable in self._vec_order_vars]
            draft["component_expressions"] = [
                self._text_value(widget) for widget in self._vec_expr_widgets
            ]

    def _apply_custom_state(self, family: str) -> None:
        """Restore the active family's custom editor from pure session data."""
        draft = self.session.family_state(family).custom_draft
        if not draft:
            return
        self._restoring_custom = True
        try:
            self.custom_order_var.set(str(draft.get("order", self.custom_order_var.get())))
            self.custom_params.delete(0, tk.END)
            self.custom_params.insert(0, str(draft.get("parameters", "")))
            if family == "vector_ode":
                self._vec_n_var.set(str(draft.get("components", "2")))
                mode = str(draft.get("mode", "per_component"))
                self._vec_mode_var.set(mode)
                self._active_vec_mode = mode
                self._refresh_vec_boxes()
                self._restore_vector_widget_state(draft, mode)
            elif family == "vector_pde":
                self._vec_n_var.set(str(draft.get("components", "2")))
                self._refresh_vector_pde_boxes()
                for widget, value in zip(
                    self._vec_expr_widgets,
                    draft.get("expressions", []),
                    strict=False,
                ):
                    self._set_text(widget, str(value))
            else:
                self._set_text(
                    getattr(self, "custom_expr", None),
                    str(draft.get("expression", "")),
                )
                if family == "pde":
                    self._pde_op_var.set(str(draft.get("operator", self._pde_op_var.get())))
                    self._pde_nvars_var.set(str(draft.get("variables", self._pde_nvars_var.get())))
        finally:
            self._restoring_custom = False

    def _restore_vector_widget_state(self, draft: dict[str, Any], mode: str) -> None:
        """Restore values for the currently visible Vector ODE editor mode."""
        if mode == "bulk":
            if self._vec_order_vars:
                self._vec_order_vars[0].set(str(draft.get("bulk_order", "2")))
            self._set_text(
                getattr(self, "_vec_bulk_expr", None),
                str(draft.get("bulk_expression", "")),
            )
            return
        for variable, value in zip(
            self._vec_order_vars,
            draft.get("component_orders", []),
            strict=False,
        ):
            variable.set(str(value))
        for widget, value in zip(
            self._vec_expr_widgets,
            draft.get("component_expressions", []),
            strict=False,
        ):
            self._set_text(widget, str(value))

    def _build_custom_scalar(
        self, ci: tk.Misc, pad: int, btn_bg: str, fg: str, font: Any, eq_type: str
    ) -> None:
        """Build the custom tab for scalar ODE / difference."""
        row_order = ttk.Frame(ci)
        row_order.pack(fill=tk.X, pady=(0, pad))
        ttk.Label(row_order, text="Order").pack(side=tk.LEFT)
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
                return f"f{prime}(x) ="
            sup = "".join(_superscript[int(d)] for d in str(order_val))
            return f"f\u207d{sup}\u207e(x) ="

        def _difference_label(order_val: int) -> str:
            sub = "".join(_subscript[int(d)] for d in str(order_val))
            return f"f\u2099\u208a{sub} ="  # fₙ₊k

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
        ttk.Label(
            ci,
            text="Next term" if eq_type == "difference" else "Highest derivative",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W)
        self._custom_expr_label = ttk.Label(ci, text=expr_label_text)
        self._custom_expr_label.pack(anchor=tk.W, pady=(2, 0))
        self.custom_order_var.trace_add("write", _update_expr_label)

        self.custom_expr = tk.Text(
            ci,
            height=3,
            width=48,
            bg=btn_bg,
            fg=fg,
            insertbackground=fg,
            font=font,
        )
        self.custom_expr.pack(fill=tk.X, pady=(4, pad))
        self._register_symbol_target(self.custom_expr)

        ttk.Label(ci, text="Parameters", style="Subtitle.TLabel").pack(anchor=tk.W)
        self.custom_params = ttk.Entry(ci, width=50, font=font)
        self.custom_params.pack(fill=tk.X, pady=(4, pad))
        self._register_symbol_target(self.custom_params)
        ToolTip(
            self.custom_params,
            "Comma-separated names to configure next, for example omega, gamma. "
            "Use name[n] for an n-value parameter.",
        )

    def _build_custom_vector_ode(
        self, ci: tk.Misc, pad: int, btn_bg: str, fg: str, font: Any
    ) -> None:
        """Build the custom tab for vector ODE (per-component expressions)."""
        top_row = ttk.Frame(ci)
        top_row.pack(fill=tk.X, pady=(0, pad))

        ttk.Label(top_row, text="Components").pack(side=tk.LEFT)
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
        ttk.Label(ci, text="Definition mode", style="Subtitle.TLabel").pack(anchor=tk.W)
        mode_frame = ttk.Frame(ci)
        mode_frame.pack(fill=tk.X, pady=(0, pad))
        self._vec_mode_var = tk.StringVar(value="per_component")
        self._active_vec_mode = "per_component"
        ttk.Radiobutton(
            mode_frame,
            text="Per component",
            variable=self._vec_mode_var,
            value="per_component",
            command=self._on_vec_mode_change,
        ).pack(side=tk.LEFT, padx=(0, pad))
        ttk.Radiobutton(
            mode_frame,
            text="Bulk",
            variable=self._vec_mode_var,
            value="bulk",
            command=self._on_vec_mode_change,
        ).pack(side=tk.LEFT)

        # Container that switches between per-component and bulk
        self._vec_content_frame = ttk.Frame(ci)
        self._vec_content_frame.pack(fill=tk.BOTH, expand=True)

        # Per-component order spinbox variables
        self._vec_order_vars: list[tk.StringVar] = []

        ttk.Label(ci, text="Parameters", style="Subtitle.TLabel").pack(anchor=tk.W, pady=(pad, 0))
        self.custom_params = ttk.Entry(ci, width=50, font=font)
        self.custom_params.pack(fill=tk.X, pady=(4, pad))
        self._register_symbol_target(self.custom_params)
        ToolTip(
            self.custom_params,
            "Comma-separated names to configure next, for example omega, k. "
            "Use name[n] for an n-value parameter.",
        )

        self._refresh_vec_boxes()

    def _on_vec_n_change(self, *args: object) -> None:
        """Update expression boxes when number of components changes (debounced)."""
        if self._restoring_custom:
            return
        self._capture_custom_state(self._active_family)
        if self._vec_n_refresh_id is not None:
            try:
                self.win.after_cancel(self._vec_n_refresh_id)
            except tk.TclError:
                pass
        self._vec_n_refresh_id = self.win.after(150, self._do_vec_n_refresh)

    def _do_vec_n_refresh(self) -> None:
        """Perform the actual refresh (called after debounce delay)."""
        self._vec_n_refresh_id = None
        draft = self.session.family_state().custom_draft
        if self._equation_type_var.get() == "vector_pde":
            self._refresh_vector_pde_boxes()
            for widget, value in zip(
                self._vec_expr_widgets,
                draft.get("expressions", []),
                strict=False,
            ):
                self._set_text(widget, str(value))
        else:
            self._refresh_vec_boxes()
            self._restore_vector_widget_state(draft, self._vec_mode_var.get())

    def _on_vec_mode_change(self) -> None:
        """Switch between per-component and bulk expression modes."""
        draft = self.session.family_state().custom_draft
        self._capture_vector_widget_state(draft)
        self._active_vec_mode = self._vec_mode_var.get()
        draft["mode"] = self._active_vec_mode
        self._refresh_vec_boxes()
        self._restore_vector_widget_state(draft, self._active_vec_mode)

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
                    return f"f{prime}\u1d62(x) ="
                sup = "".join(_superscript_bulk[int(d)] for d in str(order_val))
                return f"f\u207d{sup}\u207e\u1d62(x) ="

            bulk_top = ttk.Frame(self._vec_content_frame)
            bulk_top.pack(fill=tk.X, pady=(0, pad))
            ttk.Label(bulk_top, text="Order per component").pack(side=tk.LEFT)
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

            ttk.Label(
                self._vec_content_frame,
                text="Expression template for component i",
                style="Subtitle.TLabel",
            ).pack(anchor=tk.W)
            ttk.Label(
                self._vec_content_frame,
                text="Use i as the component index.",
                style="Small.TLabel",
            ).pack(anchor=tk.W, pady=(0, 2))
            self._vec_bulk_expr_label = ttk.Label(self._vec_content_frame, text=_bulk_label(2))
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
                width=44,
                bg=_btn_bg,
                fg=_fg,
                insertbackground=_fg,
                font=_font,
            )
            self._vec_bulk_expr.pack(fill=tk.X, pady=(4, pad))
            self._register_symbol_target(self._vec_bulk_expr)
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
            inner = ttk.Frame(canvas)
            canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            ttk.Label(inner, text="Component", style="Subtitle.TLabel").grid(
                row=0, column=0, sticky="w", padx=(0, pad)
            )
            ttk.Label(inner, text="Order", style="Subtitle.TLabel").grid(
                row=0, column=1, sticky="w", padx=(0, pad)
            )
            ttk.Label(inner, text="Highest derivative expression", style="Subtitle.TLabel").grid(
                row=0, column=2, columnspan=2, sticky="w"
            )
            inner.columnconfigure(3, weight=1)

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
                row_number = i + 1

                # Order spinbox for this component
                order_var = tk.StringVar(value="2")
                self._vec_order_vars.append(order_var)

                ttk.Label(inner, text=f"f{_sub(i)}", width=8).grid(
                    row=row_number, column=0, sticky="w", padx=(0, pad), pady=2
                )
                spin = ttk.Spinbox(
                    inner,
                    from_=1,
                    to=10,
                    width=3,
                    textvariable=order_var,
                    font=_font,
                )
                spin.grid(row=row_number, column=1, sticky="w", padx=(0, pad), pady=2)

                # Label showing f″₀ = etc., updates when order changes
                lbl = ttk.Label(inner, text=_make_label_text(i, 2), width=9)
                lbl.grid(row=row_number, column=2, sticky="w", padx=(0, pad), pady=2)
                self._vec_label_refs.append(lbl)

                # Bind order spinbox change to update label
                def _on_order_change(
                    _var: str,
                    _idx: str,
                    _mode: str,
                    comp: int = i,
                    ov: tk.StringVar = order_var,
                    lb: ttk.Label = lbl,
                ) -> None:
                    try:
                        val = int(ov.get())
                    except ValueError:
                        val = 1
                    lb.config(text=_make_label_text(comp, val))

                order_var.trace_add("write", _on_order_change)

                # Expression text box
                txt = tk.Text(
                    inner,
                    height=1,
                    width=28,
                    bg=_btn_bg,
                    fg=_fg,
                    insertbackground=_fg,
                    font=_font,
                )
                txt.grid(row=row_number, column=3, sticky="ew", pady=2)
                self._vec_expr_widgets.append(txt)
                self._register_symbol_target(txt)

            inner.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.bind(
                "<Configure>",
                lambda event: canvas.itemconfigure(canvas_window, width=event.width),
                add="+",
            )

            def _on_mousewheel(ev: tk.Event) -> str:  # type: ignore[type-arg]
                if canvas.winfo_exists():
                    if hasattr(ev, "delta") and ev.delta != 0:
                        canvas.yview_scroll(int(-1 * (ev.delta / 120)), "units")
                    elif getattr(ev, "num", 0) == 5:
                        canvas.yview_scroll(1, "units")
                    elif getattr(ev, "num", 0) == 4:
                        canvas.yview_scroll(-1, "units")
                return "break"

            def _bind_mousewheel(w: tk.Misc) -> None:
                w.bind("<MouseWheel>", _on_mousewheel)
                w.bind("<Button-4>", _on_mousewheel)
                w.bind("<Button-5>", _on_mousewheel)
                for child in w.winfo_children():
                    _bind_mousewheel(child)

            _bind_mousewheel(canvas)
            _bind_mousewheel(scrollbar)
            _bind_mousewheel(inner)

        self._refresh_custom_scroll()

    def _build_custom_pde(self, ci: tk.Misc, pad: int, btn_bg: str, fg: str, font: Any) -> None:
        """Build the custom tab for PDE."""
        self._pde_nvars_var = tk.StringVar(value="2")
        self._pde_vars_label = ttk.Label(ci, text="Variables: x, y", style="Subtitle.TLabel")
        self._pde_vars_label.pack(anchor=tk.W, pady=(0, pad))

        self.custom_order_var = tk.StringVar(value="2")

        # Operator selector (LHS of the PDE)
        ttk.Label(ci, text="Left-hand operator", style="Subtitle.TLabel").pack(anchor=tk.W)
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
        operator_combo = ttk.Combobox(
            ci,
            textvariable=self._pde_op_var,
            values=_pde_operators,
            state="readonly",
            width=22,
            font=font,
        )
        operator_combo.pack(fill=tk.X, pady=(4, pad))

        ttk.Label(ci, text="Right-hand expression", style="Subtitle.TLabel").pack(anchor=tk.W)
        self.custom_expr = tk.Text(
            ci,
            height=3,
            width=48,
            bg=btn_bg,
            fg=fg,
            insertbackground=fg,
            font=font,
        )
        self.custom_expr.pack(fill=tk.X, pady=(4, pad))
        self._register_symbol_target(self.custom_expr)

        ttk.Label(ci, text="Parameters", style="Subtitle.TLabel").pack(anchor=tk.W)
        self.custom_params = ttk.Entry(ci, width=50, font=font)
        self.custom_params.pack(fill=tk.X, pady=(4, pad))
        self._register_symbol_target(self.custom_params)
        ToolTip(
            self.custom_params,
            "Comma-separated names to configure next, for example k, alpha. "
            "Use name[n] for an n-value parameter.",
        )

    def _build_custom_pde_3d(
        self,
        ci: tk.Misc,
        pad: int,
        btn_bg: str,
        fg: str,
        font: Any,
    ) -> None:
        """Build the residual-form custom editor for scalar PDE 3D."""
        self.custom_order_var = tk.StringVar(value="2")
        ttk.Label(ci, text="Variables: x, y, z", style="Subtitle.TLabel").pack(
            anchor=tk.W, pady=(0, pad)
        )
        ttk.Label(ci, text="Residual expression = 0", style="Subtitle.TLabel").pack(anchor=tk.W)
        self.custom_expr = tk.Text(
            ci,
            height=4,
            width=48,
            bg=btn_bg,
            fg=fg,
            insertbackground=fg,
            font=font,
        )
        self.custom_expr.pack(fill=tk.X, pady=(4, pad))
        self._register_symbol_target(self.custom_expr)
        ttk.Label(ci, text="Parameters", style="Subtitle.TLabel").pack(anchor=tk.W)
        self.custom_params = ttk.Entry(ci, width=50, font=font)
        self.custom_params.pack(fill=tk.X, pady=(4, pad))
        self._register_symbol_target(self.custom_params)
        ToolTip(
            self.custom_params,
            "Comma-separated names to configure next. Use name[n] for an n-value parameter.",
        )

    def _build_custom_vector_pde(
        self,
        ci: tk.Misc,
        pad: int,
        btn_bg: str,
        fg: str,
        font: Any,
    ) -> None:
        """Build the bounded per-equation editor for a custom Vector PDE."""
        top_row = ttk.Frame(ci)
        top_row.pack(fill=tk.X, pady=(0, pad))
        ttk.Label(top_row, text="Components").pack(side=tk.LEFT)
        self._vec_n_var = tk.StringVar(value="2")
        component_spin = ttk.Spinbox(
            top_row,
            from_=1,
            to=4,
            width=5,
            textvariable=self._vec_n_var,
            font=font,
        )
        component_spin.pack(side=tk.LEFT, padx=(pad, 0))
        ToolTip(component_spin, "Vector PDE is limited to 4 components in the standard UI")
        self._vec_n_refresh_id = None
        self._vec_n_var.trace_add("write", self._on_vec_n_change)
        self.custom_order_var = tk.StringVar(value="2")

        self._vec_content_frame = ttk.Frame(ci)
        self._vec_content_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(ci, text="Parameters", style="Subtitle.TLabel").pack(anchor=tk.W, pady=(pad, 0))
        self.custom_params = ttk.Entry(ci, width=50, font=font)
        self.custom_params.pack(fill=tk.X, pady=(4, pad))
        self._register_symbol_target(self.custom_params)
        ToolTip(
            self.custom_params,
            "Comma-separated names to configure next. Use name[n] for an n-value parameter.",
        )
        self._refresh_vector_pde_boxes()

    def _refresh_vector_pde_boxes(self) -> None:
        """Rebuild the small list of Vector PDE residual expression boxes."""
        for widget in self._vec_content_frame.winfo_children():
            widget.destroy()
        self._vec_expr_widgets = []
        try:
            components = int(self._vec_n_var.get())
        except ValueError:
            components = 2
        components = max(1, min(4, components))
        button_bg: str = get_env_from_schema("UI_BUTTON_BG")
        foreground: str = get_env_from_schema("UI_FOREGROUND")
        font = get_font()
        pad: int = get_env_from_schema("UI_PADDING")
        ttk.Label(self._vec_content_frame, text="Component", style="Subtitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, pad)
        )
        ttk.Label(
            self._vec_content_frame,
            text="Residual expression = 0",
            style="Subtitle.TLabel",
        ).grid(row=0, column=1, sticky="w")
        self._vec_content_frame.columnconfigure(1, weight=1)
        subscript = "\u2080\u2081\u2082\u2083"
        for component in range(components):
            ttk.Label(self._vec_content_frame, text=f"f{subscript[component]}", width=10).grid(
                row=component + 1, column=0, sticky="w", padx=(0, pad), pady=2
            )
            expression = tk.Text(
                self._vec_content_frame,
                height=2,
                width=38,
                bg=button_bg,
                fg=foreground,
                insertbackground=foreground,
                font=font,
            )
            expression.grid(row=component + 1, column=1, sticky="ew", pady=2)
            self._vec_expr_widgets.append(expression)
            self._register_symbol_target(expression)
        self._refresh_custom_scroll()

    def _on_next(self) -> None:
        """Route to predefined or custom handler based on active tab."""
        self._capture_custom_state(self._active_family)
        self.session.family_state().mode = (
            "built_in" if self._notebook.index(self._notebook.select()) == 0 else "custom"
        )
        idx = self._notebook.index(self._notebook.select())
        if idx == 0:
            self._on_next_predefined()
        else:
            self._on_next_custom()

    def _on_select_equation(self, _event: tk.Event | None) -> None:  # type: ignore[type-arg]
        sel = self.eq_listbox.curselection()
        if not sel:
            self.desc_label.config(text="No equation selected.")
            return
        idx = sel[0]
        if idx >= len(self._filtered_keys):
            return
        key = self._filtered_keys[idx]
        eq = self.equations[key]
        self._selected_key = key
        state = self.session.family_state()
        if not self._search_var.get().strip():
            state.equation_key = key
            state.category = eq.category

        family_names = {
            "ode": "ODE",
            "difference": "Recurrence",
            "vector_ode": "Vector ODE",
            "pde": "2D PDE",
            "pde_3d": "3D PDE",
            "vector_pde": "Vector PDE",
        }
        parameters = ", ".join(eq.parameters) if eq.parameters else "None"
        details = (
            f"{eq.name}\n\n"
            f"{eq.description}\n\n"
            f"Formula\n{eq.formula or 'Not provided'}\n\n"
            f"Family\n{family_names.get(getattr(eq, 'equation_type', 'ode'), 'ODE')}\n\n"
            f"Parameters\n{parameters}"
        )
        self.desc_label.config(text=details)

    def _open_configuration(self, selection: EquationSelection) -> None:
        """Store a selection, close Equation, and open Configuration."""
        session = getattr(self, "session", None) or SolveSession(
            current_family=selection.equation_type
        )
        self.session = session
        session.family_state().category = getattr(self, "_selected_category", None)
        session.family_state().equation_key = getattr(self, "_selected_key", None)
        session.select_equation(selection)
        self.win.destroy()

        from frontend.ui_dialogs.parameters_dialog import ParametersDialog

        ParametersDialog(
            self.parent,
            **selection.parameters_kwargs(),
            session=session,
            selection=selection,
        )

    def _on_next_predefined(self) -> None:
        if self._selected_key is None:
            messagebox.showwarning(
                "Choose an equation",
                "Select a built-in equation before continuing.",
                parent=self.win,
            )
            return

        eq = self.equations[self._selected_key]
        params: dict[str, float | list[float]] = {
            pname: float(pinfo.get("default", 0.0)) for pname, pinfo in eq.parameters.items()
        }

        eq_type: str = getattr(eq, "equation_type", "ode")
        variables: list[str] = getattr(eq, "variables", ["x"])
        vector_expressions: list[str] | None = getattr(eq, "vector_expressions", None)
        vector_components: int = getattr(eq, "vector_components", 1)
        self._open_configuration(
            EquationSelection(
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
                default_boundary_conditions=eq.default_boundary_conditions,
            )
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
        elif eq_type == "vector_pde":
            self._on_next_custom_vector_pde()
        elif eq_type == "pde_3d":
            self._on_next_custom_pde_3d()
        elif eq_type == "pde":
            self._on_next_custom_pde()
        else:
            self._on_next_custom_scalar()

    def _on_next_custom_scalar(self) -> None:
        from utils import normalize_unicode_escapes

        expr = normalize_unicode_escapes(self.custom_expr.get("1.0", tk.END).strip())
        if not expr:
            messagebox.showwarning(
                "Add an expression", "Write the expression before continuing.", parent=self.win
            )
            return

        try:
            order = int(self.custom_order_var.get())
        except ValueError:
            messagebox.showerror(
                "Check the order", "Order must be a positive integer.", parent=self.win
            )
            return

        params = self._parse_custom_params()
        if params is None:
            return

        eq_type = self._equation_type_var.get()
        default_domain: list[float] = [0.0, 50.0] if eq_type == "difference" else [0.0, 10.0]
        self._open_configuration(
            EquationSelection(
                expression=expr,
                function_name=None,
                order=order,
                parameters=params,
                equation_name="Custom Recurrence" if eq_type == "difference" else "Custom ODE",
                default_y0=[1.0] * order,
                default_domain=default_domain,
                equation_type=eq_type,
            )
        )

    def _on_next_custom_vector(self) -> None:
        import re

        from utils import normalize_unicode_escapes

        try:
            n_components = int(self._vec_n_var.get())
        except ValueError:
            messagebox.showerror(
                "Check the component count",
                "Number of components must be an integer.",
                parent=self.win,
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
                    "Check the order",
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
                    "Add an expression",
                    "Write the bulk expression before continuing.",
                    parent=self.win,
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
                    "Component boxes are out of sync",
                    "The number of expression boxes does not match the component count. "
                    "Change the component count again to refresh the boxes.",
                    parent=self.win,
                )
                return
            vector_expressions = []
            for idx, widget in enumerate(self._vec_expr_widgets):
                expr = normalize_unicode_escapes(widget.get("1.0", tk.END).strip())
                if not expr:
                    messagebox.showwarning(
                        "Add an expression",
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

        self._open_configuration(
            EquationSelection(
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
        )

    def _on_next_custom_pde(self) -> None:
        from utils import normalize_unicode_escapes

        expr = normalize_unicode_escapes(self.custom_expr.get("1.0", tk.END).strip())
        if not expr:
            messagebox.showwarning(
                "Add an expression", "Write the PDE expression before continuing.", parent=self.win
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

        default_domain = [0.0, 1.0, 0.0, 1.0]
        self._open_configuration(
            EquationSelection(
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
        )

    def _on_next_custom_pde_3d(self) -> None:
        """Validate a custom scalar 3D residual and open its grid dialog."""
        from utils import normalize_unicode_escapes

        expression = normalize_unicode_escapes(self.custom_expr.get("1.0", tk.END).strip())
        if not expression:
            messagebox.showwarning(
                "Add an expression",
                "Write the PDE 3D residual before continuing.",
                parent=self.win,
            )
            return
        params = self._parse_custom_params()
        if params is None:
            return
        self._open_configuration(
            EquationSelection(
                expression=expression,
                function_name=None,
                order=2,
                parameters=params,
                equation_name="Custom PDE 3D",
                default_y0=[],
                default_domain=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                equation_type="pde_3d",
                variables=["x", "y", "z"],
            )
        )

    def _on_next_custom_vector_pde(self) -> None:
        """Validate the custom Vector PDE editor and open shared PDE parameters."""
        from utils import normalize_unicode_escapes

        try:
            components = int(self._vec_n_var.get())
        except ValueError:
            messagebox.showerror(
                "Check the component count",
                "Number of components must be an integer.",
                parent=self.win,
            )
            return
        if not 1 <= components <= 4:
            messagebox.showerror(
                "Check the component count",
                "Vector PDE supports between 1 and 4 components in the standard UI.",
                parent=self.win,
            )
            return
        if len(self._vec_expr_widgets) != components:
            messagebox.showerror(
                "Component boxes are out of sync",
                "Change the component count again to refresh the residual boxes.",
                parent=self.win,
            )
            return

        residual_expressions: list[str] = []
        for component, widget in enumerate(self._vec_expr_widgets):
            expression = normalize_unicode_escapes(widget.get("1.0", tk.END).strip())
            if not expression:
                messagebox.showwarning(
                    "Add an expression",
                    f"Residual expression for component {component} is empty.",
                    parent=self.win,
                )
                return
            residual_expressions.append(expression)

        params = self._parse_custom_params()
        if params is None:
            return
        self._open_configuration(
            EquationSelection(
                expression=None,
                function_name=None,
                order=2,
                parameters=params,
                equation_name="Custom Vector PDE",
                default_y0=[],
                default_domain=[0.0, 1.0, 0.0, 1.0],
                equation_type="vector_pde",
                variables=["x", "y"],
                vector_expressions=residual_expressions,
                vector_components=components,
            )
        )
