"""UI dialog for configuring and solving coupled harmonic oscillators."""

from __future__ import annotations

import ast
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from complex_problems.coupled_oscillators.solver import solve_coupled_oscillators
from config import DEFAULT_SOLVER_METHOD, get_env_from_schema
from config.constants import SOLVER_METHODS
from frontend.theme import get_contrast_foreground, get_font
from frontend.ui_dialogs.loading_dialog import LoadingDialog
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import fit_and_center, make_modal
from utils import build_eval_namespace, get_logger, safe_eval, validate_expression_ast

logger = get_logger(__name__)

_BOUNDARY_OPTIONS = ("Fixed ends", "Periodic")

_COUPLED_OSC_INFO = """
Configure a 1D chain of N coupled harmonic oscillators.

• Mass and k: Single number = constant. Comma-separated = list per oscillator/spring.
  Expression with "i" (e.g. 1.0+0.1*i) or containing "[" = function of index.

• Boundary: Fixed ends (x_{-1} = x_N = 0) or Periodic (ring).

• Initial conditions: Oscillators (x_i,v_i) or Modes (q_i,dq_i). Comma-separated values.
"""


def _auto_parse_mass_or_k(
    text: str, n: int, n_springs: int, name: str, default_const: float, is_mass: bool
):
    """Auto-detect: single value=constant, comma-separated=list, '[' or expr=function.

    Returns float, list[float], or callable(i)->float.
    """
    text = text.strip()
    if not text:
        return default_const

    # If "[" appears, treat as function of index
    if "[" in text:
        return _parse_function_of_index(text, name)

    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) == 1:
        try:
            return float(parts[0])
        except ValueError:
            # Not a number — try as function (e.g. 1.0+0.1*i)
            return _parse_function_of_index(text, name)

    # Comma-separated list
    n_vals = n if is_mass else n_springs
    vals = []
    for i, p in enumerate(parts):
        if i >= n_vals:
            break
        try:
            vals.append(float(p))
        except ValueError:
            raise ValueError(f"{name}: invalid number at index {i}: '{p}'")
    if len(vals) < n_vals:
        last = vals[-1] if vals else default_const
        vals.extend([last] * (n_vals - len(vals)))
    return vals[:n_vals]


def _parse_function_of_index(expr: str, name: str):
    """Parse expression like '1.0 + 0.1*i' and return callable(i) -> float."""
    expr = expr.strip()
    if not expr:
        raise ValueError(f"{name} expression cannot be empty")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"{name}: invalid expression: {e}")
    validate_expression_ast(tree)
    compiled = compile(tree, "<string>", "eval")
    ns = build_eval_namespace({})

    def _eval(i: int) -> float:
        ns["i"] = float(i)
        return float(safe_eval(compiled, ns))

    return _eval


class CoupledOscillatorsDialog:
    """Dialog for configuring coupled harmonic oscillators parameters.

    Args:
        parent: Parent window.
    """

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Coupled Harmonic Oscillators")

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        self._build_ui()
        fit_and_center(
            self.win,
            min_width=1000,
            min_height=750,
            padding=48,
            resizable=True,
        )
        self.win.minsize(1000, 750)
        make_modal(self.win, parent)
        logger.info("Coupled oscillators dialog opened")

    def _build_ui(self) -> None:
        """Construct the dialog layout."""
        pad: int = get_env_from_schema("UI_PADDING")

        main_frame = ttk.Frame(self.win, padding=pad * 2)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self._scroll = ScrollableFrame(main_frame)
        self._scroll.apply_bg(get_env_from_schema("UI_BACKGROUND"))
        self._scroll.pack(fill=tk.BOTH, expand=True)
        inner = self._scroll.inner
        inner.configure(padding=pad)

        row = ttk.Frame(inner)
        row.pack(fill=tk.X, pady=pad)
        ttk.Label(row, text="Number of oscillators:").pack(side=tk.LEFT, padx=(0, pad))
        self._n_var = tk.StringVar(value="32")
        n_spin = ttk.Spinbox(
            row,
            textvariable=self._n_var,
            from_=2,
            to=100,
            width=6,
            font=get_font(),
        )
        n_spin.pack(side=tk.LEFT)
        ToolTip(n_spin, "Number of oscillators in the chain (2–100).")

        # Info section
        from frontend.ui_dialogs.collapsible_section import CollapsibleSection

        info_section = CollapsibleSection(
            inner, self._scroll, "How to configure", expanded=False, pad=pad
        )
        info_lbl = ttk.Label(
            info_section.content,
            text=_COUPLED_OSC_INFO.strip(),
            style="Small.TLabel",
            justify=tk.LEFT,
            wraplength=520,
        )
        info_lbl.pack(anchor=tk.W)
        self._scroll.bind_new_children()

        # Mass and k in same row (auto-detect: constant, list, or function)
        row = ttk.Frame(inner)
        row.pack(fill=tk.X, pady=pad)
        ttk.Label(row, text="Mass:").pack(side=tk.LEFT, padx=(0, pad))
        self._mass_entry_var = tk.StringVar(value="1.0")
        mass_entry = ttk.Entry(
            row, textvariable=self._mass_entry_var, width=24, font=get_font()
        )
        mass_entry.pack(side=tk.LEFT, padx=(0, pad * 2))
        ttk.Label(row, text="Coupling k:").pack(side=tk.LEFT, padx=(0, pad))
        self._k_entry_var = tk.StringVar(value="1.0")
        k_entry = ttk.Entry(
            row, textvariable=self._k_entry_var, width=24, font=get_font()
        )
        k_entry.pack(side=tk.LEFT)
        ToolTip(
            row,
            "Auto-detect: single number=constant, comma-separated=list, "
            "contains [ or expression with i=function of index.",
        )

        # Boundary
        row = ttk.Frame(inner)
        row.pack(fill=tk.X, pady=pad)
        ttk.Label(row, text="Boundary:").pack(side=tk.LEFT, padx=(0, pad))
        self._boundary_var = tk.StringVar(value="Fixed ends")
        boundary_combo = ttk.Combobox(
            row,
            textvariable=self._boundary_var,
            values=_BOUNDARY_OPTIONS,
            state="readonly",
            width=12,
            font=get_font(),
        )
        boundary_combo.pack(side=tk.LEFT)
        ToolTip(boundary_combo, "Fixed ends: x_{-1}=x_N=0. Periodic: chain forms a ring.")

        # Coupling types (multi-select Listbox + Equations button)
        row = ttk.Frame(inner)
        row.pack(fill=tk.X, pady=pad)
        ttk.Label(row, text="Coupling types:").pack(side=tk.LEFT, padx=(0, pad))
        btn_bg = get_env_from_schema("UI_BUTTON_BG")
        fg = get_env_from_schema("UI_FOREGROUND")
        select_bg = get_env_from_schema("UI_BUTTON_FG")
        select_fg = get_contrast_foreground(select_bg)
        self._coupling_listbox = tk.Listbox(
            row,
            selectmode=tk.EXTENDED,
            height=8,
            width=24,
            bg=btn_bg,
            fg=fg,
            selectbackground=select_bg,
            selectforeground=select_fg,
            font=get_font(),
            exportselection=False,
        )
        for item in (
            "2nd neighbor",
            "3rd neighbor",
            "4th neighbor",
            "FPUT-α",
            "Nonlinear (cubic)",
            "Nonlinear (quartic)",
            "Nonlinear (quintic)",
            "External force",
        ):
            self._coupling_listbox.insert(tk.END, item)
        self._coupling_listbox.pack(side=tk.LEFT, padx=(0, pad))
        ttk.Button(
            row,
            text="Equations",
            command=self._show_coupling_equations,
        ).pack(side=tk.LEFT)
        self._coupling_listbox.bind("<<ListboxSelect>>", self._on_coupling_selection_change)

        # Long-range params: one row per selected neighbor (only its own k)
        self._k_2nn_frame = ttk.Frame(inner)
        ttk.Label(self._k_2nn_frame, text="k₂ (2nd neighbor):").pack(
            side=tk.LEFT, padx=(0, pad)
        )
        self._k_2nn_var = tk.StringVar(value="25")
        ttk.Entry(
            self._k_2nn_frame, textvariable=self._k_2nn_var, width=6, font=get_font()
        ).pack(side=tk.LEFT)
        self._k_3nn_frame = ttk.Frame(inner)
        ttk.Label(self._k_3nn_frame, text="k₃ (3rd neighbor):").pack(
            side=tk.LEFT, padx=(0, pad)
        )
        self._k_3nn_var = tk.StringVar(value="15")
        ttk.Entry(
            self._k_3nn_frame, textvariable=self._k_3nn_var, width=6, font=get_font()
        ).pack(side=tk.LEFT)
        self._k_4nn_frame = ttk.Frame(inner)
        ttk.Label(self._k_4nn_frame, text="k₄ (4th neighbor):").pack(
            side=tk.LEFT, padx=(0, pad)
        )
        self._k_4nn_var = tk.StringVar(value="10")
        ttk.Entry(
            self._k_4nn_frame, textvariable=self._k_4nn_var, width=6, font=get_font()
        ).pack(side=tk.LEFT)

        # Nonlinear params: one row per selected (only its own ε)
        self._fput_alpha_frame = ttk.Frame(inner)
        ttk.Label(self._fput_alpha_frame, text="α (FPUT-α):").pack(
            side=tk.LEFT, padx=(0, pad)
        )
        self._fput_alpha_var = tk.StringVar(value="0.25")
        ttk.Entry(
            self._fput_alpha_frame,
            textvariable=self._fput_alpha_var,
            width=6,
            font=get_font(),
        ).pack(side=tk.LEFT)
        self._cubic_frame = ttk.Frame(inner)
        ttk.Label(self._cubic_frame, text="ε₃ (cubic):").pack(side=tk.LEFT, padx=(0, pad))
        self._nonlinear_coeff_var = tk.StringVar(value="80")
        ttk.Entry(
            self._cubic_frame,
            textvariable=self._nonlinear_coeff_var,
            width=6,
            font=get_font(),
        ).pack(side=tk.LEFT)
        self._quartic_frame = ttk.Frame(inner)
        ttk.Label(self._quartic_frame, text="ε₄ (quartic):").pack(
            side=tk.LEFT, padx=(0, pad)
        )
        self._nonlinear_quartic_var = tk.StringVar(value="150")
        ttk.Entry(
            self._quartic_frame,
            textvariable=self._nonlinear_quartic_var,
            width=6,
            font=get_font(),
        ).pack(side=tk.LEFT)
        self._quintic_frame = ttk.Frame(inner)
        ttk.Label(self._quintic_frame, text="ε₅ (quintic):").pack(
            side=tk.LEFT, padx=(0, pad)
        )
        self._nonlinear_quintic_var = tk.StringVar(value="5")
        ttk.Entry(
            self._quintic_frame,
            textvariable=self._nonlinear_quintic_var,
            width=6,
            font=get_font(),
        ).pack(side=tk.LEFT)

        # External force params (shown only when "External force" is selected)
        self._external_params_frame = ttk.Frame(inner)
        self._external_params_frame.pack(fill=tk.X, pady=pad)
        row_ext = ttk.Frame(self._external_params_frame)
        row_ext.pack(fill=tk.X)
        ttk.Label(row_ext, text="External F:").pack(side=tk.LEFT, padx=(0, pad))
        self._external_amp_var = tk.StringVar(value="50")
        ttk.Entry(
            row_ext,
            textvariable=self._external_amp_var,
            width=8,
            font=get_font(),
        ).pack(side=tk.LEFT, padx=(0, pad * 2))
        ttk.Label(row_ext, text="External Ω:").pack(side=tk.LEFT, padx=(0, pad))
        self._external_freq_var = tk.StringVar(value="1.0")
        ttk.Entry(
            row_ext,
            textvariable=self._external_freq_var,
            width=8,
            font=get_font(),
        ).pack(side=tk.LEFT)

        # Domain (store ref for packing extra params before it)
        self._domain_row = ttk.Frame(inner)
        self._domain_row.pack(fill=tk.X, pady=pad)
        row = self._domain_row
        ttk.Label(row, text="Time domain:").pack(side=tk.LEFT, padx=(0, pad))
        self._t_min_var = tk.StringVar(value="0.0")
        self._t_max_var = tk.StringVar(value="200.0")
        ttk.Entry(
            row, textvariable=self._t_min_var, width=8, font=get_font()
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(row, text="to").pack(side=tk.LEFT, padx=4)
        ttk.Entry(
            row, textvariable=self._t_max_var, width=8, font=get_font()
        ).pack(side=tk.LEFT)
        ToolTip(row, "Integration time interval [t_min, t_max].")

        # Resolution points and solver method
        row_res = ttk.Frame(inner)
        row_res.pack(fill=tk.X, pady=pad)
        ttk.Label(row_res, text="Resolution points:").pack(side=tk.LEFT, padx=(0, pad))
        default_n_points = max(2000, int(get_env_from_schema("SOLVER_NUM_POINTS")))
        self._n_points_var = tk.StringVar(value=str(default_n_points))
        ttk.Entry(
            row_res, textvariable=self._n_points_var, width=10, font=get_font()
        ).pack(side=tk.LEFT, padx=(0, pad * 2))
        ttk.Label(row_res, text="Solver:").pack(side=tk.LEFT, padx=(pad * 2, pad))
        self._method_var = tk.StringVar(value=DEFAULT_SOLVER_METHOD)
        method_combo = ttk.Combobox(
            row_res,
            textvariable=self._method_var,
            values=list(SOLVER_METHODS),
            state="readonly",
            width=10,
            font=get_font(),
        )
        method_combo.pack(side=tk.LEFT)
        ToolTip(row_res, "Number of output points and ODE solver method.")

        self._update_extra_params_visibility()

        # Initial conditions: Oscillators or Modes
        ic_frame = ttk.Frame(inner)
        ic_frame.pack(fill=tk.X, pady=pad)
        ic_row1 = ttk.Frame(ic_frame)
        ic_row1.pack(fill=tk.X)
        ttk.Label(ic_row1, text="Initial conditions in:").pack(side=tk.LEFT, padx=(0, pad))
        self._ic_space_var = tk.StringVar(value="Modes")
        ic_space_combo = ttk.Combobox(
            ic_row1,
            textvariable=self._ic_space_var,
            values=("Oscillators", "Modes"),
            state="readonly",
            width=12,
            font=get_font(),
        )
        ic_space_combo.pack(side=tk.LEFT, padx=(0, pad * 2))
        ic_space_combo.bind("<<ComboboxSelected>>", self._on_ic_space_change)
        ToolTip(
            ic_space_combo,
            "Oscillators: xᵢ, vᵢ. Modes: qᵢ, dqᵢ (converted to oscillator space for solving).",
        )

        ic_row2 = ttk.Frame(ic_frame)
        ic_row2.pack(fill=tk.X, pady=(pad // 2, 0))
        self._ic_pos_label = ttk.Label(ic_row2, text="Positions (q₁,q₂,...):")
        self._ic_pos_label.pack(side=tk.LEFT, padx=(0, pad))
        self._ic_pos_var = tk.StringVar(value="5," + ",".join("0" for _ in range(31)))
        self._ic_pos_entry = ttk.Entry(
            ic_row2,
            textvariable=self._ic_pos_var,
            width=48,
            font=get_font(),
        )
        self._ic_pos_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, pad))
        ToolTip(self._ic_pos_entry, "Comma-separated values. Default: 1 for first, 0 for rest.")

        ic_row3 = ttk.Frame(ic_frame)
        ic_row3.pack(fill=tk.X, pady=(pad // 2, 0))
        self._ic_vel_label = ttk.Label(ic_row3, text="Velocities (dq₁,dq₂,...):")
        self._ic_vel_label.pack(side=tk.LEFT, padx=(0, pad))
        self._ic_vel_var = tk.StringVar(value=",".join("0" for _ in range(32)))
        self._ic_vel_entry = ttk.Entry(
            ic_row3,
            textvariable=self._ic_vel_var,
            width=48,
            font=get_font(),
        )
        self._ic_vel_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, pad))
        ToolTip(self._ic_vel_entry, "Comma-separated values. Default: all zeros.")

        self._n_var.trace_add("write", lambda *a: self._on_n_change())

        # Solve button
        btn_frame = ttk.Frame(inner)
        btn_frame.pack(fill=tk.X, pady=pad * 2)

        btn_solve = ttk.Button(
            btn_frame,
            text="Solve",
            command=self._on_solve,
        )
        btn_solve.pack(side=tk.LEFT, padx=(0, pad))

        btn_close = ttk.Button(
            btn_frame,
            text="Close",
            style="Cancel.TButton",
            command=self.win.destroy,
        )
        btn_close.pack(side=tk.LEFT)

        self._scroll.bind_new_children()

    def _on_ic_space_change(self, _event: tk.Event | None = None) -> None:
        """Update labels when Oscillators/Modes selection changes."""
        space = self._ic_space_var.get()
        if space == "Oscillators":
            self._ic_pos_label.config(text="Positions (x₀,x₁,...):")
            self._ic_vel_label.config(text="Velocities (v₀,v₁,...):")
        else:
            # Physics convention: q₁ = Mode 1 (fundamental), q₂ = Mode 2, etc.
            self._ic_pos_label.config(text="Positions (q₁,q₂,...):")
            self._ic_vel_label.config(text="Velocities (dq₁,dq₂,...):")

    def _on_n_change(self) -> None:
        """Update default IC length when number of oscillators changes."""
        try:
            n = int(self._n_var.get())
            if n < 2 or n > 100:
                return
        except ValueError:
            return
        # Build default: mode 1 excited with amplitude 5 (visible nonlinear effect)
        pos_default = "5," + ",".join("0" for _ in range(n - 1))
        vel_default = ",".join("0" for _ in range(n))
        self._ic_pos_var.set(pos_default)
        self._ic_vel_var.set(vel_default)

    def _on_coupling_selection_change(self, _event: tk.Event) -> None:
        """Update visibility of coefficient fields based on listbox selection."""
        self._update_extra_params_visibility()

    def _update_extra_params_visibility(self) -> None:
        """Show/hide params based on listbox selection."""
        pad = int(get_env_from_schema("UI_PADDING"))
        selected = {
            self._coupling_listbox.get(i)
            for i in self._coupling_listbox.curselection()
        }
        # Show only the k field for each selected neighbor (pack 4th first so 2nd ends on top)
        for label, frame in (
            ("4th neighbor", self._k_4nn_frame),
            ("3rd neighbor", self._k_3nn_frame),
            ("2nd neighbor", self._k_2nn_frame),
        ):
            if label in selected:
                if not frame.winfo_manager():
                    frame.pack(fill=tk.X, pady=pad, before=self._domain_row)
            else:
                frame.pack_forget()
        # Show only the ε field for each selected nonlinear
        for label, frame in (
            ("Nonlinear (quintic)", self._quintic_frame),
            ("Nonlinear (quartic)", self._quartic_frame),
            ("Nonlinear (cubic)", self._cubic_frame),
            ("FPUT-α", self._fput_alpha_frame),
        ):
            if label in selected:
                if not frame.winfo_manager():
                    frame.pack(fill=tk.X, pady=pad, before=self._domain_row)
            else:
                frame.pack_forget()
        if "External force" in selected:
            if not self._external_params_frame.winfo_manager():
                self._external_params_frame.pack(
                    fill=tk.X, pady=pad, before=self._domain_row
                )
        else:
            self._external_params_frame.pack_forget()
        self._scroll.refresh_scroll_region()

    def _show_coupling_equations(self) -> None:
        """Show equations for each coupling type in a formatted help window."""
        pad = int(get_env_from_schema("UI_PADDING"))
        bg = get_env_from_schema("UI_BACKGROUND")
        txt_bg = get_env_from_schema("UI_BUTTON_BG")
        txt_fg = get_env_from_schema("UI_FOREGROUND")

        sections: list[tuple[str, str]] = [
            ("Linear (always active)", "k·(xᵢ₊₁+xᵢ₋₁-2xᵢ)"),
            ("2nd neighbor", "k₂·(xᵢ₊₂+xᵢ₋₂-2xᵢ)"),
            ("3rd neighbor", "k₃·(xᵢ₊₃+xᵢ₋₃-2xᵢ)"),
            ("4th neighbor", "k₄·(xᵢ₊₄+xᵢ₋₄-2xᵢ)"),
            ("FPUT-α", "α·(xᵢ₊₁+xᵢ₋₁-2xᵢ)·(xᵢ₊₁-xᵢ₋₁)"),
            ("Cubic", "ε₃·(xᵢ₊₁+xᵢ₋₁-2xᵢ)³"),
            ("Quartic", "ε₄·sign(L)·|L|⁴  with  L = xᵢ₊₁+xᵢ₋₁-2xᵢ"),
            ("Quintic", "ε₅·(xᵢ₊₁+xᵢ₋₁-2xᵢ)⁵"),
            ("External force", "F·cos(Ωt)"),
        ]

        dlg = tk.Toplevel(self.win)
        dlg.title("Equations — Coupled Oscillators")
        dlg.transient(self.win)
        dlg.configure(bg=bg)

        main = ttk.Frame(dlg, padding=pad * 2)
        main.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(main)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(pad, 0))
        ttk.Button(
            btn_frame,
            text="Close",
            style="Cancel.TButton",
            command=dlg.destroy,
        ).pack(side=tk.RIGHT)

        ttk.Label(
            main,
            text="Equations of motion",
            style="Title.TLabel",
        ).pack(anchor=tk.W, pady=(0, pad))

        txt = tk.Text(
            main,
            wrap=tk.WORD,
            width=48,
            height=14,
            font=get_font(),
            bg=txt_bg,
            fg=txt_fg,
            relief=tk.FLAT,
        )
        scroll = ttk.Scrollbar(main, orient=tk.VERTICAL, command=txt.yview)
        txt.configure(yscrollcommand=scroll.set)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=(0, pad))
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, pad))

        for i, (title, body) in enumerate(sections):
            if i > 0:
                txt.insert(tk.END, "\n\n")
            txt.insert(tk.END, f"▸ {title}\n", "heading")
            txt.insert(tk.END, f"{body}\n", "body")
        txt.tag_configure("heading", font=(get_font()[0], get_font()[1], "bold"))
        txt.tag_configure("body", font=get_font())
        txt.config(state=tk.DISABLED)

        fit_and_center(dlg, min_width=380, min_height=340, padding=32)
        make_modal(dlg, self.win)

    def _resolve_masses(self, n: int):
        """Resolve mass spec from UI (auto-detect constant, list, or function)."""
        text = self._mass_entry_var.get().strip()
        n_springs = n - 1 if self._boundary_var.get() == "Fixed ends" else n
        return _auto_parse_mass_or_k(text, n, n_springs, "Mass", 1.0, is_mass=True)

    def _resolve_k_coupling(self, n: int):
        """Resolve k spec from UI (auto-detect constant, list, or function)."""
        text = self._k_entry_var.get().strip()
        n_springs = n - 1 if self._boundary_var.get() == "Fixed ends" else n
        return _auto_parse_mass_or_k(
            text, n, n_springs, "Coupling k", 0.3, is_mass=False
        )

    def _on_solve(self) -> None:
        """Start the solver in a background thread."""
        try:
            n = int(self._n_var.get())
            if n < 2 or n > 100:
                messagebox.showerror(
                    "Invalid input",
                    "Number of oscillators must be between 2 and 100.",
                    parent=self.win,
                )
                return
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Number of oscillators must be an integer.",
                parent=self.win,
            )
            return

        try:
            masses = self._resolve_masses(n)
            if isinstance(masses, (list, tuple)):
                if any(m <= 0 for m in masses):
                    raise ValueError("All masses must be positive")
            elif isinstance(masses, (int, float)):
                if masses <= 0:
                    raise ValueError("Mass must be positive")
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e), parent=self.win)
            return

        try:
            k_coupling = self._resolve_k_coupling(n)
            if isinstance(k_coupling, (list, tuple)):
                if any(k < 0 for k in k_coupling):
                    raise ValueError("All coupling constants must be non-negative")
            elif isinstance(k_coupling, (int, float)):
                if k_coupling < 0:
                    raise ValueError("Coupling k must be non-negative")
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e), parent=self.win)
            return

        try:
            t_min = float(self._t_min_var.get())
            t_max = float(self._t_max_var.get())
            if t_min >= t_max:
                raise ValueError("t_max must be greater than t_min")
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e), parent=self.win)
            return

        try:
            n_points = int(self._n_points_var.get())
            if n_points < 2:
                raise ValueError("Resolution points must be at least 2")
        except ValueError as e:
            messagebox.showerror(
                "Invalid input",
                f"Resolution points: {e}" if str(e) else "Resolution points must be an integer.",
                parent=self.win,
            )
            return

        method = self._method_var.get()
        if method not in SOLVER_METHODS:
            method = DEFAULT_SOLVER_METHOD

        # Parse initial conditions
        try:
            pos_str = self._ic_pos_var.get().strip()
            vel_str = self._ic_vel_var.get().strip()
            pos_vals = [float(p.strip()) for p in pos_str.split(",") if p.strip()]
            vel_vals = [float(v.strip()) for v in vel_str.split(",") if v.strip()]
            if len(pos_vals) < n or len(vel_vals) < n:
                raise ValueError(
                    f"Initial conditions need at least {n} values each. "
                    f"Got {len(pos_vals)} positions, {len(vel_vals)} velocities."
                )
            pos_vals = pos_vals[:n]
            vel_vals = vel_vals[:n]
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e), parent=self.win)
            return

        ic_space = self._ic_space_var.get()

        # Linear coupling is always active; add nonlinear/external/long-range from listbox
        selected_labels = {
            self._coupling_listbox.get(i)
            for i in self._coupling_listbox.curselection()
        }
        _label_to_type = {
            "FPUT-α": "nonlinear_fput_alpha",
            "Nonlinear (cubic)": "nonlinear",
            "Nonlinear (quartic)": "nonlinear_quartic",
            "Nonlinear (quintic)": "nonlinear_quintic",
            "External force": "external_force",
        }
        coupling_types: list[str] = ["linear"]
        for label in selected_labels:
            if label in _label_to_type:
                coupling_types.append(_label_to_type[label])

        # Long-range: only use k values for selected neighbors
        try:
            k_2nn = (
                max(0.0, float(self._k_2nn_var.get()))
                if "2nd neighbor" in selected_labels
                else 0.0
            )
        except ValueError:
            k_2nn = 25.0 if "2nd neighbor" in selected_labels else 0.0
        try:
            k_3nn = (
                max(0.0, float(self._k_3nn_var.get()))
                if "3rd neighbor" in selected_labels
                else 0.0
            )
        except ValueError:
            k_3nn = 15.0 if "3rd neighbor" in selected_labels else 0.0
        try:
            k_4nn = (
                max(0.0, float(self._k_4nn_var.get()))
                if "4th neighbor" in selected_labels
                else 0.0
            )
        except ValueError:
            k_4nn = 10.0 if "4th neighbor" in selected_labels else 0.0

        try:
            nonlinear_fput_alpha = (
                float(self._fput_alpha_var.get())
                if "nonlinear_fput_alpha" in coupling_types
                else 0.0
            )
        except ValueError:
            nonlinear_fput_alpha = 0.25
        try:
            nonlinear_coeff = (
                float(self._nonlinear_coeff_var.get())
                if "nonlinear" in coupling_types
                else 0.0
            )
        except ValueError:
            nonlinear_coeff = 80.0
        try:
            nonlinear_quartic = (
                float(self._nonlinear_quartic_var.get())
                if "nonlinear_quartic" in coupling_types
                else 0.0
            )
        except ValueError:
            nonlinear_quartic = 15.0
        try:
            nonlinear_quintic = (
                float(self._nonlinear_quintic_var.get())
                if "nonlinear_quintic" in coupling_types
                else 0.0
            )
        except ValueError:
            nonlinear_quintic = 5.0
        try:
            external_amp = (
                float(self._external_amp_var.get())
                if "external_force" in coupling_types
                else 0.0
            )
        except ValueError:
            external_amp = 50.0
        try:
            external_freq = (
                float(self._external_freq_var.get())
                if "external_force" in coupling_types
                else 1.0
            )
        except ValueError:
            external_freq = 1.0

        boundary = "periodic" if self._boundary_var.get() == "Periodic" else "fixed"

        # Build initial conditions: [x_0, ..., x_{N-1}, v_0, ..., v_{N-1}]
        if ic_space == "Oscillators":
            y0 = list(pos_vals) + list(vel_vals)
        else:
            # Modes: convert (q, dq) to (x, v) via x = M_modes @ q, v = M_modes @ dq
            from complex_problems.coupled_oscillators.model import compute_normal_modes

            M_modes, _ = compute_normal_modes(
                n, masses, k_coupling, boundary,
                k_2nn=k_2nn, k_3nn=k_3nn, k_4nn=k_4nn,
            )
            q = np.array(pos_vals, dtype=float)
            dq = np.array(vel_vals, dtype=float)
            x = M_modes @ q
            v = M_modes @ dq
            y0 = list(x) + list(v)

        result_queue: queue.Queue = queue.Queue()

        def _run_solver() -> None:
            try:
                result = solve_coupled_oscillators(
                    n_oscillators=n,
                    masses=masses,
                    k_coupling=k_coupling,
                    boundary=boundary,
                    coupling_types=coupling_types,
                    nonlinear_coeff=nonlinear_coeff,
                    nonlinear_fput_alpha=nonlinear_fput_alpha,
                    nonlinear_quartic=nonlinear_quartic,
                    nonlinear_quintic=nonlinear_quintic,
                    k_2nn=k_2nn,
                    k_3nn=k_3nn,
                    k_4nn=k_4nn,
                    external_amplitude=external_amp,
                    external_frequency=external_freq,
                    t_min=t_min,
                    t_max=t_max,
                    n_points=n_points,
                    y0=y0,
                    method=method,
                )
                result_queue.put(("success", result))
            except Exception as exc:
                logger.exception("Coupled oscillators solver failed")
                result_queue.put(("error", ("Solver Error", str(exc))))

        thread = threading.Thread(target=_run_solver, daemon=True)
        thread.start()

        loading = LoadingDialog(self.parent, message="Solving coupled oscillators...")
        self.win.destroy()

        def _check_result() -> None:
            try:
                status, data = result_queue.get_nowait()
            except queue.Empty:
                self.parent.after(100, _check_result)
                return

            loading.destroy()

            if status == "success":
                from complex_problems.coupled_oscillators.result_dialog import (
                    CoupledOscillatorsResultDialog,
                )

                CoupledOscillatorsResultDialog(self.parent, result=data)
            else:
                title, msg = data
                messagebox.showerror(title, msg, parent=self.parent)

        self.parent.after(100, _check_result)
