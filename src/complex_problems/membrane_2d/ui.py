"""UI dialog for configuring a 2D nonlinear membrane simulation."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from complex_problems.common import (
    add_how_to_config_section,
    compile_scalar_expression,
    parse_float,
    parse_int,
    parse_positive_float,
    parse_positive_int,
    run_solver_with_loading,
)
from complex_problems.membrane_2d.model import build_initial_displacement
from complex_problems.membrane_2d.solver import solve_membrane_2d
from config import get_env_from_schema
from frontend.theme import get_contrast_foreground, get_font
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import fit_and_center, make_modal

_BOUNDARY_OPTIONS = ("fixed", "periodic")
_INTEGRATOR_OPTIONS = ("verlet", "rk45")
_IC_SHAPES = ("gaussian", "mode", "random", "custom")

_TERM_ALPHA = "Quadratic α(Δu)²"
_TERM_BETA = "Cubic β(Δu)³"
_TERM_HIGH = "Higher-order cₚ sign(Δu)|Δu|ᵖ"
_OPTIONAL_TERMS = (_TERM_ALPHA, _TERM_BETA, _TERM_HIGH)


def resolve_optional_membrane_terms(
    selected_labels: set[str],
    *,
    alpha_text: str,
    beta_text: str,
    high_coeff_text: str,
    high_power_text: str,
) -> tuple[float, float, float, int]:
    """Resolve optional membrane nonlinear terms from selected labels."""
    alpha = parse_float(alpha_text, name="α") if _TERM_ALPHA in selected_labels else 0.0
    beta = parse_float(beta_text, name="β") if _TERM_BETA in selected_labels else 0.0
    if _TERM_HIGH in selected_labels:
        high_coeff = parse_float(high_coeff_text, name="cₚ")
        high_power = parse_positive_int(high_power_text, name="p", min_value=2)
    else:
        high_coeff = 0.0
        high_power = 5
    return alpha, beta, high_coeff, high_power


class Membrane2DDialog:
    """Configuration dialog for the membrane problem."""

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("2D Nonlinear Membrane")
        self.win.configure(bg=get_env_from_schema("UI_BACKGROUND"))
        self._build_ui()
        fit_and_center(self.win, min_width=980, min_height=760, padding=32, resizable=True)
        self.win.minsize(920, 700)
        make_modal(self.win, parent)

    def _build_ui(self) -> None:
        pad = int(get_env_from_schema("UI_PADDING"))
        root = ttk.Frame(self.win, padding=pad * 2)
        root.pack(fill=tk.BOTH, expand=True)

        scroll = ScrollableFrame(root)
        scroll.apply_bg(get_env_from_schema("UI_BACKGROUND"))
        scroll.pack(fill=tk.BOTH, expand=True)
        body = scroll.inner
        body.configure(padding=pad)

        ttk.Label(body, text="2D Nonlinear Membrane", style="Title.TLabel").pack(
            anchor=tk.W, pady=(0, pad)
        )
        ttk.Label(
            body,
            text=(
                "Discrete membrane lattice with fixed or periodic boundaries.\n"
                "The linear Laplacian term is always active; optional nonlinear terms can be added."
            ),
            style="Small.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, pad))

        add_how_to_config_section(
            body,
            scroll,
            problem_id="membrane_2d",
            pad=pad,
            wraplength=780,
        )

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._nx_var = tk.StringVar(value="32")
        self._ny_var = tk.StringVar(value="32")
        self._make_spinbox(row, "Nₓ", self._nx_var, from_=8, to=2048, width=7)
        self._make_spinbox(row, "Nᵧ", self._ny_var, from_=8, to=2048, width=7)
        ToolTip(row, "Grid size along x and y (integers).")

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._boundary_var = tk.StringVar(value="fixed")
        self._integrator_var = tk.StringVar(value="verlet")
        self._make_combo(row, "Boundary", self._boundary_var, _BOUNDARY_OPTIONS, width=10)
        self._make_combo(
            row, "Integrator", self._integrator_var, _INTEGRATOR_OPTIONS, width=10
        )
        ToolTip(
            row,
            "Verlet is faster and more energy-stable for Hamiltonian-like dynamics. "
            "RK45 is provided for comparison.",
        )

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._mass_var = tk.StringVar(value="1.0")
        self._k_var = tk.StringVar(value="1.0")
        self._make_entry(row, "Mass m", self._mass_var, width=10)
        self._make_entry(row, "Linear k", self._k_var, width=10)

        btn_bg = get_env_from_schema("UI_BUTTON_BG")
        fg = get_env_from_schema("UI_FOREGROUND")
        select_bg = get_env_from_schema("UI_BUTTON_FG")
        select_fg = get_contrast_foreground(select_bg)

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        ttk.Label(row, text="Optional nonlinear terms:").pack(side=tk.LEFT, padx=(0, pad))
        self._optional_terms_listbox = tk.Listbox(
            row,
            selectmode=tk.EXTENDED,
            height=4,
            width=36,
            bg=btn_bg,
            fg=fg,
            selectbackground=select_bg,
            selectforeground=select_fg,
            font=get_font(),
            exportselection=False,
        )
        for term in _OPTIONAL_TERMS:
            self._optional_terms_listbox.insert(tk.END, term)
        self._optional_terms_listbox.pack(side=tk.LEFT, padx=(0, pad))
        self._optional_terms_listbox.bind(
            "<<ListboxSelect>>", lambda _e: self._update_optional_terms_visibility()
        )
        ToolTip(
            self._optional_terms_listbox,
            "Linear k·Δu is always active. Select one or more optional nonlinear terms.",
        )

        self._alpha_var = tk.StringVar(value="0.0")
        self._beta_var = tk.StringVar(value="0.0")
        self._high_coeff_var = tk.StringVar(value="0.0")
        self._high_power_var = tk.StringVar(value="5")

        self._alpha_frame = ttk.Frame(body)
        self._make_entry(self._alpha_frame, "α coefficient", self._alpha_var, width=10)
        self._beta_frame = ttk.Frame(body)
        self._make_entry(self._beta_frame, "β coefficient", self._beta_var, width=10)
        self._high_frame = ttk.Frame(body)
        self._make_entry(self._high_frame, "cₚ coefficient", self._high_coeff_var, width=10)
        self._make_entry(self._high_frame, "Power p", self._high_power_var, width=8)
        ToolTip(
            self._high_frame,
            "Higher-order contribution uses cₚ·sign(Δu)·|Δu|ᵖ with integer p ≥ 2.",
        )
        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._t_min_var = tk.StringVar(value="0.0")
        self._t_max_var = tk.StringVar(value="20.0")
        self._dt_var = tk.StringVar(value="0.02")
        self._make_entry(row, "tₘᵢₙ", self._t_min_var, width=8)
        self._make_entry(row, "tₘₐₓ", self._t_max_var, width=8)
        self._make_entry(row, "Δt", self._dt_var, width=8)

        ttk.Separator(body).pack(fill=tk.X, pady=pad)
        ttk.Label(body, text="Initial condition", style="Small.TLabel").pack(anchor=tk.W)

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._ic_shape_var = tk.StringVar(value="gaussian")
        shape_combo = self._make_combo(
            row, "Shape", self._ic_shape_var, _IC_SHAPES, width=12
        )
        shape_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_ic_visibility())
        self._amp_var = tk.StringVar(value="1.0")
        self._sigma_var = tk.StringVar(value="0.12")
        self._make_entry(row, "Amplitude", self._amp_var, width=8)
        self._make_entry(row, "σ", self._sigma_var, width=8)

        self._mode_row = ttk.Frame(body)
        self._mode_row.pack(fill=tk.X, pady=pad // 2)
        self._mode_x_var = tk.StringVar(value="1")
        self._mode_y_var = tk.StringVar(value="1")
        self._make_entry(self._mode_row, "Mode nₓ", self._mode_x_var, width=6)
        self._make_entry(self._mode_row, "Mode nᵧ", self._mode_y_var, width=6)

        self._custom_row = ttk.Frame(body)
        self._custom_row.pack(fill=tk.X, pady=pad // 2)
        ttk.Label(self._custom_row, text="u₀(x,y) =").pack(side=tk.LEFT, padx=(0, 4))
        self._custom_expr_var = tk.StringVar(value="exp(-((x-0.5)**2 + (y-0.5)**2)/0.02)")
        self._custom_entry = ttk.Entry(
            self._custom_row,
            textvariable=self._custom_expr_var,
            width=54,
            font=get_font(),
        )
        self._custom_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ToolTip(self._custom_entry, "Custom expression using x and y in [0,1].")

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._center_x_var = tk.StringVar(value="0.5")
        self._center_y_var = tk.StringVar(value="0.5")
        self._seed_var = tk.StringVar(value="0")
        self._make_entry(row, "Center x₀", self._center_x_var, width=8)
        self._make_entry(row, "Center y₀", self._center_y_var, width=8)
        self._make_entry(row, "Random seed", self._seed_var, width=8)

        self._btn_row = ttk.Frame(body)
        self._btn_row.pack(fill=tk.X, pady=(pad * 2, 0))
        ttk.Button(self._btn_row, text="Solve", command=self._on_solve).pack(
            side=tk.LEFT, padx=(0, pad)
        )
        ttk.Button(
            self._btn_row,
            text="Close",
            style="Cancel.TButton",
            command=self.win.destroy,
        ).pack(
            side=tk.LEFT
        )

        self._update_optional_terms_visibility()
        self._update_ic_visibility()

        scroll.bind_new_children()

    def _make_entry(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.StringVar,
        *,
        width: int = 10,
    ) -> ttk.Entry:
        ttk.Label(parent, text=f"{label}:").pack(side=tk.LEFT, padx=(0, 4))
        entry = ttk.Entry(parent, textvariable=var, width=width, font=get_font())
        entry.pack(side=tk.LEFT, padx=(0, 12))
        return entry

    def _make_spinbox(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.StringVar,
        *,
        from_: int,
        to: int,
        width: int = 8,
    ) -> ttk.Spinbox:
        ttk.Label(parent, text=f"{label}:").pack(side=tk.LEFT, padx=(0, 4))
        spin = ttk.Spinbox(
            parent,
            textvariable=var,
            from_=from_,
            to=to,
            width=width,
            font=get_font(),
        )
        spin.pack(side=tk.LEFT, padx=(0, 12))
        return spin

    def _make_combo(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.StringVar,
        values: tuple[str, ...],
        *,
        width: int = 12,
    ) -> ttk.Combobox:
        ttk.Label(parent, text=f"{label}:").pack(side=tk.LEFT, padx=(0, 4))
        combo = ttk.Combobox(
            parent,
            textvariable=var,
            values=list(values),
            state="readonly",
            width=width,
            font=get_font(),
        )
        combo.pack(side=tk.LEFT, padx=(0, 12))
        return combo

    def _update_optional_terms_visibility(self) -> None:
        selected = {
            self._optional_terms_listbox.get(i)
            for i in self._optional_terms_listbox.curselection()
        }
        if _TERM_ALPHA in selected:
            self._alpha_frame.pack(fill=tk.X, pady=4, before=self._btn_row)
        else:
            self._alpha_frame.pack_forget()

        if _TERM_BETA in selected:
            self._beta_frame.pack(fill=tk.X, pady=4, before=self._btn_row)
        else:
            self._beta_frame.pack_forget()

        if _TERM_HIGH in selected:
            self._high_frame.pack(fill=tk.X, pady=4, before=self._btn_row)
        else:
            self._high_frame.pack_forget()

    def _update_ic_visibility(self) -> None:
        shape = self._ic_shape_var.get()
        if shape == "mode":
            self._mode_row.pack(fill=tk.X, pady=4, before=self._btn_row)
        else:
            self._mode_row.pack_forget()

        if shape == "custom":
            self._custom_row.pack(fill=tk.X, pady=4, before=self._btn_row)
        else:
            self._custom_row.pack_forget()

    def _collect_inputs(self) -> dict[str, object]:
        nx = parse_positive_int(self._nx_var.get(), name="Nₓ", min_value=8)
        ny = parse_positive_int(self._ny_var.get(), name="Nᵧ", min_value=8)
        boundary = self._boundary_var.get()
        integrator = self._integrator_var.get()
        mass = parse_positive_float(self._mass_var.get(), name="Mass m")
        k_linear = parse_float(self._k_var.get(), name="Linear k")
        if k_linear < 0:
            raise ValueError("Linear k must be non-negative.")

        t_min = parse_float(self._t_min_var.get(), name="tₘᵢₙ")
        t_max = parse_float(self._t_max_var.get(), name="tₘₐₓ")
        dt = parse_positive_float(self._dt_var.get(), name="Δt")
        if t_max <= t_min:
            raise ValueError("tₘₐₓ must be greater than tₘᵢₙ.")

        selected_terms = {
            self._optional_terms_listbox.get(i)
            for i in self._optional_terms_listbox.curselection()
        }
        alpha, beta, high_coeff, high_power = resolve_optional_membrane_terms(
            selected_terms,
            alpha_text=self._alpha_var.get(),
            beta_text=self._beta_var.get(),
            high_coeff_text=self._high_coeff_var.get(),
            high_power_text=self._high_power_var.get(),
        )

        shape = self._ic_shape_var.get()
        amp = parse_float(self._amp_var.get(), name="Amplitude")
        sigma = parse_positive_float(self._sigma_var.get(), name="σ")
        center_x = parse_float(self._center_x_var.get(), name="Center x₀")
        center_y = parse_float(self._center_y_var.get(), name="Center y₀")
        mode_x = parse_positive_int(self._mode_x_var.get(), name="Mode nₓ")
        mode_y = parse_positive_int(self._mode_y_var.get(), name="Mode nᵧ")
        seed = parse_int(self._seed_var.get(), name="Random seed")

        custom_fn = None
        if shape == "custom":
            custom_fn = compile_scalar_expression(
                self._custom_expr_var.get(),
                variables=("x", "y"),
            )

        u0 = build_initial_displacement(
            nx=nx,
            ny=ny,
            shape=shape,
            amplitude=amp,
            sigma=sigma,
            mode_x=mode_x,
            mode_y=mode_y,
            center_x=center_x,
            center_y=center_y,
            custom_fn=custom_fn,
            random_seed=seed,
            boundary=boundary,
        )
        v0 = np.zeros_like(u0)
        return {
            "u0": u0,
            "v0": v0,
            "t_min": t_min,
            "t_max": t_max,
            "dt": dt,
            "mass": mass,
            "k_linear": k_linear,
            "boundary": boundary,
            "integrator": integrator,
            "alpha": alpha,
            "beta": beta,
            "high_order_coeff": high_coeff,
            "high_order_power": high_power,
        }

    def _on_solve(self) -> None:
        try:
            params = self._collect_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=self.win)
            return

        self.win.destroy()

        def _task():
            return solve_membrane_2d(**params)

        def _on_success(result) -> None:
            from complex_problems.membrane_2d.result_dialog import Membrane2DResultDialog

            Membrane2DResultDialog(self.parent, result=result)

        run_solver_with_loading(
            parent=self.parent,
            message="Solving 2D nonlinear membrane...",
            task=_task,
            on_success=_on_success,
        )
