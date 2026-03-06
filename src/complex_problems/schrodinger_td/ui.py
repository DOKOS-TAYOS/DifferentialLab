"""UI dialog for configuring time-dependent Schrödinger simulations."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from complex_problems.common import (
    add_how_to_config_section,
    compile_scalar_expression,
    parse_float,
    parse_positive_float,
    parse_positive_int,
    run_solver_with_loading,
)
from complex_problems.schrodinger_td.solver import solve_schrodinger_td
from config import get_env_from_schema
from frontend.theme import get_font
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import fit_and_center, make_modal

_DIMENSIONS = ("1D", "2D")
_BOUNDARIES = ("periodic", "absorbing")
_POTENTIALS = ("free", "harmonic", "square_well", "barrier", "double_well", "lattice", "custom")
_PACKETS = ("gaussian", "superposition", "custom")


class SchrodingerTDDialog:
    """Configuration dialog for TDSE in 1D and 2D."""

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Schrodinger TD (1D/2D)")
        self.win.configure(bg=get_env_from_schema("UI_BACKGROUND"))
        self._build_ui()
        fit_and_center(self.win, min_width=1040, min_height=780, padding=32, resizable=True)
        self.win.minsize(960, 700)
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

        ttk.Label(body, text="Schrodinger Time Evolution", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            body,
            text=(
                "Split-operator spectral solver for TDSE in 1D or 2D.\n"
                "Use periodic boundaries for conservation checks or "
                "absorbing edges for open-domain behavior."
            ),
            style="Small.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, pad))

        add_how_to_config_section(
            body,
            scroll,
            problem_id="schrodinger_td",
            pad=pad,
            wraplength=800,
        )

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._dimension_var = tk.StringVar(value="1D")
        self._boundary_var = tk.StringVar(value="periodic")
        dim_combo = self._make_combo(row, "Dimension", self._dimension_var, _DIMENSIONS, width=8)
        bnd_combo = self._make_combo(row, "Boundary", self._boundary_var, _BOUNDARIES, width=10)
        dim_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_visibility())
        bnd_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_visibility())

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._x_min_var = tk.StringVar(value="-12.0")
        self._x_max_var = tk.StringVar(value="12.0")
        self._nx_var = tk.StringVar(value="256")
        self._make_entry(row, "xₘᵢₙ", self._x_min_var, width=10)
        self._make_entry(row, "xₘₐₓ", self._x_max_var, width=10)
        self._make_spinbox(row, "Nₓ", self._nx_var, from_=32, to=16384, width=8)

        self._y_domain_row = ttk.Frame(body)
        self._y_domain_row.pack(fill=tk.X, pady=pad // 2)
        self._y_min_var = tk.StringVar(value="-12.0")
        self._y_max_var = tk.StringVar(value="12.0")
        self._ny_var = tk.StringVar(value="128")
        self._make_entry(self._y_domain_row, "yₘᵢₙ", self._y_min_var, width=10)
        self._make_entry(self._y_domain_row, "yₘₐₓ", self._y_max_var, width=10)
        self._make_spinbox(self._y_domain_row, "Nᵧ", self._ny_var, from_=32, to=16384, width=8)

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._t_min_var = tk.StringVar(value="0.0")
        self._t_max_var = tk.StringVar(value="8.0")
        self._dt_var = tk.StringVar(value="0.002")
        self._make_entry(row, "tₘᵢₙ", self._t_min_var, width=10)
        self._make_entry(row, "tₘₐₓ", self._t_max_var, width=10)
        self._make_entry(row, "Δt", self._dt_var, width=10)

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._hbar_var = tk.StringVar(value="1.0")
        self._mass_var = tk.StringVar(value="1.0")
        self._make_entry(row, "ħ", self._hbar_var, width=10)
        self._make_entry(row, "Mass m", self._mass_var, width=10)

        self._absorb_row = ttk.Frame(body)
        self._absorb_row.pack(fill=tk.X, pady=pad // 2)
        self._absorb_ratio_var = tk.StringVar(value="0.10")
        self._absorb_strength_var = tk.StringVar(value="1.0")
        self._make_entry(self._absorb_row, "Absorb ratio", self._absorb_ratio_var, width=9)
        self._make_entry(self._absorb_row, "Absorb strength", self._absorb_strength_var, width=11)
        ToolTip(
            self._absorb_row,
            "Only used for absorbing boundaries. Ratio should be between 0 and 0.49.",
        )

        ttk.Separator(body).pack(fill=tk.X, pady=pad)
        ttk.Label(body, text="Potential", style="Small.TLabel").pack(anchor=tk.W)

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._potential_var = tk.StringVar(value="free")
        pot_combo = self._make_combo(row, "Type", self._potential_var, _POTENTIALS, width=14)
        pot_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_visibility())
        ToolTip(
            pot_combo,
            "free, harmonic, square_well, barrier, double_well, lattice, or custom expression.",
        )

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._omega_var = tk.StringVar(value="1.0")
        self._v0_var = tk.StringVar(value="5.0")
        self._width_var = tk.StringVar(value="2.0")
        self._barrier_sigma_var = tk.StringVar(value="0.4")
        self._make_entry(row, "ω", self._omega_var, width=8)
        self._make_entry(row, "V₀", self._v0_var, width=8)
        self._make_entry(row, "Width", self._width_var, width=8)
        self._make_entry(row, "Barrier σ", self._barrier_sigma_var, width=9)

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._lattice_k_var = tk.StringVar(value="2.0")
        self._a_dw_var = tk.StringVar(value="1.0")
        self._b_dw_var = tk.StringVar(value="1.0")
        self._make_entry(row, "Lattice k", self._lattice_k_var, width=8)
        self._make_entry(row, "a (double-well)", self._a_dw_var, width=12)
        self._make_entry(row, "b (double-well)", self._b_dw_var, width=12)

        self._custom_potential_row = ttk.Frame(body)
        self._custom_potential_row.pack(fill=tk.X, pady=pad // 2)
        ttk.Label(self._custom_potential_row, text="V_custom =").pack(side=tk.LEFT, padx=(0, 4))
        self._custom_potential_var = tk.StringVar(value="0.5*(x**2 + y**2)")
        self._custom_potential_entry = ttk.Entry(
            self._custom_potential_row,
            textvariable=self._custom_potential_var,
            width=56,
            font=get_font(),
        )
        self._custom_potential_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Separator(body).pack(fill=tk.X, pady=pad)
        ttk.Label(body, text="Initial packet", style="Small.TLabel").pack(anchor=tk.W)

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._packet_var = tk.StringVar(value="gaussian")
        packet_combo = self._make_combo(row, "Type", self._packet_var, _PACKETS, width=14)
        packet_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_visibility())

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._sigma_var = tk.StringVar(value="0.8")
        self._x0_var = tk.StringVar(value="-3.0")
        self._k0x_var = tk.StringVar(value="2.0")
        self._separation_var = tk.StringVar(value="2.0")
        self._make_entry(row, "σ", self._sigma_var, width=8)
        self._make_entry(row, "x₀", self._x0_var, width=8)
        self._make_entry(row, "k₀x", self._k0x_var, width=8)
        self._make_entry(row, "Separation", self._separation_var, width=9)

        self._packet_y_row = ttk.Frame(body)
        self._packet_y_row.pack(fill=tk.X, pady=pad // 2)
        self._y0_var = tk.StringVar(value="0.0")
        self._k0y_var = tk.StringVar(value="0.0")
        self._make_entry(self._packet_y_row, "y₀", self._y0_var, width=8)
        self._make_entry(self._packet_y_row, "k₀y", self._k0y_var, width=8)

        self._custom_packet_row = ttk.Frame(body)
        self._custom_packet_row.pack(fill=tk.X, pady=pad // 2)
        ttk.Label(self._custom_packet_row, text="ψ_custom amplitude =").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        self._custom_packet_var = tk.StringVar(value="exp(-(x**2 + y**2))")
        self._custom_packet_entry = ttk.Entry(
            self._custom_packet_row,
            textvariable=self._custom_packet_var,
            width=46,
            font=get_font(),
        )
        self._custom_packet_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ToolTip(
            self._custom_packet_entry,
            "Custom real amplitude profile. The phase comes from k0x/k0y.",
        )

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

        self._update_visibility()

        scroll.bind_new_children()

    def _make_entry(
        self, parent: ttk.Frame, label: str, var: tk.StringVar, *, width: int = 10
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

    def _update_visibility(self) -> None:
        is_2d = self._dimension_var.get() == "2D"
        if is_2d:
            self._y_domain_row.pack(fill=tk.X, pady=4, before=self._btn_row)
            self._packet_y_row.pack(fill=tk.X, pady=4, before=self._btn_row)
        else:
            self._y_domain_row.pack_forget()
            self._packet_y_row.pack_forget()

        if self._boundary_var.get() == "absorbing":
            self._absorb_row.pack(fill=tk.X, pady=4, before=self._btn_row)
        else:
            self._absorb_row.pack_forget()

        if self._potential_var.get() == "custom":
            self._custom_potential_row.pack(fill=tk.X, pady=4, before=self._btn_row)
        else:
            self._custom_potential_row.pack_forget()

        if self._packet_var.get() == "custom":
            self._custom_packet_row.pack(fill=tk.X, pady=4, before=self._btn_row)
        else:
            self._custom_packet_row.pack_forget()

    def _collect_inputs(self) -> dict[str, object]:
        dimension = 2 if self._dimension_var.get() == "2D" else 1
        boundary = self._boundary_var.get().strip().lower()
        potential_type = self._potential_var.get().strip().lower()
        packet_type = self._packet_var.get().strip().lower()

        x_min = parse_float(self._x_min_var.get(), name="xₘᵢₙ")
        x_max = parse_float(self._x_max_var.get(), name="xₘₐₓ")
        if x_max <= x_min:
            raise ValueError("xₘₐₓ must be greater than xₘᵢₙ.")
        nx = parse_positive_int(self._nx_var.get(), name="Nₓ", min_value=32)

        y_min = parse_float(self._y_min_var.get(), name="yₘᵢₙ")
        y_max = parse_float(self._y_max_var.get(), name="yₘₐₓ")
        ny = parse_positive_int(self._ny_var.get(), name="Nᵧ", min_value=32)
        if dimension == 2 and y_max <= y_min:
            raise ValueError("yₘₐₓ must be greater than yₘᵢₙ.")

        t_min = parse_float(self._t_min_var.get(), name="tₘᵢₙ")
        t_max = parse_float(self._t_max_var.get(), name="tₘₐₓ")
        if t_max <= t_min:
            raise ValueError("tₘₐₓ must be greater than tₘᵢₙ.")
        dt = parse_positive_float(self._dt_var.get(), name="Δt")

        hbar = parse_positive_float(self._hbar_var.get(), name="ħ")
        mass = parse_positive_float(self._mass_var.get(), name="Mass m")

        absorb_ratio = parse_float(self._absorb_ratio_var.get(), name="Absorb ratio")
        if not (0.0 <= absorb_ratio < 0.5):
            raise ValueError("Absorb ratio must be in [0, 0.5).")
        absorb_strength = parse_positive_float(
            self._absorb_strength_var.get(),
            name="Absorb strength",
        )

        omega = parse_positive_float(self._omega_var.get(), name="ω")
        v0 = parse_float(self._v0_var.get(), name="V₀")
        width = parse_positive_float(self._width_var.get(), name="Width")
        barrier_sigma = parse_positive_float(self._barrier_sigma_var.get(), name="Barrier σ")
        lattice_k = parse_positive_float(self._lattice_k_var.get(), name="Lattice k")
        a_dw = parse_positive_float(self._a_dw_var.get(), name="a (double-well)")
        b_dw = parse_positive_float(self._b_dw_var.get(), name="b (double-well)")

        sigma = parse_positive_float(self._sigma_var.get(), name="σ")
        x0 = parse_float(self._x0_var.get(), name="x₀")
        y0 = parse_float(self._y0_var.get(), name="y₀")
        k0x = parse_float(self._k0x_var.get(), name="k₀x")
        k0y = parse_float(self._k0y_var.get(), name="k₀y")
        separation = parse_positive_float(self._separation_var.get(), name="Separation")

        custom_potential_fn_1d = None
        custom_potential_fn_2d = None
        if potential_type == "custom":
            if dimension == 2:
                compiled = compile_scalar_expression(
                    self._custom_potential_var.get(),
                    variables=("x", "y"),
                )

                def custom_potential_fn_2d(
                    x: float,
                    y: float,
                    _fn=compiled,
                ) -> float:
                    return _fn(x=float(x), y=float(y))

            else:
                compiled = compile_scalar_expression(
                    self._custom_potential_var.get(),
                    variables=("x",),
                )

                def custom_potential_fn_1d(
                    x: float,
                    _fn=compiled,
                ) -> float:
                    return _fn(x=float(x))

        custom_packet_fn_1d = None
        custom_packet_fn_2d = None
        if packet_type == "custom":
            if dimension == 2:
                compiled = compile_scalar_expression(
                    self._custom_packet_var.get(),
                    variables=("x", "y"),
                )

                def custom_packet_fn_2d(
                    x: float,
                    y: float,
                    _fn=compiled,
                ) -> float:
                    return _fn(x=float(x), y=float(y))

            else:
                compiled = compile_scalar_expression(
                    self._custom_packet_var.get(),
                    variables=("x",),
                )

                def custom_packet_fn_1d(
                    x: float,
                    _fn=compiled,
                ) -> float:
                    return _fn(x=float(x))

        return {
            "dimension": dimension,
            "x_min": x_min,
            "x_max": x_max,
            "nx": nx,
            "y_min": y_min,
            "y_max": y_max,
            "ny": ny,
            "t_min": t_min,
            "t_max": t_max,
            "dt": dt,
            "hbar": hbar,
            "mass": mass,
            "boundary": boundary,
            "absorb_ratio": absorb_ratio,
            "absorb_strength": absorb_strength,
            "potential_type": potential_type,
            "omega": omega,
            "v0": v0,
            "width": width,
            "barrier_sigma": barrier_sigma,
            "lattice_k": lattice_k,
            "a_dw": a_dw,
            "b_dw": b_dw,
            "packet_type": packet_type,
            "sigma": sigma,
            "x0": x0,
            "y0": y0,
            "k0x": k0x,
            "k0y": k0y,
            "separation": separation,
            "custom_potential_fn_1d": custom_potential_fn_1d,
            "custom_potential_fn_2d": custom_potential_fn_2d,
            "custom_packet_fn_1d": custom_packet_fn_1d,
            "custom_packet_fn_2d": custom_packet_fn_2d,
        }

    def _on_solve(self) -> None:
        try:
            params = self._collect_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=self.win)
            return

        self.win.destroy()

        def _task():
            return solve_schrodinger_td(**params)

        def _on_success(result) -> None:
            from complex_problems.schrodinger_td.result_dialog import SchrodingerTDResultDialog

            SchrodingerTDResultDialog(self.parent, result=result)

        run_solver_with_loading(
            parent=self.parent,
            message="Solving Schrodinger TD...",
            task=_task,
            on_success=_on_success,
        )
