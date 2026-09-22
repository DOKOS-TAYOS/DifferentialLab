"""UI dialog for configuring time-dependent Schrödinger simulations."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
from typing import cast

from complex_problems.common import (
    add_how_to_config_section,
    compile_scalar_expression,
    parse_float,
    parse_positive_float,
    parse_positive_int,
)
from complex_problems.common.dialog_ui import (
    AdvancedDialogShell,
    AdvancedDialogSize,
    make_labeled_combo,
    make_labeled_entry,
    make_labeled_spinbox,
    run_solver_dialog,
)
from complex_problems.schrodinger_td.solver import solve_schrodinger_td
from config import get_env_from_schema
from frontend.performance_guard import assess_schrodinger_request, confirm_performance_advisory
from frontend.theme import get_font
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import make_modal

_DIMENSIONS = ("1D", "2D")
_BOUNDARIES = ("periodic", "absorbing")
_POTENTIALS = (
    "free",
    "harmonic",
    "square_well",
    "barrier",
    "double_well",
    "lattice",
    "custom",
)
_PACKETS = ("gaussian", "superposition", "custom")


def schrodinger_visible_groups(
    *, potential: str, packet: str, dimension: str, boundary: str
) -> frozenset[str]:
    """Return the conditional control groups relevant to a TDSE configuration."""
    groups = {f"potential:{potential}", f"packet:{packet}"}
    if dimension == "2D":
        groups.add("dimension:2d")
    if boundary == "absorbing":
        groups.add("boundary:absorbing")
    return frozenset(groups)


class SchrodingerTDDialog:
    """Configuration dialog for TDSE in 1D and 2D."""

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Schrodinger Time Evolution (1D/2D)")
        self._build_ui()
        self._shell.finish(AdvancedDialogSize(940, 760, 680, 520))
        make_modal(self.win, parent)
        self._initial_focus.focus_set()

    def _build_ui(self) -> None:
        pad = int(get_env_from_schema("UI_PADDING"))
        self._shell = AdvancedDialogShell(
            self.win,
            title="Schrödinger Time Evolution",
            description=(
                "Use a split-operator spectral solver for TDSE in 1D or 2D. "
                "Periodic boundaries support conservation checks; absorbing edges model "
                "open-domain behavior."
            ),
            pad=pad,
        )
        body = self._shell.body

        add_how_to_config_section(
            body,
            self._shell.scroll,
            problem_id="schrodinger_td",
            pad=pad,
            wraplength=800,
        )

        domain = ttk.LabelFrame(body, text="Domain and grid", padding=pad)
        domain.pack(fill=tk.X, pady=(0, pad))
        row = ttk.Frame(domain)
        row.pack(fill=tk.X, pady=pad // 2)
        self._dimension_var = tk.StringVar(value="1D")
        self._boundary_var = tk.StringVar(value="periodic")
        dim_combo = make_labeled_combo(row, "Dimension", self._dimension_var, _DIMENSIONS, width=8)
        self._initial_focus = dim_combo
        bnd_combo = make_labeled_combo(row, "Boundary", self._boundary_var, _BOUNDARIES, width=10)
        dim_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_visibility())
        bnd_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_visibility())

        row = ttk.Frame(domain)
        row.pack(fill=tk.X, pady=pad // 2)
        self._x_min_var = tk.StringVar(value="-12.0")
        self._x_max_var = tk.StringVar(value="12.0")
        self._nx_var = tk.StringVar(value="256")
        make_labeled_entry(row, "xₘᵢₙ", self._x_min_var, width=10)
        make_labeled_entry(row, "xₘₐₓ", self._x_max_var, width=10)
        make_labeled_spinbox(row, "Nₓ", self._nx_var, from_=32, to=16384, width=8)

        self._y_domain_row = ttk.Frame(domain)
        self._y_domain_row.pack(fill=tk.X, pady=pad // 2)
        self._y_min_var = tk.StringVar(value="-12.0")
        self._y_max_var = tk.StringVar(value="12.0")
        self._ny_var = tk.StringVar(value="128")
        make_labeled_entry(self._y_domain_row, "yₘᵢₙ", self._y_min_var, width=10)
        make_labeled_entry(self._y_domain_row, "yₘₐₓ", self._y_max_var, width=10)
        make_labeled_spinbox(self._y_domain_row, "Nᵧ", self._ny_var, from_=32, to=16384, width=8)

        integration = ttk.LabelFrame(body, text="Time evolution", padding=pad)
        integration.pack(fill=tk.X, pady=(0, pad))
        row = ttk.Frame(integration)
        row.pack(fill=tk.X, pady=pad // 2)
        self._t_min_var = tk.StringVar(value="0.0")
        self._t_max_var = tk.StringVar(value="8.0")
        self._dt_var = tk.StringVar(value="0.002")
        make_labeled_entry(row, "tₘᵢₙ", self._t_min_var, width=10)
        make_labeled_entry(row, "tₘₐₓ", self._t_max_var, width=10)
        make_labeled_entry(row, "Δt", self._dt_var, width=10)

        row = ttk.Frame(integration)
        row.pack(fill=tk.X, pady=pad // 2)
        self._hbar_var = tk.StringVar(value="1.0")
        self._mass_var = tk.StringVar(value="1.0")
        make_labeled_entry(row, "ħ", self._hbar_var, width=10)
        make_labeled_entry(row, "Mass m", self._mass_var, width=10)

        self._absorb_row = ttk.Frame(integration)
        self._absorb_row.pack(fill=tk.X, pady=pad // 2)
        self._absorb_ratio_var = tk.StringVar(value="0.10")
        self._absorb_strength_var = tk.StringVar(value="1.0")
        make_labeled_entry(self._absorb_row, "Absorb ratio", self._absorb_ratio_var, width=9)
        make_labeled_entry(self._absorb_row, "Absorb strength", self._absorb_strength_var, width=11)
        ToolTip(
            self._absorb_row,
            "Only used for absorbing boundaries. Ratio should be between 0 and 0.49.",
        )

        potential = ttk.LabelFrame(body, text="Potential", padding=pad)
        potential.pack(fill=tk.X, pady=(0, pad))
        row = ttk.Frame(potential)
        row.pack(fill=tk.X, pady=pad // 2)
        self._potential_var = tk.StringVar(value="free")
        pot_combo = make_labeled_combo(row, "Type", self._potential_var, _POTENTIALS, width=14)
        pot_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_visibility())
        ToolTip(
            pot_combo,
            "free, harmonic, square_well, barrier, double_well, lattice, or custom expression.",
        )

        self._omega_var = tk.StringVar(value="1.0")
        self._v0_var = tk.StringVar(value="5.0")
        self._width_var = tk.StringVar(value="2.0")
        self._barrier_sigma_var = tk.StringVar(value="0.4")
        self._lattice_k_var = tk.StringVar(value="2.0")
        self._a_dw_var = tk.StringVar(value="1.0")
        self._b_dw_var = tk.StringVar(value="1.0")
        self._potential_rows: dict[str, ttk.Frame] = {}
        self._potential_rows["harmonic"] = ttk.Frame(potential)
        make_labeled_entry(self._potential_rows["harmonic"], "ω", self._omega_var, width=8)
        self._potential_rows["square_well"] = ttk.Frame(potential)
        make_labeled_entry(self._potential_rows["square_well"], "V₀", self._v0_var, width=8)
        make_labeled_entry(self._potential_rows["square_well"], "Width", self._width_var, width=8)
        self._potential_rows["barrier"] = ttk.Frame(potential)
        make_labeled_entry(self._potential_rows["barrier"], "V₀", self._v0_var, width=8)
        make_labeled_entry(
            self._potential_rows["barrier"], "Barrier σ", self._barrier_sigma_var, width=9
        )
        self._potential_rows["double_well"] = ttk.Frame(potential)
        make_labeled_entry(self._potential_rows["double_well"], "a", self._a_dw_var, width=10)
        make_labeled_entry(self._potential_rows["double_well"], "b", self._b_dw_var, width=10)
        self._potential_rows["lattice"] = ttk.Frame(potential)
        make_labeled_entry(self._potential_rows["lattice"], "V₀", self._v0_var, width=8)
        make_labeled_entry(
            self._potential_rows["lattice"], "Lattice k", self._lattice_k_var, width=8
        )

        self._custom_potential_row = ttk.Frame(potential)
        self._potential_rows["custom"] = self._custom_potential_row
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

        packet = ttk.LabelFrame(body, text="Initial wave packet", padding=pad)
        packet.pack(fill=tk.X)
        row = ttk.Frame(packet)
        row.pack(fill=tk.X, pady=pad // 2)
        self._packet_var = tk.StringVar(value="gaussian")
        packet_combo = make_labeled_combo(row, "Type", self._packet_var, _PACKETS, width=14)
        packet_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_visibility())

        self._packet_shape_row = ttk.Frame(packet)
        row = self._packet_shape_row
        row.pack(fill=tk.X, pady=pad // 2)
        self._sigma_var = tk.StringVar(value="0.8")
        self._x0_var = tk.StringVar(value="-3.0")
        self._k0x_var = tk.StringVar(value="2.0")
        self._separation_var = tk.StringVar(value="2.0")
        make_labeled_entry(row, "σ", self._sigma_var, width=8)
        make_labeled_entry(row, "x₀", self._x0_var, width=8)
        self._separation_frame = ttk.Frame(row)
        self._separation_frame.pack(side=tk.LEFT)
        make_labeled_entry(self._separation_frame, "Separation", self._separation_var, width=9)

        self._packet_momentum_row = ttk.Frame(packet)
        self._packet_momentum_row.pack(fill=tk.X, pady=pad // 2)
        make_labeled_entry(self._packet_momentum_row, "k₀x", self._k0x_var, width=8)

        self._packet_y_row = ttk.Frame(packet)
        self._packet_y_row.pack(fill=tk.X, pady=pad // 2)
        self._y0_var = tk.StringVar(value="0.0")
        self._k0y_var = tk.StringVar(value="0.0")
        self._packet_y_position_frame = ttk.Frame(self._packet_y_row)
        self._packet_y_position_frame.pack(side=tk.LEFT)
        make_labeled_entry(self._packet_y_position_frame, "y₀", self._y0_var, width=8)
        make_labeled_entry(self._packet_y_row, "k₀y", self._k0y_var, width=8)

        self._custom_packet_row = ttk.Frame(packet)
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
            "Custom real amplitude profile. Momentum comes from k0x and k0y.",
        )

        self._shell.add_footer_button("Close", self.win.destroy)
        self._shell.add_footer_button("Solve", self._on_solve, primary=True)

        self._update_visibility()

    def _update_visibility(self) -> None:
        groups = schrodinger_visible_groups(
            potential=self._potential_var.get(),
            packet=self._packet_var.get(),
            dimension=self._dimension_var.get(),
            boundary=self._boundary_var.get(),
        )
        if "dimension:2d" in groups:
            self._y_domain_row.pack(fill=tk.X, pady=4)
            self._packet_y_row.pack(fill=tk.X, pady=4)
        else:
            self._y_domain_row.pack_forget()
            self._packet_y_row.pack_forget()

        if "boundary:absorbing" in groups:
            self._absorb_row.pack(fill=tk.X, pady=4)
        else:
            self._absorb_row.pack_forget()

        for row in self._potential_rows.values():
            row.pack_forget()
        potential_row = self._potential_rows.get(self._potential_var.get())
        if potential_row is not None:
            potential_row.pack(fill=tk.X, pady=4)

        if "packet:custom" in groups:
            self._packet_shape_row.pack_forget()
            self._packet_y_position_frame.pack_forget()
            self._custom_packet_row.pack(
                fill=tk.X,
                pady=4,
                before=self._packet_momentum_row,
            )
        else:
            self._custom_packet_row.pack_forget()
            self._packet_shape_row.pack(
                fill=tk.X,
                pady=4,
                before=self._packet_momentum_row,
            )
            y_momentum_label = self._packet_y_row.winfo_children()[-2]
            self._packet_y_position_frame.pack(side=tk.LEFT, before=y_momentum_label)
        if "packet:superposition" in groups:
            self._separation_frame.pack(side=tk.LEFT)
        else:
            self._separation_frame.pack_forget()
        self._shell.refresh()

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

        custom_potential_fn_1d: Callable[[float], float] | None = None
        custom_potential_fn_2d: Callable[[float, float], float] | None = None
        if potential_type == "custom":
            if dimension == 2:
                compiled = compile_scalar_expression(
                    self._custom_potential_var.get(),
                    variables=("x", "y"),
                )

                def _custom_potential_fn_2d(
                    x: float,
                    y: float,
                    _fn: Callable[..., float] = compiled,
                ) -> float:
                    return _fn(x=float(x), y=float(y))

                custom_potential_fn_2d = _custom_potential_fn_2d

            else:
                compiled = compile_scalar_expression(
                    self._custom_potential_var.get(),
                    variables=("x",),
                )

                def _custom_potential_fn_1d(
                    x: float,
                    _fn: Callable[..., float] = compiled,
                ) -> float:
                    return _fn(x=float(x))

                custom_potential_fn_1d = _custom_potential_fn_1d

        custom_packet_fn_1d: Callable[[float], float] | None = None
        custom_packet_fn_2d: Callable[[float, float], float] | None = None
        if packet_type == "custom":
            if dimension == 2:
                compiled = compile_scalar_expression(
                    self._custom_packet_var.get(),
                    variables=("x", "y"),
                )

                def _custom_packet_fn_2d(
                    x: float,
                    y: float,
                    _fn: Callable[..., float] = compiled,
                ) -> float:
                    return _fn(x=float(x), y=float(y))

                custom_packet_fn_2d = _custom_packet_fn_2d

            else:
                compiled = compile_scalar_expression(
                    self._custom_packet_var.get(),
                    variables=("x",),
                )

                def _custom_packet_fn_1d(
                    x: float,
                    _fn: Callable[..., float] = compiled,
                ) -> float:
                    return _fn(x=float(x))

                custom_packet_fn_1d = _custom_packet_fn_1d

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

    def _confirm_heavy_request(self, params: dict[str, object], window: tk.Toplevel) -> bool:
        """Warn before launching unusually large TDSE histories."""
        dimension = cast(int, params["dimension"])
        nx = cast(int, params["nx"])
        ny = cast(int, params["ny"])
        t_min = cast(float, params["t_min"])
        t_max = cast(float, params["t_max"])
        dt = cast(float, params["dt"])
        advisory = assess_schrodinger_request(
            dimension=dimension,
            nx=nx,
            ny=ny,
            t_min=t_min,
            t_max=t_max,
            dt=dt,
        )
        return confirm_performance_advisory(window, advisory)

    def _on_solve(self) -> None:
        from complex_problems.schrodinger_td.result_dialog import (
            SchrodingerTDResultDialog,
        )

        run_solver_dialog(
            parent=self.parent,
            window=self.win,
            collect_inputs=self._collect_inputs,
            solver=solve_schrodinger_td,
            message="Solving Schrodinger time evolution...",
            result_parent=self.parent,
            result_dialog_factory=SchrodingerTDResultDialog,
            confirm_run=self._confirm_heavy_request,
        )
