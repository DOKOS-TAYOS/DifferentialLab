"""UI dialog for configuring 2D aerodynamics simulations."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import cast

from complex_problems.aerodynamics_2d.solver import solve_aerodynamics_2d
from complex_problems.common import (
    add_how_to_config_section,
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
from config import get_env_from_schema
from frontend.performance_guard import assess_aerodynamics_request, confirm_performance_advisory
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import make_modal

_APPROX = ("nonlinear_ns", "stokes")
_SHAPES = ("cylinder", "ellipse", "rectangle", "naca0012")


class Aerodynamics2DDialog:
    """Configuration dialog for 2D aerodynamic flow solver."""

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Aerodynamics 2D")
        self._build_ui()
        self._shell.finish(AdvancedDialogSize(960, 760, 680, 520))
        make_modal(self.win, parent)
        self._initial_focus.focus_set()

    def _build_ui(self) -> None:
        pad = int(get_env_from_schema("UI_PADDING"))
        self._shell = AdvancedDialogShell(
            self.win,
            title="Aerodynamics 2D",
            description=(
                "Simulate incompressible flow around immersed bodies with FFT projection. "
                "Choose the full nonlinear Navier–Stokes model or a Stokes approximation."
            ),
            pad=pad,
        )
        body = self._shell.body

        add_how_to_config_section(
            body,
            self._shell.scroll,
            problem_id="aerodynamics_2d",
            pad=pad,
            wraplength=800,
        )

        model = ttk.LabelFrame(body, text="Model", padding=pad)
        model.pack(fill=tk.X, pady=(0, pad))
        row = ttk.Frame(model)
        row.pack(fill=tk.X, pady=pad // 2)
        self._approx_var = tk.StringVar(value="nonlinear_ns")
        self._shape_var = tk.StringVar(value="cylinder")
        self._initial_focus = make_labeled_combo(
            row, "Approximation", self._approx_var, _APPROX, width=13
        )
        row = ttk.Frame(model)
        row.pack(fill=tk.X, pady=pad // 2)
        make_labeled_combo(row, "Obstacle shape", self._shape_var, _SHAPES, width=14)

        domain = ttk.LabelFrame(body, text="Domain and grid", padding=pad)
        domain.pack(fill=tk.X, pady=(0, pad))
        row = ttk.Frame(domain)
        row.pack(fill=tk.X, pady=pad // 2)
        self._nx_var = tk.StringVar(value="96")
        self._ny_var = tk.StringVar(value="64")
        self._lx_var = tk.StringVar(value="4.0")
        self._ly_var = tk.StringVar(value="2.0")
        make_labeled_spinbox(row, "Nₓ", self._nx_var, from_=16, to=8192, width=8)
        make_labeled_spinbox(row, "Nᵧ", self._ny_var, from_=16, to=8192, width=8)
        row = ttk.Frame(domain)
        row.pack(fill=tk.X, pady=pad // 2)
        make_labeled_entry(row, "Lₓ", self._lx_var, width=8)
        make_labeled_entry(row, "Lᵧ", self._ly_var, width=8)

        row = ttk.Frame(domain)
        row.pack(fill=tk.X, pady=pad // 2)
        self._t_max_var = tk.StringVar(value="2.0")
        self._dt_var = tk.StringVar(value="0.002")
        self._sample_every_var = tk.StringVar(value="10")
        make_labeled_entry(row, "tₘₐₓ", self._t_max_var, width=8)
        make_labeled_entry(row, "Δt", self._dt_var, width=8)
        make_labeled_entry(row, "Sample every", self._sample_every_var, width=10)
        ToolTip(row, "Lower Δt and/or Sample every values store more animation frames.")

        physics = ttk.LabelFrame(body, text="Flow properties", padding=pad)
        physics.pack(fill=tk.X, pady=(0, pad))
        row = ttk.Frame(physics)
        row.pack(fill=tk.X, pady=pad // 2)
        self._rho_var = tk.StringVar(value="1.0")
        self._nu_var = tk.StringVar(value="0.01")
        self._u_inf_var = tk.StringVar(value="1.0")
        self._penal_var = tk.StringVar(value="0.005")
        make_labeled_entry(row, "ρ", self._rho_var, width=8)
        make_labeled_entry(row, "ν", self._nu_var, width=8)
        row = ttk.Frame(physics)
        row.pack(fill=tk.X, pady=pad // 2)
        make_labeled_entry(row, "U∞", self._u_inf_var, width=8)
        make_labeled_entry(row, "Penalization", self._penal_var, width=10)

        geometry = ttk.LabelFrame(body, text="Obstacle geometry", padding=pad)
        geometry.pack(fill=tk.X)
        row = ttk.Frame(geometry)
        row.pack(fill=tk.X, pady=pad // 2)
        self._center_x_var = tk.StringVar(value="1.3")
        self._center_y_var = tk.StringVar(value="1.0")
        self._size_x_var = tk.StringVar(value="0.30")
        self._size_y_var = tk.StringVar(value="0.30")
        self._attack_deg_var = tk.StringVar(value="0.0")
        make_labeled_entry(row, "Center x", self._center_x_var, width=8)
        make_labeled_entry(row, "Center y", self._center_y_var, width=8)
        row = ttk.Frame(geometry)
        row.pack(fill=tk.X, pady=pad // 2)
        make_labeled_entry(row, "Size x", self._size_x_var, width=8)
        make_labeled_entry(row, "Size y", self._size_y_var, width=8)
        row = ttk.Frame(geometry)
        row.pack(fill=tk.X, pady=pad // 2)
        make_labeled_entry(row, "Attack (deg)", self._attack_deg_var, width=10)
        ToolTip(
            row,
            "For naca0012: size x = chord, size y = thickness ratio (e.g. 0.12).",
        )

        self._shell.add_footer_button("Close", self.win.destroy)
        self._shell.add_footer_button("Solve", self._on_solve, primary=True)

    def _collect_inputs(self) -> dict[str, object]:
        nx = parse_positive_int(self._nx_var.get(), name="Nₓ", min_value=16)
        ny = parse_positive_int(self._ny_var.get(), name="Nᵧ", min_value=16)
        lx = parse_positive_float(self._lx_var.get(), name="Lₓ")
        ly = parse_positive_float(self._ly_var.get(), name="Lᵧ")

        t_max = parse_positive_float(self._t_max_var.get(), name="tₘₐₓ")
        dt = parse_positive_float(self._dt_var.get(), name="Δt")
        sample_every = parse_positive_int(self._sample_every_var.get(), name="Sample every")

        rho = parse_positive_float(self._rho_var.get(), name="ρ")
        nu = parse_positive_float(self._nu_var.get(), name="ν")
        u_inf = parse_positive_float(self._u_inf_var.get(), name="U∞")
        penalization = parse_positive_float(self._penal_var.get(), name="Penalization")

        center_x = parse_float(self._center_x_var.get(), name="Center x")
        center_y = parse_float(self._center_y_var.get(), name="Center y")
        size_x = parse_positive_float(self._size_x_var.get(), name="Size x")
        size_y = parse_positive_float(self._size_y_var.get(), name="Size y")
        attack_deg = parse_float(self._attack_deg_var.get(), name="Attack (deg)")

        return {
            "approximation": self._approx_var.get(),
            "nx": nx,
            "ny": ny,
            "lx": lx,
            "ly": ly,
            "t_max": t_max,
            "dt": dt,
            "sample_every": sample_every,
            "rho": rho,
            "nu": nu,
            "u_inf": u_inf,
            "penalization": penalization,
            "obstacle_shape": self._shape_var.get(),
            "obstacle_center_x": center_x,
            "obstacle_center_y": center_y,
            "obstacle_size_x": size_x,
            "obstacle_size_y": size_y,
            "obstacle_attack_deg": attack_deg,
        }

    def _confirm_heavy_request(self, params: dict[str, object], window: tk.Toplevel) -> bool:
        """Warn before launching unusually large aerodynamics histories."""
        nx = cast(int, params["nx"])
        ny = cast(int, params["ny"])
        t_max = cast(float, params["t_max"])
        dt = cast(float, params["dt"])
        sample_every = cast(int, params["sample_every"])
        advisory = assess_aerodynamics_request(
            nx=nx,
            ny=ny,
            t_max=t_max,
            dt=dt,
            sample_every=sample_every,
        )
        return confirm_performance_advisory(window, advisory)

    def _on_solve(self) -> None:
        from complex_problems.aerodynamics_2d.result_dialog import Aerodynamics2DResultDialog

        run_solver_dialog(
            parent=self.parent,
            window=self.win,
            collect_inputs=self._collect_inputs,
            solver=solve_aerodynamics_2d,
            message="Solving 2D aerodynamics...",
            result_parent=self.parent,
            result_dialog_factory=Aerodynamics2DResultDialog,
            confirm_run=self._confirm_heavy_request,
        )
