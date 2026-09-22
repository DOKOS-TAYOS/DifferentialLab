"""Configuration dialog for the 3D aerodynamics problem."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import cast

from complex_problems.aerodynamics_3d.solver import solve_aerodynamics_3d
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
from frontend.performance_guard import (
    assess_aerodynamics_3d_request,
    confirm_performance_advisory,
)
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import make_modal

_APPROXIMATIONS = ("nonlinear_ns", "stokes")
_SHAPES = ("sphere", "ellipsoid", "box", "naca0012_wing")


class Aerodynamics3DDialog:
    """Scrollable, progressive-disclosure configuration dialog."""

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Aerodynamics 3D")
        self._build_ui()
        self._shell.finish(AdvancedDialogSize(980, 800, 700, 540))
        make_modal(self.win, parent)
        self._initial_focus.focus_set()

    def _build_ui(self) -> None:
        pad = int(get_env_from_schema("UI_PADDING"))
        self._shell = AdvancedDialogShell(
            self.win,
            title="Aerodynamics 3D",
            description=(
                "Solve incompressible 3D structured-grid flow in a periodic Cartesian domain "
                "with an immersed/penalized obstacle. This is a lightweight educational and "
                "scientific solver, not an industrial CFD replacement."
            ),
            pad=pad,
        )
        body = self._shell.body
        add_how_to_config_section(
            body, self._shell.scroll, problem_id="aerodynamics_3d", pad=pad, wraplength=820
        )

        model = ttk.LabelFrame(body, text="Model", padding=pad)
        model.pack(fill=tk.X, pady=(0, pad))
        row = ttk.Frame(model)
        row.pack(fill=tk.X, pady=pad // 2)
        self._approx_var = tk.StringVar(value="nonlinear_ns")
        self._shape_var = tk.StringVar(value="sphere")
        self._initial_focus = make_labeled_combo(
            row, "Approximation", self._approx_var, _APPROXIMATIONS, width=15
        )
        row = ttk.Frame(model)
        row.pack(fill=tk.X, pady=pad // 2)
        shape_combo = make_labeled_combo(row, "Obstacle shape", self._shape_var, _SHAPES, width=18)
        shape_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_geometry())

        domain = ttk.LabelFrame(body, text="Domain and grid", padding=pad)
        domain.pack(fill=tk.X, pady=(0, pad))
        row = ttk.Frame(domain)
        row.pack(fill=tk.X, pady=pad // 2)
        self._nx_var, self._ny_var, self._nz_var = (
            tk.StringVar(value=value) for value in ("48", "32", "32")
        )
        make_labeled_spinbox(row, "Nₓ", self._nx_var, from_=4, to=2048, width=7)
        make_labeled_spinbox(row, "Nᵧ", self._ny_var, from_=4, to=2048, width=7)
        make_labeled_spinbox(row, "N_z", self._nz_var, from_=4, to=2048, width=7)
        row = ttk.Frame(domain)
        row.pack(fill=tk.X, pady=pad // 2)
        self._lx_var, self._ly_var, self._lz_var = (
            tk.StringVar(value=value) for value in ("4.0", "2.0", "2.0")
        )
        make_labeled_entry(row, "Lₓ", self._lx_var, width=7)
        make_labeled_entry(row, "Lᵧ", self._ly_var, width=7)
        make_labeled_entry(row, "L_z", self._lz_var, width=7)
        row = ttk.Frame(domain)
        row.pack(fill=tk.X, pady=pad // 2)
        self._t_max_var, self._dt_var, self._sample_every_var = (
            tk.StringVar(value=value) for value in ("1.0", "0.002", "10")
        )
        make_labeled_entry(row, "tₘₐₓ", self._t_max_var, width=8)
        make_labeled_entry(row, "Δt", self._dt_var, width=8)
        make_labeled_entry(row, "Sample every", self._sample_every_var, width=10)
        ToolTip(row, "Lower Δt and/or Sample every values store more volumetric animation frames.")

        physics = ttk.LabelFrame(body, text="Flow properties", padding=pad)
        physics.pack(fill=tk.X, pady=(0, pad))
        row = ttk.Frame(physics)
        row.pack(fill=tk.X, pady=pad // 2)
        self._rho_var, self._nu_var, self._u_inf_var, self._penal_var = (
            tk.StringVar(value=value) for value in ("1.0", "0.01", "1.0", "0.005")
        )
        make_labeled_entry(row, "ρ", self._rho_var, width=8)
        make_labeled_entry(row, "ν", self._nu_var, width=8)
        make_labeled_entry(row, "U∞", self._u_inf_var, width=8)
        make_labeled_entry(row, "Penalization", self._penal_var, width=10)

        geometry = ttk.LabelFrame(body, text="Obstacle geometry", padding=pad)
        geometry.pack(fill=tk.X)
        row = ttk.Frame(geometry)
        row.pack(fill=tk.X, pady=pad // 2)
        self._center_x_var, self._center_y_var, self._center_z_var = (
            tk.StringVar(value=value) for value in ("1.0", "1.0", "1.0")
        )
        make_labeled_entry(row, "Center x", self._center_x_var, width=8)
        make_labeled_entry(row, "Center y", self._center_y_var, width=8)
        make_labeled_entry(row, "Center z", self._center_z_var, width=8)
        self._geometry_body = ttk.Frame(geometry)
        self._geometry_body.pack(fill=tk.X)
        self._geometry_rows: dict[str, ttk.Frame] = {}
        self._make_shape_rows(pad)
        self._refresh_geometry()
        self._shell.add_footer_button("Close", self.win.destroy)
        self._shell.add_footer_button("Solve", self._on_solve, primary=True)

    def _make_shape_rows(self, pad: int) -> None:
        for shape in _SHAPES:
            frame = ttk.Frame(self._geometry_body)
            self._geometry_rows[shape] = frame
            if shape == "sphere":
                self._diameter_var = tk.StringVar(value="0.4")
                make_labeled_entry(frame, "Diameter", self._diameter_var, width=8)
            elif shape in ("ellipsoid", "box"):
                self._size_x_var = tk.StringVar(value="0.7")
                self._size_y_var = tk.StringVar(value="0.4")
                self._size_z_var = tk.StringVar(value="0.4")
                self._attack_var = tk.StringVar(value="0.0")
                make_labeled_entry(frame, "Size x", self._size_x_var, width=8)
                make_labeled_entry(frame, "Size y", self._size_y_var, width=8)
                make_labeled_entry(frame, "Size z", self._size_z_var, width=8)
                make_labeled_entry(frame, "Attack (deg)", self._attack_var, width=10)
            else:
                self._chord_var = tk.StringVar(value="0.8")
                self._span_var = tk.StringVar(value="0.8")
                self._thickness_var = tk.StringVar(value="0.12")
                self._wing_attack_var = tk.StringVar(value="0.0")
                make_labeled_entry(frame, "Chord", self._chord_var, width=8)
                make_labeled_entry(frame, "Span", self._span_var, width=8)
                make_labeled_entry(frame, "Thickness ratio", self._thickness_var, width=12)
                make_labeled_entry(frame, "Attack (deg)", self._wing_attack_var, width=10)
                ToolTip(frame, "The finite-span NACA 0012 wing has no sweep, taper, or twist.")

    def _refresh_geometry(self) -> None:
        for frame in self._geometry_rows.values():
            frame.pack_forget()
        self._geometry_rows[self._shape_var.get()].pack(fill=tk.X, pady=4)
        self._shell.refresh()

    def _collect_inputs(self) -> dict[str, object]:
        nx = parse_positive_int(self._nx_var.get(), name="Nₓ", min_value=4)
        ny = parse_positive_int(self._ny_var.get(), name="Nᵧ", min_value=4)
        nz = parse_positive_int(self._nz_var.get(), name="N_z", min_value=4)
        values = {
            "lx": parse_positive_float(self._lx_var.get(), name="Lₓ"),
            "ly": parse_positive_float(self._ly_var.get(), name="Lᵧ"),
            "lz": parse_positive_float(self._lz_var.get(), name="L_z"),
            "t_max": parse_positive_float(self._t_max_var.get(), name="tₘₐₓ"),
            "dt": parse_positive_float(self._dt_var.get(), name="Δt"),
            "sample_every": parse_positive_int(self._sample_every_var.get(), name="Sample every"),
            "rho": parse_positive_float(self._rho_var.get(), name="ρ"),
            "nu": parse_positive_float(self._nu_var.get(), name="ν"),
            "u_inf": parse_positive_float(self._u_inf_var.get(), name="U∞"),
            "penalization": parse_positive_float(self._penal_var.get(), name="Penalization"),
            "obstacle_center_x": parse_float(self._center_x_var.get(), name="Center x"),
            "obstacle_center_y": parse_float(self._center_y_var.get(), name="Center y"),
            "obstacle_center_z": parse_float(self._center_z_var.get(), name="Center z"),
        }
        shape = self._shape_var.get()
        values.update(
            {
                "approximation": self._approx_var.get(),
                "obstacle_shape": shape,
                "nx": nx,
                "ny": ny,
                "nz": nz,
            }
        )
        if shape == "sphere":
            values["obstacle_diameter"] = parse_positive_float(
                self._diameter_var.get(), name="Diameter"
            )
            values["obstacle_attack_deg"] = 0.0
        elif shape in ("ellipsoid", "box"):
            values["obstacle_size_x"] = parse_positive_float(self._size_x_var.get(), name="Size x")
            values["obstacle_size_y"] = parse_positive_float(self._size_y_var.get(), name="Size y")
            values["obstacle_size_z"] = parse_positive_float(self._size_z_var.get(), name="Size z")
            values["obstacle_attack_deg"] = parse_float(self._attack_var.get(), name="Attack")
        else:
            values["obstacle_chord"] = parse_positive_float(self._chord_var.get(), name="Chord")
            values["obstacle_span"] = parse_positive_float(self._span_var.get(), name="Span")
            values["obstacle_thickness_ratio"] = parse_positive_float(
                self._thickness_var.get(), name="Thickness ratio"
            )
            values["obstacle_attack_deg"] = parse_float(self._wing_attack_var.get(), name="Attack")
        return values

    def _confirm_heavy_request(self, params: dict[str, object], window: tk.Toplevel) -> bool:
        advisory = assess_aerodynamics_3d_request(
            nx=cast(int, params["nx"]),
            ny=cast(int, params["ny"]),
            nz=cast(int, params["nz"]),
            t_max=cast(float, params["t_max"]),
            dt=cast(float, params["dt"]),
            sample_every=cast(int, params["sample_every"]),
        )
        return confirm_performance_advisory(window, advisory)

    def _on_solve(self) -> None:
        from complex_problems.aerodynamics_3d.result_dialog import Aerodynamics3DResultDialog

        run_solver_dialog(
            parent=self.parent,
            window=self.win,
            collect_inputs=self._collect_inputs,
            solver=solve_aerodynamics_3d,
            message="Solving 3D aerodynamics...",
            result_parent=self.parent,
            result_dialog_factory=Aerodynamics3DResultDialog,
            confirm_run=self._confirm_heavy_request,
        )
