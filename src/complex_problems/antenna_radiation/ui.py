"""UI dialog for configuring antenna radiation simulations."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from complex_problems.antenna_radiation.solver import solve_antenna_radiation
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
    run_solver_dialog,
)
from config import get_env_from_schema
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import make_modal

_ANTENNA_TYPES = ("dipole", "loop", "patch", "array")


class AntennaRadiationDialog:
    """Configuration dialog for antenna radiation patterns."""

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Antenna Radiation")
        self._build_ui()
        self._shell.finish(AdvancedDialogSize(900, 720, 660, 500))
        make_modal(self.win, parent)
        self._initial_focus.focus_set()

    def _build_ui(self) -> None:
        pad = int(get_env_from_schema("UI_PADDING"))
        self._shell = AdvancedDialogShell(
            self.win,
            title="Antenna Radiation",
            description=(
                "Compute far-field radiation maps, gain/directivity, and RMS field estimates. "
                "Choose an antenna family, then set geometry and sampling resolution."
            ),
            pad=pad,
        )
        body = self._shell.body

        add_how_to_config_section(
            body,
            self._shell.scroll,
            problem_id="antenna_radiation",
            pad=pad,
            wraplength=780,
        )

        system = ttk.LabelFrame(body, text="Antenna and excitation", padding=pad)
        system.pack(fill=tk.X, pady=(0, pad))
        row = ttk.Frame(system)
        row.pack(fill=tk.X, pady=pad // 2)
        self._antenna_type_var = tk.StringVar(value="dipole")
        at_combo = make_labeled_combo(
            row,
            "Antenna type",
            self._antenna_type_var,
            _ANTENNA_TYPES,
            width=12,
        )
        self._initial_focus = at_combo
        at_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_visibility())

        row = ttk.Frame(system)
        row.pack(fill=tk.X, pady=pad // 2)
        self._frequency_mhz_var = tk.StringVar(value="1000")
        self._power_w_var = tk.StringVar(value="10")
        self._efficiency_var = tk.StringVar(value="0.90")
        self._distance_m_var = tk.StringVar(value="50")
        make_labeled_entry(row, "Frequency (MHz)", self._frequency_mhz_var, width=11)
        make_labeled_entry(row, "Pₜₓ (W)", self._power_w_var, width=9)
        row = ttk.Frame(system)
        row.pack(fill=tk.X, pady=pad // 2)
        make_labeled_entry(row, "Efficiency η", self._efficiency_var, width=10)
        make_labeled_entry(row, "Observation r (m)", self._distance_m_var, width=13)
        ToolTip(row, "Efficiency must be between 0 and 1.")

        sampling = ttk.LabelFrame(body, text="Angular sampling", padding=pad)
        sampling.pack(fill=tk.X, pady=(0, pad))
        row = ttk.Frame(sampling)
        row.pack(fill=tk.X, pady=pad // 2)
        self._n_theta_var = tk.StringVar(value="181")
        self._n_phi_var = tk.StringVar(value="360")
        make_labeled_entry(row, "N_θ", self._n_theta_var, width=8)
        make_labeled_entry(row, "N_φ", self._n_phi_var, width=8)

        geometry = ttk.LabelFrame(body, text="Geometry and array parameters", padding=pad)
        geometry.pack(fill=tk.X)

        self._dipole_row = ttk.Frame(geometry)
        self._dipole_row.pack(fill=tk.X, pady=pad // 2)
        self._dipole_length_var = tk.StringVar(value="0.5")
        make_labeled_entry(self._dipole_row, "Length (λ)", self._dipole_length_var, width=10)

        self._loop_row = ttk.Frame(geometry)
        self._loop_row.pack(fill=tk.X, pady=pad // 2)
        self._loop_radius_var = tk.StringVar(value="0.10")
        make_labeled_entry(self._loop_row, "Radius (λ)", self._loop_radius_var, width=10)

        self._patch_row = ttk.Frame(geometry)
        self._patch_row.pack(fill=tk.X, pady=pad // 2)
        self._patch_length_var = tk.StringVar(value="0.5")
        self._patch_width_var = tk.StringVar(value="0.4")
        make_labeled_entry(self._patch_row, "Patch L (λ)", self._patch_length_var, width=10)
        make_labeled_entry(self._patch_row, "Patch W (λ)", self._patch_width_var, width=10)

        self._array_row = ttk.Frame(geometry)
        self._array_row.pack(fill=tk.X, pady=pad // 2)
        self._array_elements_var = tk.StringVar(value="8")
        self._array_spacing_var = tk.StringVar(value="0.5")
        self._array_phase_var = tk.StringVar(value="0.0")
        self._array_steer_var = tk.StringVar(value="90.0")
        array_row = ttk.Frame(self._array_row)
        array_row.pack(fill=tk.X, pady=pad // 2)
        make_labeled_entry(array_row, "Elements", self._array_elements_var, width=8)
        make_labeled_entry(array_row, "Spacing (λ)", self._array_spacing_var, width=10)
        array_row = ttk.Frame(self._array_row)
        array_row.pack(fill=tk.X, pady=pad // 2)
        make_labeled_entry(array_row, "Phase (deg)", self._array_phase_var, width=10)
        make_labeled_entry(array_row, "Steer θ (deg)", self._array_steer_var, width=12)

        self._shell.add_footer_button("Close", self.win.destroy)
        self._shell.add_footer_button("Solve", self._on_solve, primary=True)

        self._update_visibility()

    def _update_visibility(self) -> None:
        atype = self._antenna_type_var.get()
        self._dipole_row.pack_forget()
        self._loop_row.pack_forget()
        self._patch_row.pack_forget()
        self._array_row.pack_forget()
        if atype == "dipole":
            self._dipole_row.pack(fill=tk.X, pady=4)
        elif atype == "loop":
            self._loop_row.pack(fill=tk.X, pady=4)
        elif atype == "patch":
            self._patch_row.pack(fill=tk.X, pady=4)
        else:
            self._array_row.pack(fill=tk.X, pady=4)
        self._shell.refresh()

    def _collect_inputs(self) -> dict[str, object]:
        frequency_mhz = parse_positive_float(self._frequency_mhz_var.get(), name="Frequency (MHz)")
        frequency_hz = frequency_mhz * 1.0e6
        power_w = parse_positive_float(self._power_w_var.get(), name="Pₜₓ")
        efficiency = parse_float(self._efficiency_var.get(), name="Efficiency η")
        if efficiency <= 0 or efficiency > 1:
            raise ValueError("Efficiency η must be in (0, 1].")
        distance_m = parse_positive_float(self._distance_m_var.get(), name="Observation r")

        n_theta = parse_positive_int(self._n_theta_var.get(), name="N_θ", min_value=21)
        n_phi = parse_positive_int(self._n_phi_var.get(), name="N_φ", min_value=32)

        length_lambda = parse_positive_float(self._dipole_length_var.get(), name="Length (λ)")
        loop_radius_lambda = parse_positive_float(
            self._loop_radius_var.get(),
            name="Radius (λ)",
        )
        patch_length_lambda = parse_positive_float(self._patch_length_var.get(), name="Patch L")
        patch_width_lambda = parse_positive_float(self._patch_width_var.get(), name="Patch W")
        array_elements = parse_positive_int(
            self._array_elements_var.get(),
            name="Elements",
            min_value=2,
        )
        array_spacing_lambda = parse_positive_float(
            self._array_spacing_var.get(),
            name="Spacing (λ)",
        )
        array_phase_deg = parse_float(self._array_phase_var.get(), name="Phase")
        array_steer_deg = parse_float(self._array_steer_var.get(), name="Steer θ")

        return {
            "antenna_type": self._antenna_type_var.get(),
            "frequency_hz": frequency_hz,
            "transmit_power_w": power_w,
            "efficiency": efficiency,
            "observation_distance_m": distance_m,
            "n_theta": n_theta,
            "n_phi": n_phi,
            "length_lambda": length_lambda,
            "loop_radius_lambda": loop_radius_lambda,
            "patch_length_lambda": patch_length_lambda,
            "patch_width_lambda": patch_width_lambda,
            "array_elements": array_elements,
            "array_spacing_lambda": array_spacing_lambda,
            "array_phase_deg": array_phase_deg,
            "array_steer_theta_deg": array_steer_deg,
        }

    def _on_solve(self) -> None:
        from complex_problems.antenna_radiation.result_dialog import (
            AntennaRadiationResultDialog,
        )

        run_solver_dialog(
            parent=self.parent,
            window=self.win,
            collect_inputs=self._collect_inputs,
            solver=solve_antenna_radiation,
            message="Solving antenna radiation...",
            result_parent=self.parent,
            result_dialog_factory=AntennaRadiationResultDialog,
        )
