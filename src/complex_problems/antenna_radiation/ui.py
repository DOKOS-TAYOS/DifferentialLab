"""UI dialog for configuring antenna radiation simulations."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from complex_problems.antenna_radiation.solver import solve_antenna_radiation
from complex_problems.common import (
    add_how_to_config_section,
    parse_float,
    parse_positive_float,
    parse_positive_int,
    run_solver_with_loading,
)
from config import get_env_from_schema
from frontend.theme import get_font
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.ui_dialogs.tooltip import ToolTip
from frontend.window_utils import fit_and_center, make_modal

_ANTENNA_TYPES = ("dipole", "loop", "patch", "array")


class AntennaRadiationDialog:
    """Configuration dialog for antenna radiation patterns."""

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Antenna Radiation")
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

        ttk.Label(body, text="Antenna Radiation", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            body,
            text=(
                "Far-field radiation maps, gain/directivity, and field magnitudes.\n"
                "Choose the antenna family and set its geometric parameters."
            ),
            style="Small.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, pad))

        add_how_to_config_section(
            body,
            scroll,
            problem_id="antenna_radiation",
            pad=pad,
            wraplength=780,
        )

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._antenna_type_var = tk.StringVar(value="dipole")
        at_combo = self._make_combo(
            row,
            "Antenna type",
            self._antenna_type_var,
            _ANTENNA_TYPES,
            width=12,
        )
        at_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_visibility())

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._frequency_mhz_var = tk.StringVar(value="1000")
        self._power_w_var = tk.StringVar(value="10")
        self._efficiency_var = tk.StringVar(value="0.90")
        self._distance_m_var = tk.StringVar(value="50")
        self._make_entry(row, "Frequency (MHz)", self._frequency_mhz_var, width=11)
        self._make_entry(row, "Pₜₓ (W)", self._power_w_var, width=9)
        self._make_entry(row, "Efficiency η", self._efficiency_var, width=10)
        self._make_entry(row, "Observation r (m)", self._distance_m_var, width=13)
        ToolTip(row, "Efficiency must be between 0 and 1.")

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=pad // 2)
        self._n_theta_var = tk.StringVar(value="181")
        self._n_phi_var = tk.StringVar(value="360")
        self._make_entry(row, "N_θ", self._n_theta_var, width=8)
        self._make_entry(row, "N_φ", self._n_phi_var, width=8)

        ttk.Separator(body).pack(fill=tk.X, pady=pad)
        ttk.Label(body, text="Antenna parameters", style="Small.TLabel").pack(anchor=tk.W)

        self._dipole_row = ttk.Frame(body)
        self._dipole_row.pack(fill=tk.X, pady=pad // 2)
        self._dipole_length_var = tk.StringVar(value="0.5")
        self._make_entry(self._dipole_row, "Length (λ)", self._dipole_length_var, width=10)

        self._loop_row = ttk.Frame(body)
        self._loop_row.pack(fill=tk.X, pady=pad // 2)
        self._loop_radius_var = tk.StringVar(value="0.10")
        self._make_entry(self._loop_row, "Radius (λ)", self._loop_radius_var, width=10)

        self._patch_row = ttk.Frame(body)
        self._patch_row.pack(fill=tk.X, pady=pad // 2)
        self._patch_length_var = tk.StringVar(value="0.5")
        self._patch_width_var = tk.StringVar(value="0.4")
        self._make_entry(self._patch_row, "Patch L (λ)", self._patch_length_var, width=10)
        self._make_entry(self._patch_row, "Patch W (λ)", self._patch_width_var, width=10)

        self._array_row = ttk.Frame(body)
        self._array_row.pack(fill=tk.X, pady=pad // 2)
        self._array_elements_var = tk.StringVar(value="8")
        self._array_spacing_var = tk.StringVar(value="0.5")
        self._array_phase_var = tk.StringVar(value="0.0")
        self._array_steer_var = tk.StringVar(value="90.0")
        self._make_entry(self._array_row, "Elements", self._array_elements_var, width=8)
        self._make_entry(self._array_row, "Spacing (λ)", self._array_spacing_var, width=10)
        self._make_entry(self._array_row, "Phase (deg)", self._array_phase_var, width=10)
        self._make_entry(self._array_row, "Steer θ (deg)", self._array_steer_var, width=12)

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
        atype = self._antenna_type_var.get()
        self._dipole_row.pack_forget()
        self._loop_row.pack_forget()
        self._patch_row.pack_forget()
        self._array_row.pack_forget()
        if atype == "dipole":
            self._dipole_row.pack(fill=tk.X, pady=4, before=self._btn_row)
        elif atype == "loop":
            self._loop_row.pack(fill=tk.X, pady=4, before=self._btn_row)
        elif atype == "patch":
            self._patch_row.pack(fill=tk.X, pady=4, before=self._btn_row)
        else:
            self._array_row.pack(fill=tk.X, pady=4, before=self._btn_row)

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
        try:
            params = self._collect_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=self.win)
            return

        self.win.destroy()

        def _task():
            return solve_antenna_radiation(**params)

        def _on_success(result) -> None:
            from complex_problems.antenna_radiation.result_dialog import (
                AntennaRadiationResultDialog,
            )

            AntennaRadiationResultDialog(self.parent, result=result)

        run_solver_with_loading(
            parent=self.parent,
            message="Solving antenna radiation...",
            task=_task,
            on_success=_on_success,
        )
