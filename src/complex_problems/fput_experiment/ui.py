"""Tk configuration dialog for the dedicated FPUT scientific workflow."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import cast

import numpy as np

from complex_problems.common import parse_float, parse_positive_float, parse_positive_int
from complex_problems.common.dialog_ui import (
    make_labeled_combo,
    make_labeled_entry,
    run_solver_dialog,
)
from complex_problems.fput_experiment.model import (
    FPUT_PRESETS,
    legacy_kink_pair_initial_state,
    particle_coordinates,
    single_mode_initial_state,
    two_adjacent_modes_initial_state,
)
from complex_problems.fput_experiment.solver import solve_fput, solve_fput_recurrence_scaling
from config import get_env_from_schema
from frontend.performance_guard import assess_fput_request, confirm_performance_advisory
from frontend.window_utils import fit_and_center, make_modal

_STUDY_MODES = ("Single simulation", "Recurrence scaling")
_INITIAL_STATES = (
    "Single normal mode",
    "Two adjacent modes",
    "Custom particle state",
    "Custom modal state",
    "Legacy kink-pair study",
)


def parse_exact_vector(text: str, n: int, name: str) -> np.ndarray:
    """Parse exactly N comma-separated finite values, never padding or truncating."""
    try:
        values = np.asarray(
            [float(item.strip()) for item in text.split(",") if item.strip()], dtype=float
        )
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated numeric values.") from exc
    if values.shape != (n,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain exactly {n} finite values.")
    return values


class FPUTExperimentDialog:
    """Compact editor for simulations and sequential recurrence scaling studies."""

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Fermi-Pasta-Ulam-Tsingou Experiment")
        self.win.configure(bg=get_env_from_schema("UI_BACKGROUND"))
        self._build_ui()
        fit_and_center(self.win, min_width=860, min_height=680, padding=24, resizable=True)
        make_modal(self.win, parent)

    def _build_ui(self) -> None:
        root = ttk.Frame(self.win, padding=18)
        root.pack(fill=tk.BOTH, expand=True)
        ttk.Label(root, text="Fermi-Pasta-Ulam-Tsingou Experiment", style="Title.TLabel").pack(
            anchor=tk.W
        )
        ttk.Label(
            root,
            text=(
                "Velocity Verlet is recommended for long-time recurrence; "
                "RK4 reproduces the historical implementation."
            ),
            style="Small.TLabel",
            wraplength=800,
        ).pack(anchor=tk.W, pady=(4, 12))
        self._preset_var = tk.StringVar(value="Interactive alpha recurrence")
        self._study_var = tk.StringVar(value=_STUDY_MODES[0])
        self._model_var = tk.StringVar(value="alpha")
        self._state_var = tk.StringVar(value=_INITIAL_STATES[0])
        row = ttk.Frame(root)
        row.pack(fill=tk.X, pady=3)
        preset = make_labeled_combo(row, "Preset", self._preset_var, tuple(FPUT_PRESETS), width=32)
        make_labeled_combo(row, "Study", self._study_var, _STUDY_MODES, width=20)
        preset.bind("<<ComboboxSelected>>", lambda _event: self._apply_preset())
        row = ttk.Frame(root)
        row.pack(fill=tk.X, pady=3)
        self._n_var, self._coefficient_var, self._amplitude_var, self._mode_var = (
            tk.StringVar(value=value) for value in ("16", "0.25", "1", "1")
        )
        make_labeled_entry(row, "N", self._n_var, width=7)
        make_labeled_combo(row, "Model", self._model_var, ("alpha", "beta"), width=10)
        make_labeled_entry(row, "α / β", self._coefficient_var, width=10)
        make_labeled_entry(row, "Amplitude", self._amplitude_var, width=10)
        make_labeled_entry(row, "Mode", self._mode_var, width=7)
        row = ttk.Frame(root)
        row.pack(fill=tk.X, pady=3)
        self._t_end_var, self._dt_var, self._sample_var = (
            tk.StringVar(value=value) for value in ("2200", "0.02", "20")
        )
        self._integrator_var = tk.StringVar(value="verlet")
        make_labeled_entry(row, "t end", self._t_end_var, width=10)
        make_labeled_entry(row, "dt", self._dt_var, width=10)
        make_labeled_entry(row, "Sample every", self._sample_var, width=10)
        make_labeled_combo(row, "Integrator", self._integrator_var, ("verlet", "rk4"), width=12)
        row = ttk.Frame(root)
        row.pack(fill=tk.X, pady=3)
        state = make_labeled_combo(row, "Initial state", self._state_var, _INITIAL_STATES, width=28)
        state.bind("<<ComboboxSelected>>", lambda _event: self._update_state_hint())
        self._state_hint = ttk.Label(
            root,
            text="Single-mode displacement x_j = A sin(mπj/(N+1)); velocities are zero.",
            style="Small.TLabel",
        )
        self._state_hint.pack(anchor=tk.W, pady=3)
        self._custom_x_var, self._custom_v_var, self._custom_q_var, self._custom_p_var = (
            tk.StringVar() for _ in range(4)
        )
        self._custom_frame = ttk.Frame(root)
        self._custom_frame.pack(fill=tk.X, pady=3)
        make_labeled_entry(self._custom_frame, "x values", self._custom_x_var, width=42)
        make_labeled_entry(self._custom_frame, "v values", self._custom_v_var, width=42)
        self._sweep_var = tk.StringVar(value="N")
        self._sweep_values_var = tk.StringVar(value="8, 12, 16")
        row = ttk.Frame(root)
        row.pack(fill=tk.X, pady=3)
        make_labeled_combo(
            row, "Scaling variable", self._sweep_var, ("N", "alpha", "amplitude"), width=12
        )
        make_labeled_entry(row, "Values (3--8)", self._sweep_values_var, width=30)
        row = ttk.Frame(root)
        row.pack(fill=tk.X, pady=(14, 0))
        ttk.Button(row, text="Run", command=self._on_solve).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row, text="Close", style="Cancel.TButton", command=self.win.destroy).pack(
            side=tk.LEFT
        )
        self._apply_preset()

    def _apply_preset(self) -> None:
        """Populate ordinary editable controls from the selected preset."""
        preset = FPUT_PRESETS[self._preset_var.get()]
        self._n_var.set(str(preset.n_particles))
        self._model_var.set(preset.model)
        self._coefficient_var.set(str(preset.coefficient))
        self._amplitude_var.set(str(preset.amplitude))
        self._mode_var.set(str(preset.mode))
        self._t_end_var.set(str(preset.t_end))
        self._dt_var.set(str(preset.dt))
        self._sample_var.set(str(preset.sample_every))
        self._state_var.set(
            "Legacy kink-pair study"
            if preset.initial_state == "legacy_kink_pair"
            else "Single normal mode"
        )
        self._update_state_hint()

    def _update_state_hint(self) -> None:
        """Describe the selected state without obscuring editable controls."""
        hints = {
            "Two adjacent modes": "Historical equal A/2 superposition of modes m and m+1.",
            "Custom particle state": "Enter exactly N comma-separated x and v values.",
            "Custom modal state": "Enter exactly N comma-separated Q and P values in the x fields.",
            "Legacy kink-pair study": (
                "Historical alpha-FPUT kink-pair study; not claimed to be an "
                "exact discrete soliton."
            ),
        }
        self._state_hint.configure(
            text=hints.get(
                self._state_var.get(),
                "Single-mode displacement x_j = A sin(mπj/(N+1)); velocities are zero.",
            )
        )

    def _collect_single(self) -> dict[str, object]:
        n = parse_positive_int(self._n_var.get(), name="N", min_value=2)
        coefficient = parse_float(self._coefficient_var.get(), name="α / β")
        model = self._model_var.get()
        amplitude = parse_float(self._amplitude_var.get(), name="Amplitude")
        mode = parse_positive_int(self._mode_var.get(), name="Mode")
        state = self._state_var.get()
        if state == "Single normal mode":
            x0, v0 = single_mode_initial_state(n, amplitude, mode)
        elif state == "Two adjacent modes":
            x0, v0 = two_adjacent_modes_initial_state(n, amplitude, mode)
        elif state == "Custom particle state":
            x0, v0 = (
                parse_exact_vector(self._custom_x_var.get(), n, "Displacements"),
                parse_exact_vector(self._custom_v_var.get(), n, "Velocities"),
            )
        elif state == "Custom modal state":
            x0, v0 = particle_coordinates(
                parse_exact_vector(self._custom_x_var.get(), n, "Modal Q"),
                parse_exact_vector(self._custom_v_var.get(), n, "Modal P"),
            )
        else:
            x0, v0 = legacy_kink_pair_initial_state(n, amplitude, coefficient)
        return {
            "n_particles": n,
            "model": model,
            "coefficient": coefficient,
            "x0": x0,
            "v0": v0,
            "t_end": parse_positive_float(self._t_end_var.get(), name="t end"),
            "dt": parse_positive_float(self._dt_var.get(), name="dt"),
            "sample_every": parse_positive_int(self._sample_var.get(), name="Sample every"),
            "integrator": self._integrator_var.get(),
        }

    def _on_solve(self) -> None:
        """Dispatch either a cached single trajectory or sequential sweep."""
        from complex_problems.fput_experiment.result_dialog import FPUTResultDialog

        if self._study_var.get() == "Recurrence scaling":
            values = [float(value.strip()) for value in self._sweep_values_var.get().split(",")]
            params: dict[str, object] = {
                "sweep_variable": self._sweep_var.get(),
                "parameter_values": values,
                "n_particles": parse_positive_int(self._n_var.get(), name="N", min_value=2),
                "alpha": parse_float(self._coefficient_var.get(), name="alpha"),
                "amplitude": parse_float(self._amplitude_var.get(), name="Amplitude"),
                "initial_mode": parse_positive_int(self._mode_var.get(), name="Mode"),
                "t_end": parse_positive_float(self._t_end_var.get(), name="t end"),
                "dt": parse_positive_float(self._dt_var.get(), name="dt"),
                "sample_every": parse_positive_int(self._sample_var.get(), name="Sample every"),
            }
            run_solver_dialog(
                parent=self.parent,
                window=self.win,
                collect_inputs=lambda: params,
                solver=solve_fput_recurrence_scaling,
                message="Running sequential FPUT recurrence scaling...",
                result_parent=self.parent,
                result_dialog_factory=FPUTResultDialog,
                confirm_run=self._confirm_fput_request,
            )
        else:
            run_solver_dialog(
                parent=self.parent,
                window=self.win,
                collect_inputs=self._collect_single,
                solver=solve_fput,
                message="Solving FPUT chain...",
                result_parent=self.parent,
                result_dialog_factory=FPUTResultDialog,
                confirm_run=self._confirm_fput_request,
            )

    def _confirm_fput_request(self, params: dict[str, object], window: tk.Toplevel) -> bool:
        """Require explicit confirmation for historically expensive FPUT requests."""
        return confirm_performance_advisory(
            window,
            assess_fput_request(
                n_particles=cast(int, params["n_particles"]),
                t_start=0.0,
                t_end=cast(float, params["t_end"]),
                dt=cast(float, params["dt"]),
                sample_every=cast(int, params["sample_every"]),
            ),
        )
