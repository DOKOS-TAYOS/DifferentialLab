"""Tk configuration dialog for the dedicated FPUT scientific workflow."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import cast

import numpy as np

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
from complex_problems.fput_experiment.model import (
    FPUT_PRESETS,
    FPUTModel,
    legacy_kink_pair_initial_state,
    particle_coordinates,
    single_mode_initial_state,
    two_adjacent_modes_initial_state,
)
from complex_problems.fput_experiment.solver import solve_fput, solve_fput_recurrence_scaling
from config import get_env_from_schema
from frontend.performance_guard import assess_fput_request, confirm_performance_advisory
from frontend.window_utils import make_modal

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


def preset_study_transition(study: str, preset_model: FPUTModel) -> tuple[str, FPUTModel]:
    """Keep alpha-only scaling distinct from every beta preset configuration."""
    if study == "Recurrence scaling" and preset_model == "beta":
        return "Single simulation", "beta"
    return study, "alpha" if study == "Recurrence scaling" else preset_model


def fput_study_visibility(study: str) -> tuple[bool, bool]:
    """Return visibility for single-simulation and recurrence-scaling controls."""
    scaling = study == "Recurrence scaling"
    return not scaling, scaling


class FPUTExperimentDialog:
    """Compact editor for simulations and sequential recurrence scaling studies."""

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Fermi-Pasta-Ulam-Tsingou Experiment")
        self._build_ui()
        self._shell.finish(AdvancedDialogSize(880, 720, 650, 500))
        make_modal(self.win, parent)
        self._initial_focus.focus_set()

    def _build_ui(self) -> None:
        pad = int(get_env_from_schema("UI_PADDING"))
        self._shell = AdvancedDialogShell(
            self.win,
            title="Fermi-Pasta-Ulam-Tsingou Experiment",
            description=(
                "Velocity Verlet is recommended for long-time recurrence; "
                "RK4 reproduces the historical implementation."
            ),
            pad=pad,
        )
        root = self._shell.body
        add_how_to_config_section(
            root,
            self._shell.scroll,
            problem_id="fput_experiment",
            pad=pad,
        )
        self._preset_var = tk.StringVar(value="Interactive alpha recurrence")
        self._study_var = tk.StringVar(value=_STUDY_MODES[0])
        self._model_var = tk.StringVar(value="alpha")
        self._state_var = tk.StringVar(value=_INITIAL_STATES[0])
        experiment = ttk.LabelFrame(root, text="Experiment", padding=pad)
        experiment.pack(fill=tk.X, pady=(0, pad))
        row = ttk.Frame(experiment)
        row.pack(fill=tk.X, pady=3)
        preset = make_labeled_combo(row, "Preset", self._preset_var, tuple(FPUT_PRESETS), width=32)
        self._initial_focus = preset
        row = ttk.Frame(experiment)
        row.pack(fill=tk.X, pady=3)
        make_labeled_combo(row, "Study", self._study_var, _STUDY_MODES, width=20)
        preset.bind("<<ComboboxSelected>>", lambda _event: self._apply_preset())
        system = ttk.LabelFrame(root, text="System", padding=pad)
        system.pack(fill=tk.X, pady=(0, pad))
        row = ttk.Frame(system)
        row.pack(fill=tk.X, pady=3)
        self._n_var, self._coefficient_var, self._amplitude_var, self._mode_var = (
            tk.StringVar(value=value) for value in ("16", "0.25", "1", "1")
        )
        make_labeled_entry(row, "N", self._n_var, width=7)
        self._model_combo = make_labeled_combo(
            row, "Model", self._model_var, ("alpha", "beta"), width=10
        )
        self._coefficient_label = ttk.Label(row, text="α / β")
        self._coefficient_label.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(row, textvariable=self._coefficient_var, width=10).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        row = ttk.Frame(system)
        row.pack(fill=tk.X, pady=3)
        make_labeled_entry(row, "Amplitude", self._amplitude_var, width=10)
        make_labeled_entry(row, "Mode", self._mode_var, width=7)
        self._single_frame = ttk.LabelFrame(root, text="Initial conditions", padding=pad)
        self._single_frame.pack(fill=tk.X, pady=(0, pad))
        row = ttk.Frame(self._single_frame)
        row.pack(fill=tk.X, pady=3)
        state = make_labeled_combo(row, "Initial state", self._state_var, _INITIAL_STATES, width=28)
        state.bind("<<ComboboxSelected>>", lambda _event: self._update_state_hint())

        self._state_hint = ttk.Label(
            self._single_frame,
            text="Single-mode displacement x_j = A sin(mπj/(N+1)); velocities are zero.",
            style="Small.TLabel",
        )
        self._state_hint.pack(fill=tk.X, anchor=tk.W, pady=3)
        self._state_options_frame = ttk.Frame(self._single_frame)
        self._state_options_frame.pack(fill=tk.X)
        self._custom_x_var, self._custom_v_var, self._custom_q_var, self._custom_p_var = (
            tk.StringVar() for _ in range(4)
        )
        self._custom_frame = ttk.Frame(self._state_options_frame)
        self._custom_frame.pack(fill=tk.X, pady=3)
        self._custom_first_label = ttk.Label(self._custom_frame, text="x values")
        self._custom_first_label.pack(anchor=tk.W)
        self._custom_first_entry = ttk.Entry(self._custom_frame, textvariable=self._custom_x_var)
        self._custom_first_entry.pack(fill=tk.X, pady=(0, 4))
        self._custom_second_label = ttk.Label(self._custom_frame, text="v values")
        self._custom_second_label.pack(anchor=tk.W)
        self._custom_second_entry = ttk.Entry(self._custom_frame, textvariable=self._custom_v_var)
        self._custom_second_entry.pack(fill=tk.X)
        self._legacy_frame = ttk.Frame(self._state_options_frame)
        self._legacy_frame.pack(fill=tk.X, pady=3)
        self._legacy_width_var = tk.StringVar(value="0.5")
        self._legacy_first_center_var = tk.StringVar(value="6")
        self._legacy_second_center_var = tk.StringVar(value="26")
        make_labeled_entry(self._legacy_frame, "Width", self._legacy_width_var, width=10)
        make_labeled_entry(
            self._legacy_frame, "First center", self._legacy_first_center_var, width=10
        )
        make_labeled_entry(
            self._legacy_frame, "Second center", self._legacy_second_center_var, width=10
        )

        integration = ttk.LabelFrame(root, text="Integration", padding=pad)
        integration.pack(fill=tk.X, pady=(0, pad))
        row = ttk.Frame(integration)
        row.pack(fill=tk.X, pady=3)
        self._t_end_var, self._dt_var, self._sample_var = (
            tk.StringVar(value=value) for value in ("2200", "0.02", "20")
        )
        self._integrator_var = tk.StringVar(value="verlet")
        make_labeled_entry(row, "t end", self._t_end_var, width=10)
        make_labeled_entry(row, "dt", self._dt_var, width=10)
        make_labeled_entry(row, "Sample every", self._sample_var, width=10)
        self._integrator_frame = ttk.Frame(integration)
        self._integrator_frame.pack(fill=tk.X, pady=3)
        make_labeled_combo(
            self._integrator_frame,
            "Integrator",
            self._integrator_var,
            ("verlet", "rk4"),
            width=12,
        )
        self._study_var.trace_add("write", self._update_study_mode)
        self._sweep_var = tk.StringVar(value="N")
        self._sweep_values_var = tk.StringVar(value="8, 12, 16")
        self._scaling_frame = ttk.LabelFrame(root, text="Recurrence scaling", padding=pad)
        self._scaling_frame.pack(fill=tk.X, pady=(0, pad))
        row = ttk.Frame(self._scaling_frame)
        row.pack(fill=tk.X, pady=3)
        make_labeled_combo(
            row, "Scaling variable", self._sweep_var, ("N", "alpha", "amplitude"), width=12
        )
        make_labeled_entry(row, "Values (3--8)", self._sweep_values_var, width=30)
        self._shell.add_footer_button("Close", self.win.destroy)
        self._shell.add_footer_button("Run", self._on_solve, primary=True)
        self._apply_preset()

    def _apply_preset(self) -> None:
        """Populate ordinary editable controls from the selected preset."""
        preset = FPUT_PRESETS[self._preset_var.get()]
        study, model = preset_study_transition(self._study_var.get(), preset.model)
        if study != self._study_var.get():
            self._study_var.set(study)
        self._n_var.set(str(preset.n_particles))
        self._model_var.set(model)
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
            "Custom modal state": "Enter exactly N comma-separated Q and P modal values.",
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
        state = self._state_var.get()
        custom = state in {"Custom particle state", "Custom modal state"}
        legacy = state == "Legacy kink-pair study"
        self._custom_frame.pack_forget()
        self._legacy_frame.pack_forget()
        if custom:
            modal = state == "Custom modal state"
            self._custom_first_label.configure(text="Q values" if modal else "x values")
            self._custom_second_label.configure(text="P values" if modal else "v values")
            self._custom_first_entry.configure(
                textvariable=self._custom_q_var if modal else self._custom_x_var
            )
            self._custom_second_entry.configure(
                textvariable=self._custom_p_var if modal else self._custom_v_var
            )
            self._custom_frame.pack(fill=tk.X, pady=3)
        if legacy:
            self._legacy_frame.pack(fill=tk.X, pady=3)
        self._shell.refresh()

    def _update_study_mode(self, *_args: object) -> None:
        """Make the alpha-only scaling interpretation visible and unambiguous."""
        single_visible, scaling = fput_study_visibility(self._study_var.get())
        if single_visible:
            self._single_frame.pack(
                fill=tk.X,
                pady=(0, self._shell.pad),
                before=self._scaling_frame,
            )
            self._integrator_frame.pack(fill=tk.X, pady=3)
        else:
            self._single_frame.pack_forget()
            self._integrator_frame.pack_forget()
        if scaling:
            self._scaling_frame.pack(fill=tk.X, pady=(0, self._shell.pad))
        else:
            self._scaling_frame.pack_forget()
        if scaling:
            self._model_var.set("alpha")
            self._model_combo.configure(state="disabled")
            self._coefficient_label.configure(text="α (alpha-FPUT)")
        else:
            self._model_combo.configure(state="readonly")
            self._coefficient_label.configure(text="α / β")
        self._shell.refresh()

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
                parse_exact_vector(self._custom_q_var.get(), n, "Modal Q"),
                parse_exact_vector(self._custom_p_var.get(), n, "Modal P"),
            )
        else:
            if model != "alpha":
                raise ValueError(
                    "Legacy kink-pair study is available only for the alpha-FPUT model."
                )
            x0, v0 = legacy_kink_pair_initial_state(
                n,
                amplitude,
                coefficient,
                width=parse_positive_float(self._legacy_width_var.get(), name="Width"),
                first_center=parse_float(self._legacy_first_center_var.get(), name="First center"),
                second_center=parse_float(
                    self._legacy_second_center_var.get(), name="Second center"
                ),
            )
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
                run_count=(
                    len(cast(list[float], params["parameter_values"]))
                    if "parameter_values" in params
                    else 1
                ),
                n_particles_values=(
                    cast(list[float], params["parameter_values"])
                    if params.get("sweep_variable") == "N"
                    else None
                ),
            ),
        )
