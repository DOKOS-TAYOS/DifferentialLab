"""Deterministic fixed-step solvers and diagnostics for the FPUT experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.signal import find_peaks

from complex_problems.fput_experiment.model import (
    FPUTModel,
    alpha_nonconvex_bonds,
    energy_components,
    fput_force,
    normal_mode_basis,
    single_mode_initial_state,
)

_DEPARTURE_THRESHOLD = 0.8
_PEAK_THRESHOLD = 0.8
_PEAK_PROMINENCE = 0.01


@dataclass(frozen=True, slots=True)
class FPUTResult:
    """Saved FPUT trajectory and cached diagnostics."""

    t: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    model: FPUTModel
    coefficient: float
    kinetic_energy: np.ndarray
    harmonic_potential_energy: np.ndarray
    nonlinear_potential_energy: np.ndarray
    potential_energy: np.ndarray
    total_energy: np.ndarray
    modal_q: np.ndarray
    modal_p: np.ndarray
    modal_energy: np.ndarray
    modal_energy_fraction: np.ndarray
    spectral_entropy: np.ndarray
    participation_number: np.ndarray
    recurrence_fidelity: np.ndarray
    recurrence_peak_indices: np.ndarray
    recurrence_peak_times: np.ndarray
    recurrence_peak_fidelities: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, float | int | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FPUTSweepResult:
    """Metrics retained from sequential alpha-FPUT recurrence scaling runs."""

    sweep_variable: str
    parameter_values: np.ndarray
    first_recurrence_times: np.ndarray
    recurrence_fidelities: np.ndarray
    detected: np.ndarray
    max_relative_hamiltonian_drift: np.ndarray
    slope: float | None
    intercept: float | None
    r_squared: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


def recurrence_fidelity(modal_fractions: np.ndarray) -> np.ndarray:
    """Compute the phase-insensitive modal-energy recurrence fidelity F(t)."""
    p = np.asarray(modal_fractions, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0:
        raise ValueError("Modal fractions must be a non-empty two-dimensional array.")
    values = 1.0 - 0.5 * np.sum(np.abs(p - p[0]), axis=1)
    return np.clip(values, 0.0, 1.0)


def detect_recurrence_peaks(
    t: np.ndarray, fidelity: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically identify post-departure high-fidelity recurrence peaks."""
    time = np.asarray(t, dtype=float)
    values = np.asarray(fidelity, dtype=float)
    if time.shape != values.shape or time.ndim != 1:
        raise ValueError("Time and fidelity must be equal-length vectors.")
    departed = np.flatnonzero(values < _DEPARTURE_THRESHOLD)
    if departed.size == 0:
        empty = np.empty(0, dtype=int)
        return empty, np.empty(0), np.empty(0)
    start = int(departed[0] + 1)
    # Include the departure sample as the left neighbour, so the first post-departure
    # local maximum is not accidentally discarded at the sliced-array boundary.
    peaks, _ = find_peaks(values[start - 1 :], prominence=_PEAK_PROMINENCE, height=_PEAK_THRESHOLD)
    accepted = peaks + start - 1
    accepted = accepted[accepted >= start]
    return accepted, time[accepted], values[accepted]


def _validate_inputs(
    n_particles: int,
    model: FPUTModel,
    coefficient: float,
    x0: np.ndarray,
    v0: np.ndarray,
    t_start: float,
    t_end: float,
    dt: float,
    sample_every: int,
    integrator: str,
) -> None:
    """Validate a solve request before allocating trajectory histories."""
    if n_particles < 2 or int(n_particles) != n_particles:
        raise ValueError("N must be an integer of at least 2.")
    if model not in {"alpha", "beta"} or not np.isfinite(coefficient):
        raise ValueError("Choose alpha or beta with a finite coefficient.")
    if model == "beta" and coefficient < 0.0:
        raise ValueError("beta must be non-negative.")
    if x0.shape != (n_particles,) or v0.shape != (n_particles,):
        raise ValueError("Initial displacement and velocity must have exactly N values each.")
    if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(v0)):
        raise ValueError("Initial state must be finite.")
    if not all(np.isfinite(value) for value in (t_start, t_end, dt)) or t_end <= t_start or dt <= 0:
        raise ValueError("Require finite t_start < t_end and dt > 0.")
    if sample_every < 1 or int(sample_every) != sample_every:
        raise ValueError("sample_every must be a positive integer.")
    if integrator not in {"verlet", "rk4"}:
        raise ValueError("integrator must be 'verlet' or 'rk4'.")


def _verlet_step(
    x: np.ndarray, v: np.ndarray, h: float, model: FPUTModel, coefficient: float
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one velocity-Verlet step for unit masses."""
    acceleration = fput_force(x, model=model, coefficient=coefficient)
    new_x = x + h * v + 0.5 * h**2 * acceleration
    new_v = v + 0.5 * h * (acceleration + fput_force(new_x, model=model, coefficient=coefficient))
    return new_x, new_v


def _rk4_step(
    x: np.ndarray, v: np.ndarray, h: float, model: FPUTModel, coefficient: float
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one historical fourth-order Runge--Kutta step."""
    k1x, k1v = v, fput_force(x, model=model, coefficient=coefficient)
    k2x = v + 0.5 * h * k1v
    k2v = fput_force(x + 0.5 * h * k1x, model=model, coefficient=coefficient)
    k3x = v + 0.5 * h * k2v
    k3v = fput_force(x + 0.5 * h * k2x, model=model, coefficient=coefficient)
    k4x = v + h * k3v
    k4v = fput_force(x + h * k3x, model=model, coefficient=coefficient)
    return x + h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x), v + h / 6.0 * (
        k1v + 2.0 * k2v + 2.0 * k3v + k4v
    )


def solve_fput(
    *,
    n_particles: int,
    model: FPUTModel = "alpha",
    coefficient: float = 0.25,
    x0: np.ndarray | list[float] | None = None,
    v0: np.ndarray | list[float] | None = None,
    amplitude: float = 1.0,
    initial_mode: int = 1,
    t_start: float = 0.0,
    t_end: float = 100.0,
    dt: float = 0.02,
    sample_every: int = 10,
    integrator: Literal["verlet", "rk4"] = "verlet",
) -> FPUTResult:
    """Solve one traditional normalized fixed-end FPUT chain.

    If x0/v0 are omitted, the historical physical-displacement single normal mode is used.
    The requested endpoint is reached exactly with one deterministic final partial step.
    """
    n = int(n_particles)
    if x0 is None:
        initial_x, initial_v = single_mode_initial_state(n, amplitude, initial_mode)
        x_arr = initial_x
    else:
        x_arr = np.asarray(x0, dtype=float).copy()
        initial_v = np.zeros(n, dtype=float)
    v_arr = initial_v if v0 is None else np.asarray(v0, dtype=float).copy()
    _validate_inputs(
        n, model, coefficient, x_arr, v_arr, t_start, t_end, dt, sample_every, integrator
    )
    span = t_end - t_start
    full_steps = int(np.floor(span / dt + 1.0e-12))
    partial_step = span - full_steps * dt
    if partial_step < 1.0e-12 * max(1.0, span):
        partial_step = 0.0
    n_steps = full_steps + int(partial_step > 0.0)
    saved_t: list[float] = [float(t_start)]
    saved_x: list[np.ndarray] = [x_arr.copy()]
    saved_v: list[np.ndarray] = [v_arr.copy()]
    time = float(t_start)
    stepper = _verlet_step if integrator == "verlet" else _rk4_step
    for step in range(1, n_steps + 1):
        h = partial_step if step == n_steps and partial_step > 0.0 else dt
        x_arr, v_arr = stepper(x_arr, v_arr, h, model, coefficient)
        time = t_end if step == n_steps else time + h
        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(v_arr)):
            raise FloatingPointError("FPUT numerical state became non-finite.")
        if step % sample_every == 0 or step == n_steps:
            saved_t.append(float(time))
            saved_x.append(x_arr.copy())
            saved_v.append(v_arr.copy())
    t = np.asarray(saved_t)
    displacement = np.asarray(saved_x)
    velocity = np.asarray(saved_v)
    basis, omega = normal_mode_basis(n)
    modal_q = displacement @ basis
    modal_p = velocity @ basis
    modal_energy = 0.5 * (modal_p**2 + (omega[np.newaxis, :] * modal_q) ** 2)
    modal_sum = np.sum(modal_energy, axis=1)
    fractions = np.divide(
        modal_energy,
        modal_sum[:, np.newaxis],
        out=np.zeros_like(modal_energy),
        where=modal_sum[:, np.newaxis] > 0.0,
    )
    entropy = np.zeros(t.size)
    valid = modal_sum > 0.0
    if n > 1:
        p = fractions[valid]
        entropy[valid] = -np.sum(np.where(p > 0.0, p * np.log(p), 0.0), axis=1) / np.log(n)
    participation = np.zeros(t.size)
    participation[valid] = 1.0 / np.sum(fractions[valid] ** 2, axis=1)
    energies = np.asarray(
        [
            energy_components(x, v, model=model, coefficient=coefficient)
            for x, v in zip(displacement, velocity, strict=True)
        ]
    )
    fidelity = recurrence_fidelity(fractions)
    peak_indices, peak_times, peak_fidelities = detect_recurrence_peaks(t, fidelity)
    total = energies[:, 4]
    max_abs_drift = float(np.max(np.abs(total - total[0])))
    denominator = max(abs(float(total[0])), np.finfo(float).eps)
    summary: dict[str, float | int | None] = {
        "initial_hamiltonian": float(total[0]),
        "final_hamiltonian": float(total[-1]),
        "maximum_absolute_hamiltonian_drift": max_abs_drift,
        "maximum_relative_hamiltonian_drift": max_abs_drift / denominator,
        "first_recurrence_time": float(peak_times[0]) if peak_times.size else None,
        "first_recurrence_fidelity": float(peak_fidelities[0]) if peak_fidelities.size else None,
        "number_of_recurrence_peaks": int(peak_times.size),
        "highest_late_recurrence_time": float(peak_times[np.argmax(peak_fidelities)])
        if peak_times.size
        else None,
        "highest_late_recurrence_fidelity": float(np.max(peak_fidelities))
        if peak_times.size
        else None,
    }
    warnings: list[str] = []
    if model == "alpha" and np.any(alpha_nonconvex_bonds(displacement[0], coefficient)):
        warnings.append(
            "Initial alpha state includes a bond with non-positive local potential curvature."
        )
    return FPUTResult(
        t,
        displacement,
        velocity,
        model,
        float(coefficient),
        energies[:, 0],
        energies[:, 1],
        energies[:, 2],
        energies[:, 3],
        total,
        modal_q,
        modal_p,
        modal_energy,
        fractions,
        entropy,
        participation,
        fidelity,
        peak_indices,
        peak_times,
        peak_fidelities,
        {
            "requested_dt": dt,
            "final_partial_step": partial_step,
            "number_of_integration_steps": n_steps,
            "sample_every": sample_every,
            "integrator": integrator,
            "model": model,
            "coefficient": float(coefficient),
            "warnings": tuple(warnings),
        },
        summary,
    )


def solve_fput_recurrence_scaling(
    *,
    sweep_variable: Literal["N", "alpha", "amplitude"],
    parameter_values: list[float] | np.ndarray,
    n_particles: int = 16,
    alpha: float = 0.25,
    amplitude: float = 1.0,
    initial_mode: int = 1,
    t_end: float = 3000.0,
    dt: float = 0.02,
    sample_every: int = 20,
) -> FPUTSweepResult:
    """Run sequential alpha-FPUT recurrence measurements and fit available log-log data."""
    values = np.asarray(parameter_values, dtype=float)
    if (
        values.ndim != 1
        or not 3 <= values.size <= 8
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
    ):
        raise ValueError("Provide 3--8 finite positive sweep values.")
    times = np.full(values.size, np.nan)
    fidelities = np.full(values.size, np.nan)
    drifts = np.full(values.size, np.nan)
    for idx, value in enumerate(values):
        run_n = int(value) if sweep_variable == "N" else n_particles
        if sweep_variable == "N" and (run_n != value or run_n < 2):
            raise ValueError("N sweep values must be integer values of at least 2.")
        result = solve_fput(
            n_particles=run_n,
            model="alpha",
            coefficient=value if sweep_variable == "alpha" else alpha,
            amplitude=value if sweep_variable == "amplitude" else amplitude,
            initial_mode=initial_mode,
            t_end=t_end,
            dt=dt,
            sample_every=sample_every,
        )
        drift = result.summary["maximum_relative_hamiltonian_drift"]
        assert isinstance(drift, (float, int))
        drifts[idx] = float(drift)
        if result.recurrence_peak_times.size:
            times[idx] = result.recurrence_peak_times[0]
            fidelities[idx] = result.recurrence_peak_fidelities[0]
    detected = np.isfinite(times)
    slope: float | None = None
    intercept: float | None = None
    r_squared: float | None = None
    if np.count_nonzero(detected) >= 3:
        log_x, log_y = np.log(values[detected]), np.log(times[detected])
        slope_value, intercept_value = np.polyfit(log_x, log_y, 1)
        fit = slope_value * log_x + intercept_value
        residual = float(np.sum((log_y - fit) ** 2))
        total = float(np.sum((log_y - np.mean(log_y)) ** 2))
        slope, intercept = float(slope_value), float(intercept_value)
        r_squared = 1.0 if total == 0.0 and residual == 0.0 else 1.0 - residual / total
    return FPUTSweepResult(
        sweep_variable,
        values,
        times,
        fidelities,
        detected,
        drifts,
        slope,
        intercept,
        r_squared,
        {"model": "alpha", "dt": dt, "t_end": t_end, "sample_every": sample_every},
    )


solve_fput_experiment = solve_fput
