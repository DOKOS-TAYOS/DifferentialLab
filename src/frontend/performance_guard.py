"""Helpers for warning about expensive numerical requests before execution."""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox
from typing import Literal

_MIB = 1024 * 1024
_WARN_OUTPUT_POINTS = 50_000
_CONFIRM_OUTPUT_POINTS = 250_000
_WARN_PDE_GRID_POINTS = 150_000
_CONFIRM_PDE_GRID_POINTS = 750_000
_WARN_HISTORY_BYTES = 128 * _MIB
_CONFIRM_HISTORY_BYTES = 512 * _MIB


@dataclass(frozen=True, slots=True)
class PerformanceAdvisory:
    """A UI advisory shown before running an expensive configuration."""

    severity: Literal["warn", "confirm"]
    title: str
    message: str


def _format_bytes(num_bytes: int) -> str:
    """Format a byte count using binary units for user-facing warnings."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < _MIB:
        return f"{num_bytes / 1024:.1f} KiB"
    gib = 1024 * _MIB
    if num_bytes < gib:
        return f"{num_bytes / _MIB:.1f} MiB"
    return f"{num_bytes / gib:.2f} GiB"


def _estimate_frames(t_min: float, t_max: float, dt: float) -> int:
    """Estimate the number of stored frames for uniform time-stepping."""
    if dt <= 0 or t_max <= t_min:
        return 0
    return int(math.ceil((t_max - t_min) / dt)) + 1


def _severity_from_thresholds(
    *,
    value: int,
    warn_threshold: int,
    confirm_threshold: int,
) -> Literal["warn", "confirm"] | None:
    """Map a scalar request size to an advisory severity."""
    if value >= confirm_threshold:
        return "confirm"
    if value >= warn_threshold:
        return "warn"
    return None


def assess_parameters_dialog_request(
    *,
    equation_type: str,
    n_points: int,
    state_size: int,
    n_points_y: int | None = None,
) -> PerformanceAdvisory | None:
    """Assess the cost of the standard ParametersDialog request."""
    if equation_type == "pde":
        ny = n_points if n_points_y is None else n_points_y
        grid_points = n_points * ny
        severity = _severity_from_thresholds(
            value=grid_points,
            warn_threshold=_WARN_PDE_GRID_POINTS,
            confirm_threshold=_CONFIRM_PDE_GRID_POINTS,
        )
        if severity is None:
            return None
        return PerformanceAdvisory(
            severity=severity,
            title="Large PDE grid request",
            message=(
                f"This PDE run will assemble a {n_points:,} x {ny:,} grid "
                f"({grid_points:,} points). High-resolution PDE grids can take noticeably longer "
                "to assemble, solve, plot, and export."
            ),
        )

    total_output_points = n_points * max(1, state_size)
    severity = _severity_from_thresholds(
        value=max(n_points, total_output_points),
        warn_threshold=_WARN_OUTPUT_POINTS,
        confirm_threshold=_CONFIRM_OUTPUT_POINTS,
    )
    if severity is None:
        return None

    return PerformanceAdvisory(
        severity=severity,
        title="Dense output request",
        message=(
            f"This run will generate {n_points:,} sampled points"
            + (
                f" across {state_size:,} component(s) ({total_output_points:,} values total)."
                if state_size > 1
                else "."
            )
            + " Dense output can make solving, plotting and exporting noticeably slower."
        ),
    )


def assess_time_history_request(
    *,
    label: str,
    frames: int,
    points_per_frame: int,
    array_count: int,
    bytes_per_value: int,
) -> PerformanceAdvisory | None:
    """Assess the cost of storing a time history in memory."""
    if frames <= 0 or points_per_frame <= 0 or array_count <= 0 or bytes_per_value <= 0:
        return None

    total_bytes = frames * points_per_frame * array_count * bytes_per_value
    severity = _severity_from_thresholds(
        value=total_bytes,
        warn_threshold=_WARN_HISTORY_BYTES,
        confirm_threshold=_CONFIRM_HISTORY_BYTES,
    )
    if severity is None:
        return None

    return PerformanceAdvisory(
        severity=severity,
        title=f"Large {label} history",
        message=(
            f"This setup is expected to store about {frames:,} frame(s) with "
            f"{points_per_frame:,} point(s) per frame across {array_count} array(s), "
            f"roughly {_format_bytes(total_bytes)} of raw history data. "
            "Continue only if this is intentional."
        ),
    )


def assess_coupled_oscillators_request(
    *,
    n_oscillators: int,
    n_points: int,
) -> PerformanceAdvisory | None:
    """Assess a coupled-oscillators solve request."""
    return assess_parameters_dialog_request(
        equation_type="vector_ode",
        n_points=n_points,
        state_size=2 * max(1, n_oscillators),
    )


def assess_membrane_request(
    *,
    nx: int,
    ny: int,
    t_min: float,
    t_max: float,
    dt: float,
) -> PerformanceAdvisory | None:
    """Assess a membrane solve request."""
    return assess_time_history_request(
        label="2D membrane",
        frames=_estimate_frames(t_min, t_max, dt),
        points_per_frame=max(1, nx * ny),
        array_count=2,
        bytes_per_value=8,
    )


def assess_schrodinger_request(
    *,
    dimension: int,
    nx: int,
    ny: int,
    t_min: float,
    t_max: float,
    dt: float,
) -> PerformanceAdvisory | None:
    """Assess a time-dependent Schrödinger solve request."""
    points_per_frame = nx if dimension == 1 else nx * ny
    return assess_time_history_request(
        label="Schrodinger TD",
        frames=_estimate_frames(t_min, t_max, dt),
        points_per_frame=max(1, points_per_frame),
        array_count=1,
        bytes_per_value=16,
    )


def assess_nonlinear_waves_request(
    *,
    model_type: str,
    nx: int,
    t_min: float,
    t_max: float,
    dt: float,
) -> PerformanceAdvisory | None:
    """Assess an NLSE/KdV solve request."""
    bytes_per_value = 16 if model_type == "nlse" else 8
    return assess_time_history_request(
        label="nonlinear wave",
        frames=_estimate_frames(t_min, t_max, dt),
        points_per_frame=max(1, nx),
        array_count=1,
        bytes_per_value=bytes_per_value,
    )


def assess_pipe_flow_request(
    *,
    model_type: str,
    nx: int,
    t_max: float | None = None,
    dt: float | None = None,
    sample_every: int | None = None,
) -> PerformanceAdvisory | None:
    """Assess a pipe-flow solve request."""
    if model_type == "steady":
        return assess_parameters_dialog_request(
            equation_type="ode",
            n_points=nx,
            state_size=2,
        )

    if t_max is None or dt is None or sample_every is None or sample_every <= 0:
        return None

    n_steps = int(math.ceil(t_max / dt))
    n_samples = len(range(0, n_steps + 1, sample_every))
    if n_steps % sample_every != 0:
        n_samples += 1
    return assess_time_history_request(
        label="pipe-flow transient",
        frames=n_samples,
        points_per_frame=max(1, nx),
        array_count=6,
        bytes_per_value=8,
    )


def assess_aerodynamics_request(
    *,
    nx: int,
    ny: int,
    t_max: float,
    dt: float,
    sample_every: int,
) -> PerformanceAdvisory | None:
    """Assess a 2D aerodynamics solve request."""
    n_steps = int(math.ceil(t_max / dt))
    n_samples = len(range(0, n_steps + 1, sample_every))
    if n_steps % sample_every != 0:
        n_samples += 1
    return assess_time_history_request(
        label="2D aerodynamics",
        frames=n_samples,
        points_per_frame=max(1, nx * ny),
        array_count=5,
        bytes_per_value=8,
    )


def confirm_performance_advisory(
    parent: tk.Misc | None,
    advisory: PerformanceAdvisory | None,
) -> bool:
    """Display a performance advisory and return whether execution should continue."""
    if advisory is None:
        return True

    if advisory.severity == "warn":
        if parent is None:
            messagebox.showwarning(advisory.title, advisory.message)
        else:
            messagebox.showwarning(advisory.title, advisory.message, parent=parent)
        return True

    if parent is None:
        return bool(messagebox.askyesno(advisory.title, advisory.message))
    return bool(messagebox.askyesno(advisory.title, advisory.message, parent=parent))
