"""Compute statistics and physical magnitudes from ODE solutions."""

from __future__ import annotations

from typing import Any

import numpy as np

from utils import get_logger

logger = get_logger(__name__)


def compute_statistics(
    x: np.ndarray,
    y: np.ndarray,
    selected: set[str] | None = None,
) -> dict[str, Any]:
    """Compute requested statistics for the primary solution component.

    Args:
        x: Independent variable values.
        y: Solution array — shape ``(n_vars, n_points)`` or ``(n_points,)``.
        selected: Set of statistic keys to compute. ``None`` means all.

    Returns:
        Dictionary mapping statistic names to their computed values.
    """
    y_2d = np.atleast_2d(y)
    y_primary = y_2d[0]
    all_stats = selected or {
        "mean", "rms", "std", "max", "min", "integral",
        "zero_crossings", "period", "amplitude", "energy",
    }

    results: dict[str, Any] = {}

    if "mean" in all_stats:
        results["mean"] = _compute_mean(x, y_primary)

    if "rms" in all_stats:
        results["rms"] = _compute_rms(x, y_primary)

    if "std" in all_stats:
        results["std"] = float(np.std(y_primary))

    if "max" in all_stats:
        idx = int(np.argmax(y_primary))
        results["max"] = {"value": float(y_primary[idx]), "x": float(x[idx])}

    if "min" in all_stats:
        idx = int(np.argmin(y_primary))
        results["min"] = {"value": float(y_primary[idx]), "x": float(x[idx])}

    if "integral" in all_stats:
        results["integral"] = float(np.trapezoid(y_primary, x))

    if "zero_crossings" in all_stats:
        results["zero_crossings"] = _count_zero_crossings(y_primary)

    if "period" in all_stats:
        results["period"] = _estimate_period(x, y_primary)

    if "amplitude" in all_stats:
        results["amplitude"] = _estimate_amplitude(y_primary)

    if "energy" in all_stats and y_2d.shape[0] >= 2:
        results["energy"] = _estimate_energy(x, y_2d)

    logger.debug("Computed statistics: %s", list(results.keys()))
    return results


def _compute_mean(x: np.ndarray, y: np.ndarray) -> float:
    """Weighted mean over the domain using trapezoidal integration."""
    span = x[-1] - x[0]
    if span == 0:
        return float(np.mean(y))
    return float(np.trapezoid(y, x) / span)


def _compute_rms(x: np.ndarray, y: np.ndarray) -> float:
    """Root mean square over the domain."""
    span = x[-1] - x[0]
    if span == 0:
        return float(np.sqrt(np.mean(y ** 2)))
    return float(np.sqrt(np.trapezoid(y ** 2, x) / span))


def _count_zero_crossings(y: np.ndarray) -> int:
    """Count the number of sign changes in *y*."""
    signs = np.sign(y)
    crossings = np.where(np.diff(signs) != 0)[0]
    return len(crossings)


def _estimate_period(x: np.ndarray, y: np.ndarray) -> float | None:
    """Estimate the period of an oscillatory signal via peak detection.

    Returns:
        Estimated period, or ``None`` if fewer than 2 peaks found.
    """
    from scipy.signal import find_peaks

    y_centered = y - np.mean(y)
    peaks, _ = find_peaks(y_centered, distance=max(3, len(y) // 50))
    if len(peaks) < 2:
        return None
    intervals = np.diff(x[peaks])
    return float(np.mean(intervals))


def _estimate_amplitude(y: np.ndarray) -> float:
    """Estimate amplitude as half the peak-to-peak range."""
    return float((np.max(y) - np.min(y)) / 2.0)


def _estimate_energy(x: np.ndarray, y_2d: np.ndarray) -> dict[str, float]:
    """Rough energy estimate for a 2nd-order oscillator.

    Assumes ``y[0]`` is position and ``y[1]`` is velocity with unit
    mass/frequency.  Returns kinetic, potential, and total energy at
    the *first* time step (conserved for undamped systems).

    Args:
        x: Independent variable values.
        y_2d: Solution array of shape ``(n_vars, n_points)``.

    Returns:
        Dict with ``kinetic``, ``potential``, and ``total`` energies.
    """
    position = y_2d[0]
    velocity = y_2d[1]

    kinetic = 0.5 * velocity ** 2
    potential = 0.5 * position ** 2

    return {
        "kinetic_initial": float(kinetic[0]),
        "potential_initial": float(potential[0]),
        "total_initial": float(kinetic[0] + potential[0]),
        "kinetic_mean": float(np.mean(kinetic)),
        "potential_mean": float(np.mean(potential)),
        "total_mean": float(np.mean(kinetic + potential)),
    }
