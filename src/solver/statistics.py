"""Compute statistics and physical magnitudes from ODE solutions."""

from __future__ import annotations

from typing import Any

import numpy as np

from config import AVAILABLE_STATISTICS
from utils import get_logger

logger = get_logger(__name__)


def compute_statistics(
    x: np.ndarray,
    y: np.ndarray,
    selected: set[str] | None = None,
) -> dict[str, Any]:
    """Compute requested statistics for the primary solution component.

    Args:
        x: Independent variable values (1D) or tuple of grids for 2D.
        y: Solution array — shape ``(n_vars, n_points)`` or ``(n_points,)``
            for 1D, or ``(ny, nx)`` for 2D PDE.
        selected: Set of statistic keys to compute. ``None`` means all.

    Returns:
        Dictionary mapping statistic names to their computed values.
    """
    y_2d = np.atleast_2d(y)
    y_primary = y_2d[0]
    all_stats = selected or set(AVAILABLE_STATISTICS.keys())

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

    if "median" in all_stats:
        results["median"] = float(np.median(y_primary))

    if "l2_norm" in all_stats:
        results["l2_norm"] = _compute_l2_norm(x, y_primary)

    if "dominant_frequency" in all_stats:
        results["dominant_frequency"] = _estimate_dominant_frequency(x, y_primary)

    if "exponential_rate" in all_stats:
        results["exponential_rate"] = _estimate_exponential_rate(x, y_primary)

    if "half_life" in all_stats:
        results["half_life"] = _compute_half_life(x, y_primary)

    if "time_constant" in all_stats:
        results["time_constant"] = _compute_time_constant(x, y_primary)

    if "doubling_time" in all_stats:
        results["doubling_time"] = _compute_doubling_time(x, y_primary)

    if "angular_frequency" in all_stats:
        results["angular_frequency"] = _compute_angular_frequency(x, y_primary)

    logger.debug("Computed statistics: %s", list(results.keys()))
    return results


def _compute_mean(x: np.ndarray, y: np.ndarray) -> float:
    """Weighted mean over the domain using trapezoidal integration.

    Args:
        x: Independent variable values.
        y: Dependent variable values.

    Returns:
        Weighted mean value.
    """
    span = x[-1] - x[0]
    if span == 0:
        return float(np.mean(y))
    return float(np.trapezoid(y, x) / span)


def _compute_rms(x: np.ndarray, y: np.ndarray) -> float:
    """Root mean square over the domain.

    Args:
        x: Independent variable values.
        y: Dependent variable values.

    Returns:
        RMS value.
    """
    span = x[-1] - x[0]
    if span == 0:
        return float(np.sqrt(np.mean(y**2)))
    return float(np.sqrt(np.trapezoid(y**2, x) / span))


def _count_zero_crossings(y: np.ndarray) -> int:
    """Count the number of sign changes in *y*.

    Args:
        y: 1D array of values.

    Returns:
        Number of zero crossings.
    """
    signs = np.sign(y)
    crossings = np.where(np.diff(signs) != 0)[0]
    return len(crossings)


def _estimate_period(x: np.ndarray, y: np.ndarray) -> float | None:
    """Estimate the period of an oscillatory signal via peak detection.

    Args:
        x: Independent variable values.
        y: Dependent variable values.

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
    """Estimate amplitude as half the peak-to-peak range.

    Args:
        y: 1D array of values.

    Returns:
        Estimated amplitude.
    """
    return float((np.max(y) - np.min(y)) / 2.0)


def _compute_l2_norm(x: np.ndarray, y: np.ndarray) -> float:
    """L2 norm over the domain: sqrt(∫f² dx). Common in functional analysis.

    Args:
        x: Independent variable values.
        y: Dependent variable values.

    Returns:
        L2 norm value.
    """
    span = x[-1] - x[0]
    if span == 0:
        return float(np.sqrt(np.mean(y**2)) * len(y))
    return float(np.sqrt(np.trapezoid(y**2, x)))


def _estimate_dominant_frequency(x: np.ndarray, y: np.ndarray) -> float | None:
    """Dominant frequency via FFT (cycles per unit of x).

    Args:
        x: Independent variable values.
        y: Dependent variable values.

    Returns:
        Dominant frequency in cycles per unit, or None if invalid.
    """
    n = len(y)
    if n < 4:
        return None
    y_centered = y - np.mean(y)
    fft_vals = np.fft.rfft(y_centered)
    magnitudes = np.abs(fft_vals)
    magnitudes[0] = 0  # Ignore DC
    if np.max(magnitudes) < 1e-10:
        return None
    peak_idx = int(np.argmax(magnitudes))
    span = x[-1] - x[0]
    if span <= 0:
        return None
    return float(peak_idx / span)


def _estimate_exponential_rate(x: np.ndarray, y: np.ndarray) -> float | None:
    """Fit y = A*exp(λ*x). Returns λ when fit is good (monotonic, all same sign).

    Args:
        x: Independent variable values.
        y: Dependent variable values.

    Returns:
        Exponential rate λ, or None if fit fails.
    """
    if len(y) < 3 or len(x) < 3:
        return None
    if np.any(y <= 0):
        return None
    try:
        from scipy.optimize import curve_fit

        def model(t: np.ndarray, a: float, lam: float) -> np.ndarray:
            return a * np.exp(lam * t)

        span = x[-1] - x[0]
        lam0 = np.log(y[-1] / y[0]) / span if span > 0 and y[0] > 0 else 0.0
        popt, _ = curve_fit(model, x, y, p0=(float(y[0]), float(lam0)))
        return float(popt[1])
    except Exception:
        return None


def _compute_half_life(x: np.ndarray, y: np.ndarray) -> float | None:
    """Half-life for exponential decay: t_1/2 = ln(2)/|λ|.

    Args:
        x: Independent variable values.
        y: Dependent variable values.

    Returns:
        Half-life, or None if not exponential decay.
    """
    lam = _estimate_exponential_rate(x, y)
    if lam is None or lam >= 0:
        return None
    return float(np.log(2) / abs(lam))


def _compute_time_constant(x: np.ndarray, y: np.ndarray) -> float | None:
    """Time constant τ = 1/|λ| for exponential decay (e.g. RC circuit).

    Args:
        x: Independent variable values.
        y: Dependent variable values.

    Returns:
        Time constant, or None if not exponential decay.
    """
    lam = _estimate_exponential_rate(x, y)
    if lam is None or lam >= 0:
        return None
    return float(1.0 / abs(lam))


def _compute_doubling_time(x: np.ndarray, y: np.ndarray) -> float | None:
    """Doubling time t_2 = ln(2)/λ for exponential growth.

    Args:
        x: Independent variable values.
        y: Dependent variable values.

    Returns:
        Doubling time, or None if not exponential growth.
    """
    lam = _estimate_exponential_rate(x, y)
    if lam is None or lam <= 0:
        return None
    return float(np.log(2) / lam)


def _compute_angular_frequency(x: np.ndarray, y: np.ndarray) -> float | None:
    """Angular frequency ω = 2πf (rad per unit of x) from dominant frequency.

    Args:
        x: Independent variable values.
        y: Dependent variable values.

    Returns:
        Angular frequency in rad per unit, or None.
    """
    freq = _estimate_dominant_frequency(x, y)
    if freq is None:
        return None
    return float(2.0 * np.pi * freq)


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

    kinetic = 0.5 * velocity**2
    potential = 0.5 * position**2

    return {
        "kinetic_initial": float(kinetic[0]),
        "potential_initial": float(potential[0]),
        "total_initial": float(kinetic[0] + potential[0]),
        "kinetic_mean": float(np.mean(kinetic)),
        "potential_mean": float(np.mean(potential)),
        "total_mean": float(np.mean(kinetic + potential)),
    }


def compute_statistics_2d(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    u: np.ndarray,
    selected: set[str] | None = None,
) -> dict[str, Any]:
    """Compute statistics for a 2D scalar field u(x,y).

    Args:
        x_grid: 1D array of x values.
        y_grid: 1D array of y values.
        u: 2D array of shape (len(y_grid), len(x_grid)).
        selected: Set of statistic keys. ``None`` means all 2D stats.

    Returns:
        Dictionary with mean, std, max, min, integral_2d.
    """
    all_stats = selected or {"mean", "std", "max", "min", "integral"}
    results: dict[str, Any] = {}

    flat = u.ravel()
    if "mean" in all_stats:
        results["mean"] = float(np.mean(flat))
    if "std" in all_stats:
        results["std"] = float(np.std(flat))
    if "max" in all_stats:
        ij = np.unravel_index(np.argmax(u), u.shape)
        results["max"] = {
            "value": float(u[ij]),
            "x": float(x_grid[ij[1]]),
            "y": float(y_grid[ij[0]]),
        }
    if "min" in all_stats:
        ij = np.unravel_index(np.argmin(u), u.shape)
        results["min"] = {
            "value": float(u[ij]),
            "x": float(x_grid[ij[1]]),
            "y": float(y_grid[ij[0]]),
        }
    if "integral" in all_stats:
        results["integral"] = float(np.trapezoid(np.trapezoid(u, x_grid, axis=1), y_grid, axis=0))

    if "l2_norm" in all_stats:
        integrand = np.trapezoid(np.trapezoid(u**2, x_grid, axis=1), y_grid, axis=0)
        results["l2_norm"] = float(np.sqrt(max(0, integrand)))

    if "gradient_norm" in all_stats:
        results["gradient_norm"] = _compute_gradient_norm_2d(x_grid, y_grid, u)

    return results


def _compute_gradient_norm_2d(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    u: np.ndarray,
) -> float:
    """Mean magnitude of gradient |∇u| over the 2D domain.

    Args:
        x_grid: 1D array of x values.
        y_grid: 1D array of y values.
        u: 2D solution array (ny, nx).

    Returns:
        Mean gradient magnitude.
    """
    uy, ux = np.gradient(u, y_grid, x_grid)
    grad_mag = np.sqrt(ux**2 + uy**2)
    return float(np.mean(grad_mag))
