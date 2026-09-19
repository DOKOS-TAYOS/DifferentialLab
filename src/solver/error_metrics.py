"""Compute error and quality metrics for ODE solutions."""

from __future__ import annotations

from typing import Callable

import numpy as np

_ZERO_RESIDUAL_METRICS = {"residual_max": 0.0, "residual_mean": 0.0, "residual_rms": 0.0}


def compute_ode_residual_error_from_rhs(
    x: np.ndarray,
    y: np.ndarray,
    rhs_values: np.ndarray,
) -> dict[str, float]:
    """Compute residual metrics from precomputed ODE right-hand-side values.

    Args:
        x: Independent variable values.
        y: Solution array of shape (n_vars, n_points).
        rhs_values: Precomputed ``f(x_i, y_i)`` values with the same shape as ``y``.

    Returns:
        Dict with residual_max, residual_mean, residual_rms (L2 norm per point).
    """
    y_2d = np.atleast_2d(y)
    rhs_2d = np.atleast_2d(rhs_values)
    n_points = y_2d.shape[1]

    if n_points < 2:
        return dict(_ZERO_RESIDUAL_METRICS)
    if rhs_2d.shape != y_2d.shape:
        raise ValueError("rhs_values must have the same shape as y")

    dy_dx = np.gradient(y_2d, x, axis=1)
    residuals = np.linalg.norm(rhs_2d - dy_dx, axis=0)
    return {
        "residual_max": float(np.max(residuals)),
        "residual_mean": float(np.mean(residuals)),
        "residual_rms": float(np.sqrt(np.mean(residuals**2))),
    }


def compute_ode_residual_error(
    ode_func: Callable[[float, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Compute residual error: how well does y satisfy dy/dx = f(x, y)?

    Compares the ODE right-hand side f(x,y) with the numerical derivative
    of the solution. Large residuals indicate the solution may not
    satisfy the ODE well (e.g. coarse tolerances, stiff problem).

    Args:
        ode_func: Right-hand side f(x, y) of the ODE dy/dx = f(x, y).
        x: Independent variable values.
        y: Solution array of shape (n_vars, n_points).

    Returns:
        Dict with residual_max, residual_mean, residual_rms (L2 norm per point).
    """
    y_2d = np.atleast_2d(y)
    n_vars, n_points = y_2d.shape

    if n_points < 2:
        return dict(_ZERO_RESIDUAL_METRICS)

    rhs_values = np.empty((n_vars, n_points), dtype=float)
    for i in range(n_points):
        y_i = y_2d[:, i].copy()
        rhs_values[:, i] = np.asarray(ode_func(float(x[i]), y_i), dtype=float).ravel()

    return compute_ode_residual_error_from_rhs(x, y_2d, rhs_values)
