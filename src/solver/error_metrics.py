"""Compute error and quality metrics for ODE solutions."""

from __future__ import annotations

from typing import Callable

import numpy as np

from utils import get_logger

logger = get_logger(__name__)


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
        return {"residual_max": 0.0, "residual_mean": 0.0, "residual_rms": 0.0}

    # Numerical derivative dy/dx
    dy_dx = np.gradient(y_2d, x, axis=1)

    residuals = np.zeros(n_points)
    for i in range(n_points):
        y_i = y_2d[:, i].copy()
        f_val = np.asarray(ode_func(float(x[i]), y_i), dtype=float).ravel()
        diff = f_val - dy_dx[:, i]
        residuals[i] = float(np.linalg.norm(diff))

    return {
        "residual_max": float(np.max(residuals)),
        "residual_mean": float(np.mean(residuals)),
        "residual_rms": float(np.sqrt(np.mean(residuals**2))),
    }

