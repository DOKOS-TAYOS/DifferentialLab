"""Tests for solver.error_metrics."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from solver.equation_parser import _parse_expression
from solver.error_metrics import compute_ode_residual_error
from solver.ode_solver import solve_ode

_SOLVER_ENV = {
    "SOLVER_MAX_STEP": 0.0,
    "SOLVER_RTOL": 1e-8,
    "SOLVER_ATOL": 1e-10,
    "SOLVER_NUM_POINTS": 100,
}


@patch("solver.ode_solver.get_env_from_schema")
def test_exact_ode_low_residual(mock_get_env: object) -> None:
    """ODE y'=y with exact solution exp(x) should have low residual."""
    mock_get_env.side_effect = lambda k: _SOLVER_ENV.get(k, 100)
    ode_func = _parse_expression("y[0]", order=1)
    x = np.linspace(0.0, 2.0, 200)
    # Use solve_ode to get a high-quality numerical solution
    result = solve_ode(
        ode_func,
        t_span=(0.0, 2.0),
        y0=[1.0],
        method="RK45",
        t_eval=x,
    )
    assert result.success is True
    metrics = compute_ode_residual_error(ode_func, result.x, result.y)
    assert "residual_max" in metrics
    assert "residual_mean" in metrics
    assert "residual_rms" in metrics
    # RK45 solution: residual from numerical gradient, typically < 0.05
    assert metrics["residual_max"] < 0.05
    assert metrics["residual_mean"] < 0.02


def test_wrong_solution_high_residual() -> None:
    """Wrong solution y=1 (constant) for y'=y should yield large residual."""
    def ode_func(t: float, y: np.ndarray) -> np.ndarray:
        return np.array([y[0]])

    x = np.linspace(0, 1, 50)
    # y=1 constant: ODE says dy/dx=1, numerical gradient≈0 -> residual≈1
    y_wrong = np.ones((1, 50))
    metrics = compute_ode_residual_error(ode_func, x, y_wrong)
    assert metrics["residual_max"] > 0.5
    assert metrics["residual_mean"] > 0.1


def test_residual_keys_and_shape() -> None:
    """Verify return dict structure and handling of 1D y with exact solution."""
    def ode_func(t: float, y: np.ndarray) -> np.ndarray:
        return np.array([-y[0]])

    x = np.linspace(0, 1, 50)
    y = np.exp(-x).reshape(1, -1)  # Exact solution to y' = -y
    metrics = compute_ode_residual_error(ode_func, x, y)
    assert set(metrics.keys()) == {"residual_max", "residual_mean", "residual_rms"}
    assert metrics["residual_mean"] >= 0
    assert metrics["residual_rms"] >= 0
    # Exact solution: residual from numerical gradient only, should be small
    assert metrics["residual_max"] < 0.02
    # Detect "always returns 0" bug: residual must be positive for non-trivial case
    assert metrics["residual_max"] > 1e-10


def test_few_points_returns_zeros() -> None:
    """n_points < 2 should return zero metrics (cannot compute derivative)."""
    def ode_func(t: float, y: np.ndarray) -> np.ndarray:
        return y

    metrics = compute_ode_residual_error(ode_func, np.array([0.0]), np.array([[1.0]]))
    assert set(metrics.keys()) == {"residual_max", "residual_mean", "residual_rms"}
    assert metrics["residual_max"] == 0.0
    assert metrics["residual_mean"] == 0.0
    assert metrics["residual_rms"] == 0.0
