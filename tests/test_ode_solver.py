"""Tests for solver.ode_solver."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from solver.ode_solver import ODESolution, solve_ode, solve_multipoint
from solver.equation_parser import parse_expression
from utils import SolverFailedError


# Default env values used when mocking get_env_from_schema
_SOLVER_ENV = {
    "SOLVER_DEFAULT_METHOD": "RK45",
    "SOLVER_MAX_STEP": 0.0,
    "SOLVER_RTOL": 1e-8,
    "SOLVER_ATOL": 1e-10,
    "SOLVER_NUM_POINTS": 100,
}


@patch("solver.ode_solver.get_env_from_schema")
def test_solve_ode_success(
    mock_get_env: object,
    sample_domain: tuple[float, float],
    sample_y0_order1: list[float],
    sample_t_eval: np.ndarray,
) -> None:
    mock_get_env.side_effect = lambda k: _SOLVER_ENV.get(k, 100)
    ode_func = parse_expression("0.5 * y[0]", order=1, parameters={"k": 0.5})
    # Actually use y0 passed in; expression is k*y[0], so we need param
    ode_func = parse_expression("k * y[0]", order=1, parameters={"k": 0.5})

    result = solve_ode(
        ode_func,
        t_span=sample_domain,
        y0=sample_y0_order1,
        method="RK45",
        t_eval=sample_t_eval,
    )

    assert isinstance(result, ODESolution)
    assert result.success is True
    assert len(result.x) == len(sample_t_eval)
    assert result.y.shape == (1, len(sample_t_eval))
    np.testing.assert_allclose(result.y[0, 0], 1.0)
    # Exponential growth y'=0.5*y, y(0)=1 => y(10) = exp(5)
    np.testing.assert_allclose(
        result.y[0, -1], np.exp(0.5 * sample_domain[1]), rtol=1e-5
    )


@patch("solver.ode_solver.get_env_from_schema")
def test_solve_ode_uses_env_when_args_none(mock_get_env: object) -> None:
    mock_get_env.side_effect = lambda k: _SOLVER_ENV.get(k, 100)
    ode_func = parse_expression("y[0]", order=1)
    result = solve_ode(
        ode_func,
        t_span=(0.0, 1.0),
        y0=[1.0],
        method=None,
        t_eval=None,
        max_step=None,
        rtol=None,
        atol=None,
    )
    assert result.success is True
    assert len(result.x) == 100  # from SOLVER_NUM_POINTS


@patch("solver.ode_solver.get_env_from_schema")
def test_solve_multipoint_all_at_start_reduces_to_ivp(mock_get_env: object) -> None:
    mock_get_env.side_effect = lambda k: _SOLVER_ENV.get(k, 100)
    ode_func = parse_expression("-y[0]", order=2, parameters={})
    conditions = [(0, 0.0, 1.0), (1, 0.0, 0.0)]  # all at x=0
    result = solve_multipoint(
        ode_func,
        conditions=conditions,
        order=2,
        x_min=0.0,
        x_max=10.0,
        method="RK45",
        t_eval=np.linspace(0, 10, 50),
    )
    assert result.success is True
    assert result.y.shape[0] == 2


@patch("solver.ode_solver.get_env_from_schema")
def test_solve_multipoint_different_points(mock_get_env: object) -> None:
    mock_get_env.side_effect = lambda k: _SOLVER_ENV.get(k, 100)
    # Multipoint: y'' = -y with y(0)=1 and y(pi/2)=0. Shooting finds y'(0).
    ode_func = parse_expression("-y[0]", order=2)
    conditions = [(0, 0.0, 1.0), (0, np.pi / 2, 0.0)]
    t_eval = np.linspace(0, np.pi, 100)
    result = solve_multipoint(
        ode_func,
        conditions=conditions,
        order=2,
        x_min=0.0,
        x_max=np.pi,
        method="RK45",
        t_eval=t_eval,
    )
    assert result.success is True
    assert result.y.shape[0] == 2
    np.testing.assert_allclose(result.y[0, 0], 1.0, atol=1e-5)
    # Solution at pi/2 should be close to 0 (shooting target)
    idx_mid = np.argmin(np.abs(result.x - np.pi / 2))
    np.testing.assert_allclose(result.y[0, idx_mid], 0.0, atol=0.02)


class TestODESolution:
    def test_dataclass_fields(self) -> None:
        x = np.array([0.0, 1.0])
        y = np.array([[1.0, 2.0]])
        sol = ODESolution(
            x=x,
            y=y,
            success=True,
            message="ok",
            method_used="RK45",
            n_eval=10,
        )
        assert sol.success is True
        assert sol.method_used == "RK45"
        assert sol.n_eval == 10
        np.testing.assert_array_equal(sol.x, x)
        np.testing.assert_array_equal(sol.y, y)
