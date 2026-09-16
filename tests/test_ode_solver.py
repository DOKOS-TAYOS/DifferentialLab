"""Tests for solver.ode_solver."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from solver.equation_parser import _parse_expression
from solver.ode_solver import (
    BVPOptions,
    BVPSolution,
    IVPOptions,
    ODESolution,
    solve_bvp,
    solve_multipoint,
    solve_ode,
)
from utils import ValidationError

# Default env values used when mocking get_env_from_schema
_SOLVER_ENV = {
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
    ode_func = _parse_expression("k * y[0]", order=1, parameters={"k": 0.5})

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
    np.testing.assert_allclose(result.y[0, -1], np.exp(0.5 * sample_domain[1]), rtol=1e-5)


@patch("solver.ode_solver.get_env_from_schema")
def test_solve_ode_uses_env_when_args_none(mock_get_env: object) -> None:
    mock_get_env.side_effect = lambda k: _SOLVER_ENV.get(k, 100)
    ode_func = _parse_expression("y[0]", order=1)
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


@patch("solver.ode_solver.solve_ivp")
@patch("solver.ode_solver.get_env_from_schema")
def test_solve_ode_keeps_zero_max_step_sentinel_with_explicit_t_eval(
    mock_get_env: object,
    mock_solve_ivp: object,
) -> None:
    mock_get_env.side_effect = lambda k: _SOLVER_ENV.get(k, 100)
    mock_solve_ivp.return_value = SimpleNamespace(
        t=np.array([0.0, 1.0]),
        y=np.array([[1.0, 1.0]]),
        success=True,
        message="ok",
        nfev=2,
        njev=0,
        nlu=0,
        status=0,
        t_events=None,
        y_events=None,
    )

    solve_ode(
        lambda _x, y: y,
        (0.0, 1.0),
        [1.0],
        t_eval=np.array([0.0, 1.0]),
        max_step=0,
    )

    assert np.isinf(mock_solve_ivp.call_args.kwargs["max_step"])


@patch("solver.ode_solver.get_env_from_schema")
def test_solve_ode_terminal_and_directional_events(mock_get_env: object) -> None:
    mock_get_env.side_effect = lambda k: _SOLVER_ENV.get(k, 100)

    def upward_event(_x: float, y: np.ndarray) -> float:
        return float(y[0] - 1.0)

    upward_event.terminal = True  # type: ignore[attr-defined]
    upward_event.direction = 1.0  # type: ignore[attr-defined]
    result = solve_ode(
        lambda _x, _y: np.array([1.0]),
        (0.0, 2.0),
        [0.0],
        t_eval=np.linspace(0.0, 2.0, 101),
        options=IVPOptions(events=(upward_event,)),
    )

    assert result.status == 1
    np.testing.assert_allclose(result.t_events[0], [1.0], atol=1e-8)
    np.testing.assert_allclose(result.y_events[0], [[1.0]], atol=1e-8)
    assert result.x[-1] <= 1.0

    def downward_only(_x: float, y: np.ndarray) -> float:
        return float(y[0] - 1.0)

    downward_only.direction = -1.0  # type: ignore[attr-defined]
    no_event = solve_ode(
        lambda _x, _y: np.array([1.0]),
        (0.0, 2.0),
        [0.0],
        t_eval=np.linspace(0.0, 2.0, 21),
        options=IVPOptions(events=(downward_only,)),
    )
    assert no_event.status == 0
    assert no_event.t_events[0].size == 0


@patch("solver.ode_solver.get_env_from_schema")
def test_solve_ode_exposes_scipy_diagnostics_and_jacobian_hook(mock_get_env: object) -> None:
    mock_get_env.side_effect = lambda k: _SOLVER_ENV.get(k, 100)
    result = solve_ode(
        lambda _x, y: -y,
        (0.0, 1.0),
        [1.0],
        method="BDF",
        t_eval=np.linspace(0.0, 1.0, 20),
        options=IVPOptions(
            jac=lambda _x, _y: np.array([[-1.0]]),
            vectorized=False,
            first_step=0.01,
        ),
    )

    assert result.nfev == result.n_eval > 0
    assert result.njev is not None
    assert result.nlu is not None
    assert result.status == 0


@patch("solver.ode_solver.get_env_from_schema")
def test_solve_multipoint_all_at_start_reduces_to_ivp(mock_get_env: object) -> None:
    mock_get_env.side_effect = lambda k: _SOLVER_ENV.get(k, 100)
    ode_func = _parse_expression("-y[0]", order=2, parameters={})
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
    ode_func = _parse_expression("-y[0]", order=2)
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
    # Solution at pi/2 should be close to 0 (shooting target; interpolation at grid)
    idx_mid = np.argmin(np.abs(result.x - np.pi / 2))
    np.testing.assert_allclose(result.y[0, idx_mid], 0.0, atol=0.02)


def test_solve_bvp_known_two_point_problem() -> None:
    def harmonic(_x: float, y: np.ndarray) -> np.ndarray:
        return np.array([y[1], -y[0]])

    def boundary(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        return np.array([ya[0] - 1.0, yb[0]])

    result = solve_bvp(
        harmonic,
        (0.0, np.pi / 2),
        boundary,
        np.array([1.0, 0.0]),
        options=BVPOptions(initial_mesh_points=20, tol=1e-6),
    )

    assert isinstance(result, BVPSolution)
    assert result.success is True
    assert result.niter > 0
    np.testing.assert_allclose(result.y[0], np.cos(result.x), atol=2e-5)


@patch("solver.ode_solver.get_env_from_schema")
def test_multipoint_auto_uses_bvp_for_endpoint_conditions(mock_get_env: object) -> None:
    mock_get_env.side_effect = lambda k: _SOLVER_ENV.get(k, 100)
    ode_func = _parse_expression("-y[0]", order=2)
    result = solve_multipoint(
        ode_func,
        conditions=[(0, 0.0, 1.0), (0, np.pi / 2, 0.0)],
        order=2,
        x_min=0.0,
        x_max=np.pi / 2,
        t_eval=np.linspace(0.0, np.pi / 2, 41),
    )

    assert result.method_used == "BVP"
    np.testing.assert_allclose(result.y[0], np.cos(result.x), atol=2e-4)


@patch("solver.ode_solver.get_env_from_schema")
def test_multipoint_forced_shooting_and_bvp_strategies(mock_get_env: object) -> None:
    mock_get_env.side_effect = lambda k: _SOLVER_ENV.get(k, 100)
    ode_func = _parse_expression("-y[0]", order=2)
    common = {
        "ode_func": ode_func,
        "conditions": [(0, 0.0, 1.0), (0, np.pi / 2, 0.0)],
        "order": 2,
        "x_min": 0.0,
        "x_max": np.pi / 2,
        "t_eval": np.linspace(0.0, np.pi / 2, 31),
    }

    shooting = solve_multipoint(**common, strategy="shooting")
    bvp = solve_multipoint(**common, strategy="bvp")

    assert shooting.method_used != "BVP"
    assert bvp.method_used == "BVP"
    np.testing.assert_allclose(shooting.y[0], np.cos(shooting.x), atol=2e-4)
    np.testing.assert_allclose(bvp.y[0], np.cos(bvp.x), atol=2e-4)


@patch("solver.ode_solver.get_env_from_schema")
def test_multipoint_forced_bvp_rejects_interior_conditions(mock_get_env: object) -> None:
    mock_get_env.side_effect = lambda k: _SOLVER_ENV.get(k, 100)
    ode_func = _parse_expression("-y[0]", order=2)

    with pytest.raises(ValidationError, match="only at x_min or x_max"):
        solve_multipoint(
            ode_func,
            conditions=[(0, 0.0, 1.0), (0, 0.5, 0.0)],
            order=2,
            x_min=0.0,
            x_max=1.0,
            strategy="bvp",
        )


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
        assert sol.nfev == 0
        assert sol.status == 0
        np.testing.assert_array_equal(sol.x, x)
        np.testing.assert_array_equal(sol.y, y)
