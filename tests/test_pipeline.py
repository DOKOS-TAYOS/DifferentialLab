"""Tests for pipeline.run_solver_pipeline."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from pipeline import SolverResult, run_solver_pipeline
from utils import ValidationError


@patch("solver.ode_solver.get_env_from_schema")
def test_run_solver_pipeline_success(
    mock_ode_env: object,
    tmp_path: object,
    sample_expression_order1: str,
    sample_y0_order1: list[float],
    sample_domain: tuple[float, float],
) -> None:
    def env_side_effect(key: str) -> object:
        env = {
            "SOLVER_MAX_STEP": 0.0,
            "SOLVER_RTOL": 1e-8,
            "SOLVER_ATOL": 1e-10,
            "SOLVER_NUM_POINTS": 100,
        }
        return env.get(key, 100)

    mock_ode_env.side_effect = env_side_effect

    result = run_solver_pipeline(
        expression="k * y[0]",
        function_name=None,
        order=1,
        parameters={"k": 0.5},
        equation_name="Exponential",
        x_min=sample_domain[0],
        x_max=sample_domain[1],
        y0=sample_y0_order1,
        n_points=100,
        method="RK45",
        selected_stats={"mean", "rms", "integral"},
    )

    assert isinstance(result, SolverResult)
    assert result.x.shape == (100,)
    # Augmented with highest derivative: (order+1, n_points) = (2, 100)
    assert result.y.shape == (2, 100)
    assert "mean" in result.statistics
    assert result.metadata["equation_name"] == "Exponential"
    # Verify numerical solution: y'=0.5*y, y(0)=1 => y(x)=exp(0.5*x)
    np.testing.assert_allclose(result.y[0, 0], 1.0)
    np.testing.assert_allclose(
        result.y[0, -1], np.exp(0.5 * sample_domain[1]), rtol=1e-5
    )


def test_run_solver_pipeline_validation_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        run_solver_pipeline(
            expression="",
            function_name=None,
            order=1,
            parameters={},
            equation_name="Bad",
            x_min=0.0,
            x_max=10.0,
            y0=[1.0],
            n_points=5,  # too few points
            method="RK45",
            selected_stats=set(),
        )
    msg = str(exc_info.value).lower()
    assert "empty" in msg, "Expected empty expression error in message"
    assert "10" in msg or "points" in msg, "Expected num_points error in message"


@patch("solver.ode_solver.get_env_from_schema")
def test_run_solver_pipeline_multipoint(
    mock_ode_env: object,
    tmp_path: object,
) -> None:
    def env_side_effect(key: str) -> object:
        env = {
            "SOLVER_MAX_STEP": 0.0,
            "SOLVER_RTOL": 1e-8,
            "SOLVER_ATOL": 1e-10,
            "SOLVER_NUM_POINTS": 50,
        }
        return env.get(key, 50)

    mock_ode_env.side_effect = env_side_effect

    # Standard IVP with x0_list all at start
    result = run_solver_pipeline(
        expression="-y[0]",
        function_name=None,
        order=2,
        parameters={},
        equation_name="Harmonic",
        x_min=0.0,
        x_max=10.0,
        y0=[1.0, 0.0],
        n_points=50,
        method="RK45",
        selected_stats={"mean"},
        x0_list=[0.0, 0.0],
    )
    # Augmented with highest derivative: (order+1, n_points) = (3, 50)
    assert result.y.shape[0] == 3
    np.testing.assert_allclose(result.y[0, 0], 1.0)
    np.testing.assert_allclose(result.y[1, 0], 0.0)


def test_run_solver_pipeline_difference_equation() -> None:
    result = run_solver_pipeline(
        expression="r * y[0]",
        function_name=None,
        order=1,
        parameters={"r": 1.5},
        equation_name="Geometric growth",
        x_min=0,
        x_max=20,
        y0=[1.0],
        n_points=21,
        method="iteration",
        selected_stats={"mean", "max"},
        equation_type="difference",
    )

    assert isinstance(result, SolverResult)
    assert result.x.shape == (21,)
    assert result.y.shape == (1, 21)
    np.testing.assert_allclose(result.y[0, 0], 1.0)
    np.testing.assert_allclose(result.y[0, -1], 1.5**20)
    assert result.metadata["equation_type"] == "difference"


@patch("solver.ode_solver.get_env_from_schema")
def test_run_solver_pipeline_vector_ode(mock_ode_env: object) -> None:
    """Vector ODE: coupled system f0'=f1, f1'=-f0 (harmonic oscillator)."""
    def env_side_effect(key: str) -> object:
        env = {
            "SOLVER_MAX_STEP": 0.0,
            "SOLVER_RTOL": 1e-8,
            "SOLVER_ATOL": 1e-10,
            "SOLVER_NUM_POINTS": 100,
        }
        return env.get(key, 100)

    mock_ode_env.side_effect = env_side_effect

    result = run_solver_pipeline(
        expression=None,
        function_name=None,
        order=1,
        parameters={},
        equation_name="Vector harmonic",
        x_min=0.0,
        x_max=2 * np.pi,
        y0=[1.0, 0.0],
        n_points=100,
        method="RK45",
        selected_stats={"mean", "rms"},
        equation_type="vector_ode",
        vector_expressions=["y[1]", "-y[0]"],
        vector_components=2,
    )

    assert isinstance(result, SolverResult)
    assert result.metadata["equation_type"] == "vector_ode"
    assert result.is_vector is True
    assert result.x.shape == (100,)
    np.testing.assert_allclose(result.y[0, 0], 1.0)
    np.testing.assert_allclose(result.y[1, 0], 0.0)
    # cos(2π) ≈ 1, sin(2π) ≈ 0
    np.testing.assert_allclose(result.y[0, -1], 1.0, atol=1e-5)
    np.testing.assert_allclose(result.y[1, -1], 0.0, atol=1e-5)


def test_run_solver_pipeline_pde_2d() -> None:
    """PDE 2D: Laplace -f_xx - f_yy = 0 with zero BC."""
    result = run_solver_pipeline(
        expression="0",
        function_name=None,
        order=1,
        parameters={},
        equation_name="Laplace",
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        y0=[],
        n_points=11,
        n_points_y=11,
        method="fdm",
        selected_stats={"mean", "std"},
        equation_type="pde",
        variables=["x", "y"],
    )

    assert isinstance(result, SolverResult)
    assert result.metadata["equation_type"] == "pde"
    assert result.y_grid is not None
    assert result.x.shape == (11,)
    assert result.y_grid.shape == (11,)
    assert result.y.shape == (11, 11)
    np.testing.assert_allclose(result.y, 0.0, atol=1e-10)
