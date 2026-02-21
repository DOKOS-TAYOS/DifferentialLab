"""Tests for pipeline.run_solver_pipeline."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from pipeline import SolverResult, run_solver_pipeline
from utils import ValidationError


@patch("config.paths.get_env_from_schema")
@patch("solver.ode_solver.get_env_from_schema")
def test_run_solver_pipeline_success(
    mock_ode_env: object,
    mock_paths_env: object,
    tmp_path: object,
    sample_expression_order1: str,
    sample_y0_order1: list[float],
    sample_domain: tuple[float, float],
) -> None:
    def env_side_effect(key: str) -> object:
        env = {
            "SOLVER_DEFAULT_METHOD": "RK45",
            "SOLVER_MAX_STEP": 0.0,
            "SOLVER_RTOL": 1e-8,
            "SOLVER_ATOL": 1e-10,
            "SOLVER_NUM_POINTS": 100,
            "FILE_OUTPUT_DIR": str(tmp_path),
            "FILE_PLOT_FORMAT": "png",
        }
        return env.get(key, 100)

    mock_ode_env.side_effect = env_side_effect
    mock_paths_env.side_effect = env_side_effect

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
    assert result.y.shape == (1, 100)
    assert "mean" in result.statistics
    assert result.metadata["equation_name"] == "Exponential"
    assert result.csv_path.exists()
    assert result.json_path.exists()
    assert result.plot_path.exists()


def test_run_solver_pipeline_validation_error() -> None:
    with pytest.raises(ValidationError):
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


@patch("config.paths.get_env_from_schema")
@patch("solver.ode_solver.get_env_from_schema")
def test_run_solver_pipeline_multipoint(
    mock_ode_env: object,
    mock_paths_env: object,
    tmp_path: object,
) -> None:
    def env_side_effect(key: str) -> object:
        env = {
            "SOLVER_DEFAULT_METHOD": "RK45",
            "SOLVER_MAX_STEP": 0.0,
            "SOLVER_RTOL": 1e-8,
            "SOLVER_ATOL": 1e-10,
            "SOLVER_NUM_POINTS": 50,
            "FILE_OUTPUT_DIR": str(tmp_path),
            "FILE_PLOT_FORMAT": "png",
        }
        return env.get(key, 50)

    mock_ode_env.side_effect = env_side_effect
    mock_paths_env.side_effect = env_side_effect

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
    assert result.y.shape[0] == 2
    np.testing.assert_allclose(result.y[0, 0], 1.0)
    np.testing.assert_allclose(result.y[1, 0], 0.0)


@patch("config.paths.get_env_from_schema")
def test_run_solver_pipeline_difference_equation(
    mock_paths_env: object,
    tmp_path: object,
) -> None:
    def env_side_effect(key: str) -> object:
        env = {
            "FILE_OUTPUT_DIR": str(tmp_path),
            "FILE_PLOT_FORMAT": "png",
        }
        return env.get(key, 100)

    mock_paths_env.side_effect = env_side_effect

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
