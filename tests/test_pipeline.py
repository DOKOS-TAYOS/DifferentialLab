"""Tests for pipeline.run_solver_pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pipeline import SolverResult, _build_mask, run_solver_pipeline
from solver.pde_solver import PDESolution
from utils import EquationParseError, ValidationError


@patch("solver.ode_solver.get_env_from_schema")
def test_run_solver_pipeline_success(
    mock_ode_env: object,
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
    np.testing.assert_allclose(result.y[0, -1], np.exp(0.5 * sample_domain[1]), rtol=1e-5)


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
def test_run_solver_pipeline_can_skip_rhs_recalculation(
    mock_ode_env: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def env_side_effect(key: str) -> object:
        env = {
            "SOLVER_MAX_STEP": 0.0,
            "SOLVER_RTOL": 1e-8,
            "SOLVER_ATOL": 1e-10,
            "SOLVER_NUM_POINTS": 25,
        }
        return env.get(key, 25)

    def fail_rhs_values(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("RHS values should not be recomputed when both flags are disabled")

    mock_ode_env.side_effect = env_side_effect
    monkeypatch.setattr("pipeline._evaluate_ode_rhs_values", fail_rhs_values)

    result = run_solver_pipeline(
        expression="-y[0]",
        function_name=None,
        order=1,
        parameters={},
        equation_name="Decay",
        x_min=0.0,
        x_max=1.0,
        y0=[1.0],
        n_points=25,
        method="RK45",
        selected_stats={"mean"},
        augment_highest_derivative=False,
        compute_residual_metrics=False,
    )

    assert result.y.shape == (1, 25)
    assert result.vector_order == 1
    assert result.metadata["residual_max"] is None


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


def test_build_mask_rejects_unsafe_expression() -> None:
    with pytest.raises(EquationParseError):
        _build_mask(
            "().__class__.__mro__[1].__subclasses__()",
            np.linspace(0.0, 1.0, 5),
            np.linspace(0.0, 1.0, 5),
            {},
        )


def test_run_solver_pipeline_pde_uses_fast_coefficients_for_coordinate_rhs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_solve_pde_2d(*args: object, **kwargs: object) -> PDESolution:
        provider = kwargs.get("coefficient_provider")
        captured["coefficient_provider"] = provider
        assert callable(provider)
        assert provider(0.25, 0.5, {}) == (-1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -0.75)
        x_grid = np.linspace(0.0, 1.0, 5)
        y_grid = np.linspace(0.0, 1.0, 5)
        return PDESolution(
            grid=(x_grid, y_grid),
            u=np.zeros((5, 5)),
            success=True,
            message="OK",
        )

    monkeypatch.setattr("pipeline.solve_pde_2d", fake_solve_pde_2d)

    run_solver_pipeline(
        expression="x + y",
        function_name=None,
        order=1,
        parameters={},
        equation_name="Poisson",
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        y0=[],
        n_points=5,
        n_points_y=5,
        method="fdm",
        selected_stats=set(),
        equation_type="pde",
        variables=["x", "y"],
    )

    assert captured["coefficient_provider"] is not None


def test_run_solver_pipeline_pde_keeps_generic_path_for_solution_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_solve_pde_2d(*args: object, **kwargs: object) -> PDESolution:
        captured["coefficient_provider"] = kwargs.get("coefficient_provider")
        x_grid = np.linspace(0.0, 1.0, 5)
        y_grid = np.linspace(0.0, 1.0, 5)
        return PDESolution(
            grid=(x_grid, y_grid),
            u=np.zeros((5, 5)),
            success=True,
            message="OK",
        )

    monkeypatch.setattr("pipeline.solve_pde_2d", fake_solve_pde_2d)

    run_solver_pipeline(
        expression="f + x",
        function_name=None,
        order=1,
        parameters={},
        equation_name="Reaction diffusion",
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        y0=[],
        n_points=5,
        n_points_y=5,
        method="fdm",
        selected_stats=set(),
        equation_type="pde",
        variables=["x", "y"],
    )

    assert captured["coefficient_provider"] is None


def test_solver_package_import_is_lazy_for_scipy() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(project_root / "src")}
    code = (
        "import sys\n"
        "def has_scipy():\n"
        "    return any(name == 'scipy' or name.startswith('scipy.') for name in sys.modules)\n"
        "import solver\n"
        "print(has_scipy())\n"
        "from solver import solve_ode\n"
        "print(callable(solve_ode))\n"
        "print(has_scipy())\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.splitlines() == ["False", "True", "True"]
