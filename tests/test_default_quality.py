"""Regression coverage for meaningful built-in defaults."""

from __future__ import annotations

from math import isfinite

import numpy as np
import pytest

from pipeline import run_solver_pipeline
from solver.equation_parser import (
    get_difference_function,
    get_ode_function,
    get_vector_ode_function,
    parse_pde_3d_residual_expression,
    parse_pde_rhs_expression,
    parse_vector_pde_residual_expressions,
)
from solver.predefined import load_predefined_equations


def _catalog_parameters(key: str) -> dict[str, float]:
    return {
        name: float(info["default"])
        for name, info in load_predefined_equations()[key].parameters.items()
    }


def test_simple_pendulum_default_has_physical_gravity_sign() -> None:
    equation = load_predefined_equations()["simple_pendulum"]
    assert _catalog_parameters("simple_pendulum")["g"] > 0
    function = get_ode_function(
        expression=equation.expression,
        function_name=equation.function_name,
        order=equation.order,
        parameters=_catalog_parameters("simple_pendulum"),
    )
    acceleration = function(0.0, np.asarray(equation.default_initial_conditions))[1]
    assert acceleration < 0.0


def test_allee_default_starts_below_threshold_and_declines() -> None:
    equation = load_predefined_equations()["allee_effect"]
    parameters = _catalog_parameters("allee_effect")
    y0 = equation.default_initial_conditions[0]
    assert y0 != parameters["m"] * parameters["K"]
    function = get_ode_function(
        function_name=equation.function_name,
        order=equation.order,
        parameters=parameters,
    )
    assert function(0.0, np.asarray(equation.default_initial_conditions))[0] < 0.0


def test_laplace_default_preset_solves_to_x() -> None:
    equation = load_predefined_equations()["laplace_2d"]
    conditions = equation.default_boundary_conditions
    result = run_solver_pipeline(
        expression=equation.expression,
        function_name=equation.function_name,
        order=equation.order,
        parameters={},
        equation_name=equation.name,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        y0=[],
        n_points=11,
        n_points_y=11,
        method="fdm",
        selected_stats=set(),
        equation_type=equation.equation_type,
        variables=equation.variables,
        bc_expressions=[
            conditions[face]["expression"] for face in ("bottom", "top", "left", "right")
        ],
        bc_types=[conditions[face]["type"].lower() for face in ("bottom", "top", "left", "right")],
    )
    assert np.isfinite(result.y).all()
    assert not np.allclose(result.y, 0.0)
    assert result.y[5, 5] == pytest.approx(0.5, abs=1e-12)
    np.testing.assert_allclose(result.y, np.broadcast_to(result.x, result.y.shape), atol=0.03)


def test_manufactured_vector_pde_default_is_nontrivial() -> None:
    equation = load_predefined_equations()["coupled_elliptic_system_2d"]
    result = run_solver_pipeline(
        expression=equation.expression,
        function_name=equation.function_name,
        order=equation.order,
        parameters={},
        equation_name=equation.name,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        y0=[],
        n_points=11,
        n_points_y=11,
        method="fdm",
        selected_stats=set(),
        equation_type=equation.equation_type,
        variables=equation.variables,
        vector_expressions=equation.vector_expressions,
        vector_components=equation.vector_components,
        bc_expressions=["0", "0", "0", "0"],
        bc_types=["dirichlet"] * 4,
    )
    assert np.isfinite(result.y).all()
    assert not np.allclose(result.y[0], 0.0)
    assert not np.allclose(result.y[1], 0.0)
    assert result.y[0, 5, 5] == pytest.approx(1.0, abs=0.04)
    assert result.y[1, 5, 5] == pytest.approx(-1.0, abs=0.04)


def test_catalog_non_pde_defaults_resolve_and_start_finitely() -> None:
    equations = load_predefined_equations()
    for equation in equations.values():
        if equation.equation_type == "ode":
            function = get_ode_function(
                expression=equation.expression,
                function_name=equation.function_name,
                order=equation.order,
                parameters=_catalog_parameters(equation.key),
            )
            values = function(0.0, np.asarray(equation.default_initial_conditions, dtype=float))
            assert np.isfinite(values).all(), equation.key
        elif equation.equation_type == "vector_ode":
            function = get_vector_ode_function(
                vector_expressions=equation.vector_expressions or [],
                function_name=equation.function_name,
                order=equation.order,
                vector_components=equation.vector_components,
                parameters=_catalog_parameters(equation.key),
            )
            values = function(0.0, np.asarray(equation.default_initial_conditions, dtype=float))
            assert np.isfinite(values).all(), equation.key
        elif equation.equation_type == "difference":
            function = get_difference_function(
                expression=equation.expression,
                function_name=equation.function_name,
                order=equation.order,
                parameters=_catalog_parameters(equation.key),
            )
            assert isfinite(
                function(0, np.asarray(equation.default_initial_conditions, dtype=float))
            ), equation.key
        elif equation.equation_type == "pde":
            function = parse_pde_rhs_expression(
                equation.expression or "0", equation.variables, _catalog_parameters(equation.key)
            )
            assert isfinite(function(*([0.0] * len(equation.variables)))), equation.key
        elif equation.equation_type == "pde_3d":
            function = parse_pde_3d_residual_expression(
                equation.expression or "0", equation.variables, _catalog_parameters(equation.key)
            )
            assert isfinite(
                function(
                    0.0,
                    0.0,
                    0.0,
                    f=0.0,
                    fx=0.0,
                    fy=0.0,
                    fz=0.0,
                    fxx=0.0,
                    fxy=0.0,
                    fxz=0.0,
                    fyy=0.0,
                    fyz=0.0,
                    fzz=0.0,
                )
            ), equation.key
        elif equation.equation_type == "vector_pde":
            function = parse_vector_pde_residual_expressions(
                equation.vector_expressions or [],
                equation.vector_components,
                equation.variables,
                _catalog_parameters(equation.key),
            )
            values = function(
                0.0,
                0.0,
                f=np.zeros(equation.vector_components),
                fx=np.zeros(equation.vector_components),
                fy=np.zeros(equation.vector_components),
                fxx=np.zeros(equation.vector_components),
                fxy=np.zeros(equation.vector_components),
                fyy=np.zeros(equation.vector_components),
            )
            assert np.isfinite(values).all(), equation.key
