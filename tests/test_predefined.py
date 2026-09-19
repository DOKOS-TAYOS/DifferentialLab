"""Tests for solver.predefined."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from unittest.mock import patch

import pytest

from solver.predefined import PredefinedEquation, load_predefined_equations


def test_predefined_equation_dataclass() -> None:
    eq = PredefinedEquation(
        key="test",
        name="Test Eq",
        formula="y' = y",
        description="Desc",
        order=1,
        parameters={"k": {"default": 1.0, "description": "rate"}},
        expression="k * y[0]",
        function_name=None,
        default_initial_conditions=[1.0],
        default_domain=[0.0, 10.0],
    )
    assert eq.formula == "y' = y"
    assert eq.key == "test"
    assert eq.order == 1
    assert eq.default_domain == [0.0, 10.0]
    assert eq.default_boundary_conditions == {}


def test_load_predefined_equations_returns_dict() -> None:
    equations = load_predefined_equations()
    assert isinstance(equations, dict)
    assert len(equations) >= 1


def test_load_predefined_equations_entries_are_predefined_equation() -> None:
    equations = load_predefined_equations()
    for key, eq in equations.items():
        assert isinstance(eq, PredefinedEquation)
        assert eq.key == key
        assert eq.order >= 1
        assert eq.formula
        vec_exprs = getattr(eq, "vector_expressions", None)
        has_defn = eq.expression or eq.function_name or (vec_exprs and len(vec_exprs) > 0)
        assert has_defn
        eq_type = getattr(eq, "equation_type", "ode")
        is_pde = eq_type in ("pde", "pde_3d", "vector_pde")
        is_vector = (vec_exprs is not None and len(vec_exprs) > 0) or eq_type in (
            "vector_ode",
            "vector_pde",
        )
        if not is_pde:
            vec_comp = getattr(eq, "vector_components", 1)
            expected_ic_len = eq.order * vec_comp if is_vector else eq.order
            assert len(eq.default_initial_conditions) == expected_ic_len
            assert len(eq.default_domain) >= 2
        elif eq_type == "pde_3d":
            assert len(eq.default_domain) >= 6
        else:
            assert len(eq.default_domain) >= 4  # x_min, x_max, y_min, y_max


def test_load_predefined_equations_known_keys() -> None:
    equations = load_predefined_equations()
    # From config/equations/*.yaml
    assert "harmonic_oscillator" in equations
    assert "exponential_growth" in equations or "damped_oscillator" in equations
    assert "coupled_elliptic_system_2d" in equations
    assert {
        "poisson_sine_3d",
        "localized_heat_source_3d",
        "anisotropic_diffusion_3d",
        "screened_poisson_3d",
    } <= equations.keys()


def test_catalog_boundary_defaults_are_valid_and_laplace_is_nontrivial() -> None:
    equations = load_predefined_equations()
    laplace = equations["laplace_2d"]
    assert laplace.default_boundary_conditions == {
        "bottom": {"type": "Dirichlet", "expression": "x"},
        "top": {"type": "Dirichlet", "expression": "x"},
        "left": {"type": "Dirichlet", "expression": "0"},
        "right": {"type": "Dirichlet", "expression": "1"},
    }

    valid_faces = {
        "pde": {"bottom", "top", "left", "right"},
        "vector_pde": {"bottom", "top", "left", "right"},
        "pde_3d": {"z_min", "z_max", "y_min", "y_max", "x_min", "x_max"},
    }
    for equation in equations.values():
        assert all(isfinite(float(value)) for value in equation.default_domain)
        assert all(isfinite(float(value)) for value in equation.default_initial_conditions)
        assert all(
            isfinite(float(info.get("default", 0.0))) for info in equation.parameters.values()
        )
        assert all(
            equation.default_domain[index] < equation.default_domain[index + 1]
            for index in range(0, len(equation.default_domain), 2)
        )
        for face, condition in equation.default_boundary_conditions.items():
            assert face in valid_faces[equation.equation_type]
            assert condition["type"] in {"Dirichlet", "Neumann"}
            assert isinstance(condition["expression"], str)


def test_predefined_pde_3d_entries_have_required_metadata() -> None:
    equations = load_predefined_equations()
    keys = (
        "poisson_sine_3d",
        "localized_heat_source_3d",
        "anisotropic_diffusion_3d",
        "screened_poisson_3d",
    )

    for key in keys:
        equation = equations[key]
        assert equation.equation_type == "pde_3d"
        assert equation.variables == ["x", "y", "z"]
        assert equation.default_initial_conditions == []
        assert len(equation.default_domain) == 6


def test_load_predefined_equations_cached() -> None:
    first = load_predefined_equations()
    second = load_predefined_equations()
    assert first is second


def test_load_predefined_equations_missing_file_raises() -> None:
    import solver.predefined as mod

    with patch("solver.predefined._EQUATIONS_DIR", Path("/nonexistent/equations")):
        mod._cache = None
        with pytest.raises(FileNotFoundError):
            load_predefined_equations()
    # Restore cache so other tests get the real equations
    mod._cache = None
    load_predefined_equations()
