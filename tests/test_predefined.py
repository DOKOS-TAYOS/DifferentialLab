"""Tests for solver.predefined."""

from __future__ import annotations

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
        assert eq.expression or eq.function_name
        is_pde = getattr(eq, "equation_type", "ode") == "pde"
        if not is_pde:
            assert len(eq.default_initial_conditions) == eq.order
            assert len(eq.default_domain) >= 2
        else:
            assert len(eq.default_domain) >= 4  # x_min, x_max, y_min, y_max


def test_load_predefined_equations_known_keys() -> None:
    equations = load_predefined_equations()
    # From equations.yaml
    assert "harmonic_oscillator" in equations
    assert "exponential_growth" in equations or "damped_oscillator" in equations


def test_load_predefined_equations_cached() -> None:
    first = load_predefined_equations()
    second = load_predefined_equations()
    assert first is second


def test_load_predefined_equations_missing_file_raises() -> None:
    with patch("solver.predefined._EQUATIONS_PATH", Path("/nonexistent/equations.yaml")):
        # Clear cache so load is attempted
        import solver.predefined as mod
        mod._cache = None
        with pytest.raises(FileNotFoundError):
            load_predefined_equations()
        # Restore cache for other tests (next load will use real path again)
        mod._cache = None
