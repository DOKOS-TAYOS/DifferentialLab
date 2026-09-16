"""Tests for solver.equation_parser."""

from __future__ import annotations

import numpy as np
import pytest

from solver.equation_parser import (
    _parse_expression,
    _validate_expression,
    normalize_unicode_escapes,
    parse_ode_event_expression,
    parse_vector_pde_residual_expressions,
)
from utils import EquationParseError


class TestNormalizeUnicodeEscapes:
    def test_empty_string(self) -> None:
        assert normalize_unicode_escapes("") == ""

    def test_no_escapes(self) -> None:
        assert normalize_unicode_escapes("y[0] + x") == "y[0] + x"

    def test_single_escape(self) -> None:
        # \u03C9 is Unicode omega
        assert normalize_unicode_escapes(r"\u03C9") == "ω"

    def test_expression_with_escape(self) -> None:
        assert normalize_unicode_escapes(r"\u03C9**2 * y[0]") == "ω**2 * y[0]"

    def test_multiple_escapes(self) -> None:
        # \u03B1 = α, \u03B2 = β
        result = normalize_unicode_escapes(r"\u03B1 + \u03B2")
        assert result == "α + β"


class TestValidateExpression:
    def test_empty_returns_error(self) -> None:
        errors = _validate_expression("")
        assert len(errors) == 1
        assert "empty" in errors[0].lower()

    def test_whitespace_only_returns_error(self) -> None:
        errors = _validate_expression("   \n\t  ")
        assert len(errors) == 1
        assert "empty" in errors[0].lower()

    def test_valid_expression_returns_no_errors(self) -> None:
        errors = _validate_expression("y[0] * 2 + x")
        assert errors == []

    def test_safe_math_calls_and_subscripts_remain_allowed(self) -> None:
        ode_func = _parse_expression("sin(x) + heaviside(y[0], 0.0)", order=1)

        result = ode_func(np.pi / 2, np.array([1.0]))

        np.testing.assert_allclose(result, [2.0])

    def test_dunder_attribute_escape_is_rejected(self) -> None:
        errors = _validate_expression("().__class__.__mro__[1].__subclasses__()")

        assert errors
        assert "disallowed" in errors[0].lower() or "unsafe" in errors[0].lower()

    def test_syntax_error_reported(self) -> None:
        errors = _validate_expression("y[0] + (")
        assert len(errors) == 1
        assert "syntax" in errors[0].lower() or "error" in errors[0].lower()

    def test_disallowed_construct_reported(self) -> None:
        # Lambda is not in allowed AST nodes
        errors = _validate_expression("(lambda x: x)(1)")
        assert len(errors) == 1
        assert "disallowed" in errors[0].lower() or "construct" in errors[0].lower()

    def test_strips_whitespace(self) -> None:
        errors = _validate_expression("  y[0] + 1  ")
        assert errors == []


class TestParseExpression:
    def test_order1_parses_and_evaluates(
        self,
        sample_expression_order1: str,
        sample_y0_order1: list[float],
        sample_parameters: dict[str, float],
    ) -> None:
        params = {"k": 0.5}
        ode_func = _parse_expression(sample_expression_order1, order=1, parameters=params)
        x, y = 0.0, np.array([1.0])
        dydx = ode_func(x, y)
        assert dydx.shape == (1,)
        np.testing.assert_allclose(dydx, [0.5])  # k * y[0] = 0.5 * 1 = 0.5

    def test_order2_parses_and_evaluates(
        self,
        sample_expression_order2: str,
        sample_y0_order2: list[float],
    ) -> None:
        params = {"omega": 1.0}
        ode_func = _parse_expression(sample_expression_order2, order=2, parameters=params)
        x, y = 0.0, np.array([1.0, 0.0])
        dydx = ode_func(x, y)
        assert dydx.shape == (2,)
        assert dydx[0] == 0.0  # y' = y[1]
        np.testing.assert_allclose(dydx[1], -1.0)  # y'' = -omega**2 * y[0]

    def test_unicode_escape_in_expression(self) -> None:
        # Expression uses omega; param key must be ASCII for Python dict
        ode_func = _parse_expression("omega**2 * y[0]", order=1, parameters={"omega": 2.0})
        x, y = 0.0, np.array([1.0])
        dydx = ode_func(x, y)
        np.testing.assert_allclose(dydx, [4.0])

    def test_invalid_expression_raises(self) -> None:
        with pytest.raises(EquationParseError):
            _parse_expression("y[0] + ", order=1)

    def test_empty_parameters_allowed(self) -> None:
        ode_func = _parse_expression("y[0]", order=1, parameters=None)
        x, y = 0.0, np.array([3.0])
        dydx = ode_func(x, y)
        np.testing.assert_allclose(dydx, [3.0])

    def test_parameter_name_with_dunder_is_rejected(self) -> None:
        with pytest.raises(EquationParseError, match="Unsafe parameter name"):
            _parse_expression("k * y[0]", order=1, parameters={"__class__": 1.0})


class TestParseODEEventExpression:
    def test_scalar_finite_event_is_valid(self) -> None:
        event = parse_ode_event_expression(
            "sin(x) + f[0] - threshold", state_size=2, parameters={"threshold": 0.5}
        )

        result = event(np.pi / 2, np.array([0.25, 0.0]))

        np.testing.assert_allclose(result, 0.75)

    @pytest.mark.parametrize(
        "expression",
        [
            "y",
            "[y[0], y[1]]",
            "(y[0],)",
            "1j",
            "1e309",
            "True",
        ],
    )
    def test_non_scalar_or_non_real_finite_event_is_rejected_during_parse(
        self,
        expression: str,
    ) -> None:
        with pytest.raises(EquationParseError, match="exactly one finite real scalar"):
            parse_ode_event_expression(expression, state_size=2)


class TestVectorPDEResidualParser:
    def test_exact_component_notation_returns_vector(self) -> None:
        residual = parse_vector_pde_residual_expressions(
            [
                "fxx[0] + fyy[0] + 0.5*f[1] + x",
                "fxy[0] + fxx[1] + fyy[1] - 0.25*fy[0] + y",
            ],
            2,
            ["x", "y"],
        )
        zeros = np.zeros(2)
        result = residual(
            0.2,
            0.3,
            np.array([1.0, 2.0]),
            zeros,
            np.array([4.0, 0.0]),
            np.array([3.0, 5.0]),
            np.array([7.0, 0.0]),
            np.array([11.0, 13.0]),
        )

        np.testing.assert_allclose(result, [15.2, 24.3])

    @pytest.mark.parametrize(
        ("expression", "message"),
        [
            ("f[2]", "outside"),
            ("fx[-1]", "integer literals"),
            ("f[0, 1]", "integer literals"),
            ("f[i]", "integer literals"),
            ("f", "explicit component index"),
            ("weights[0]", "subscripts only"),
            ("f[0].real", "Disallowed construct"),
        ],
    )
    def test_invalid_or_unsafe_component_access_is_rejected(
        self,
        expression: str,
        message: str,
    ) -> None:
        with pytest.raises(EquationParseError, match=message):
            parse_vector_pde_residual_expressions(
                [expression, "fxx[1] + fyy[1]"],
                2,
                ["x", "y"],
            )

    def test_expression_count_must_match_system_length(self) -> None:
        with pytest.raises(EquationParseError, match="exactly 2.*got 1"):
            parse_vector_pde_residual_expressions(
                ["fxx[0] + fyy[0]"],
                2,
                ["x", "y"],
            )
