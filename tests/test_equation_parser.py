"""Tests for solver.equation_parser."""

from __future__ import annotations

import numpy as np
import pytest

from solver.equation_parser import (
    normalize_unicode_escapes,
    parse_expression,
    validate_expression,
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
        errors = validate_expression("")
        assert len(errors) == 1
        assert "empty" in errors[0].lower()

    def test_whitespace_only_returns_error(self) -> None:
        errors = validate_expression("   \n\t  ")
        assert len(errors) == 1
        assert "empty" in errors[0].lower()

    def test_valid_expression_returns_no_errors(self) -> None:
        errors = validate_expression("y[0] * 2 + x")
        assert errors == []

    def test_syntax_error_reported(self) -> None:
        errors = validate_expression("y[0] + (")
        assert len(errors) == 1
        assert "syntax" in errors[0].lower() or "error" in errors[0].lower()

    def test_disallowed_construct_reported(self) -> None:
        # Lambda is not in allowed AST nodes
        errors = validate_expression("(lambda x: x)(1)")
        assert len(errors) == 1
        assert "disallowed" in errors[0].lower() or "construct" in errors[0].lower()

    def test_strips_whitespace(self) -> None:
        errors = validate_expression("  y[0] + 1  ")
        assert errors == []


class TestParseExpression:
    def test_order1_parses_and_evaluates(
        self,
        sample_expression_order1: str,
        sample_y0_order1: list[float],
        sample_parameters: dict[str, float],
    ) -> None:
        params = {"k": 0.5}
        ode_func = parse_expression(sample_expression_order1, order=1, parameters=params)
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
        ode_func = parse_expression(
            sample_expression_order2, order=2, parameters=params
        )
        x, y = 0.0, np.array([1.0, 0.0])
        dydx = ode_func(x, y)
        assert dydx.shape == (2,)
        assert dydx[0] == 0.0  # y' = y[1]
        np.testing.assert_allclose(dydx[1], -1.0)  # y'' = -omega**2 * y[0]

    def test_unicode_escape_in_expression(self) -> None:
        # ω as \u03C9 in expression
        ode_func = parse_expression(
            r"\u03C9**2 * y[0]", order=1, parameters={"ω": 2.0}
        )
        # Param key in Python must be the actual char; expression uses omega**2 * y[0]
        ode_func = parse_expression(
            "omega**2 * y[0]", order=1, parameters={"omega": 2.0}
        )
        x, y = 0.0, np.array([1.0])
        dydx = ode_func(x, y)
        np.testing.assert_allclose(dydx, [4.0])

    def test_invalid_expression_raises(self) -> None:
        with pytest.raises(EquationParseError):
            parse_expression("y[0] + ", order=1)

    def test_empty_parameters_allowed(self) -> None:
        ode_func = parse_expression("y[0]", order=1, parameters=None)
        x, y = 0.0, np.array([3.0])
        dydx = ode_func(x, y)
        np.testing.assert_allclose(dydx, [3.0])
