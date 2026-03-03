"""Tests for transforms.function_parser."""

from __future__ import annotations

import numpy as np
import pytest

from transforms.function_parser import parse_scalar_function
from utils import EquationParseError


def test_sin_x() -> None:
    """Parse sin(x) and evaluate."""
    func = parse_scalar_function("sin(x)")
    x = np.array([0.0, np.pi / 2, np.pi])
    y = func(x)
    np.testing.assert_allclose(y, [0.0, 1.0, 0.0], atol=1e-10)


def test_exp_with_parameter() -> None:
    """Parse exp(-a*x) with parameter a."""
    func = parse_scalar_function("exp(-a*x)", parameters={"a": 1.0})
    x = np.array([0.0, 1.0, 2.0])
    y = func(x)
    np.testing.assert_allclose(y, [1.0, np.exp(-1), np.exp(-2)])


def test_polynomial() -> None:
    """Parse x**2 + 2*x + 1."""
    func = parse_scalar_function("x**2 + 2*x + 1")
    x = np.array([0.0, 1.0, -1.0])
    y = func(x)
    np.testing.assert_allclose(y, [1.0, 4.0, 0.0])


def test_invalid_expression_raises() -> None:
    """Invalid expression should raise EquationParseError."""
    with pytest.raises(EquationParseError):
        parse_scalar_function("y[0] + ")  # syntax error

    with pytest.raises(EquationParseError):
        parse_scalar_function("open('file')")  # disallowed


def test_empty_parameters_allowed() -> None:
    """Parameters can be None or empty."""
    func = parse_scalar_function("x + 1", parameters=None)
    np.testing.assert_allclose(func(np.array([1.0])), [2.0])
