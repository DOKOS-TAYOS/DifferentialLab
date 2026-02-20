"""Tests for utils.exceptions."""

from __future__ import annotations

import pytest

from utils.exceptions import (
    DifferentialLabError,
    EquationParseError,
    SolverFailedError,
    ValidationError,
)


def test_differential_lab_error_is_exception() -> None:
    err = DifferentialLabError("msg")
    assert isinstance(err, Exception)
    assert str(err) == "msg"


def test_validation_error_inherits_base() -> None:
    err = ValidationError("invalid input")
    assert isinstance(err, DifferentialLabError)


def test_equation_parse_error_inherits_base() -> None:
    err = EquationParseError("syntax error")
    assert isinstance(err, DifferentialLabError)


def test_solver_failed_error_inherits_base() -> None:
    err = SolverFailedError("did not converge")
    assert isinstance(err, DifferentialLabError)


def test_catch_base_catches_all() -> None:
    for exc_cls in (ValidationError, EquationParseError, SolverFailedError):
        exc = exc_cls("test")
        with pytest.raises(DifferentialLabError):
            raise exc
