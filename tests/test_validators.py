"""Tests for solver.validators."""

from __future__ import annotations

import pytest

from solver import validators


class TestValidateAllInputs:
    def test_valid_inputs_return_empty_list(self) -> None:
        errors = validators.validate_all_inputs(
            expression="y[0]",
            order=1,
            x_min=0.0,
            x_max=10.0,
            y0=[1.0],
            num_points=100,
            method="RK45",
        )
        assert errors == []

    def test_empty_expression_reported(self) -> None:
        errors = validators.validate_all_inputs(
            expression="",
            order=1,
            x_min=0.0,
            x_max=10.0,
            y0=[1.0],
            num_points=100,
            method="RK45",
        )
        assert any("empty" in e.lower() for e in errors)

    def test_domain_x_min_ge_x_max(self) -> None:
        errors = validators.validate_all_inputs(
            expression="y[0]",
            order=1,
            x_min=10.0,
            x_max=0.0,
            y0=[1.0],
            num_points=100,
            method="RK45",
        )
        assert any("x_min" in e and "x_max" in e for e in errors)

    def test_domain_equal_bounds_invalid(self) -> None:
        errors = validators.validate_all_inputs(
            expression="y[0]",
            order=1,
            x_min=5.0,
            x_max=5.0,
            y0=[1.0],
            num_points=100,
            method="RK45",
        )
        assert any("x_min" in e or "less" in e for e in errors)

    def test_wrong_number_of_initial_conditions(self) -> None:
        errors = validators.validate_all_inputs(
            expression="y[0]",
            order=1,
            x_min=0.0,
            x_max=10.0,
            y0=[1.0, 2.0],  # order=1 expects 1 value
            num_points=100,
            method="RK45",
        )
        assert any("initial condition" in e.lower() for e in errors)

    def test_order2_requires_two_ics(self) -> None:
        errors = validators.validate_all_inputs(
            expression="-y[0]",
            order=2,
            x_min=0.0,
            x_max=10.0,
            y0=[1.0],  # need 2 for order 2
            num_points=100,
            method="RK45",
        )
        assert any("2" in e and "initial" in e.lower() for e in errors)

    def test_num_points_too_small(self) -> None:
        errors = validators.validate_all_inputs(
            expression="y[0]",
            order=1,
            x_min=0.0,
            x_max=10.0,
            y0=[1.0],
            num_points=5,
            method="RK45",
        )
        assert any("10" in e or "points" in e.lower() for e in errors)

    def test_num_points_too_large(self) -> None:
        errors = validators.validate_all_inputs(
            expression="y[0]",
            order=1,
            x_min=0.0,
            x_max=10.0,
            y0=[1.0],
            num_points=2_000_000,
            method="RK45",
        )
        assert any("1,000,000" in e or "exceed" in e.lower() for e in errors)

    def test_unknown_method(self) -> None:
        errors = validators.validate_all_inputs(
            expression="y[0]",
            order=1,
            x_min=0.0,
            x_max=10.0,
            y0=[1.0],
            num_points=100,
            method="INVALID_METHOD",
        )
        assert any("unknown" in e.lower() or "method" in e.lower() for e in errors)

    def test_valid_methods_accepted(self) -> None:
        for method in ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"):
            errors = validators.validate_all_inputs(
                expression="y[0]",
                order=1,
                x_min=0.0,
                x_max=10.0,
                y0=[1.0],
                num_points=100,
                method=method,
            )
            assert not any("method" in e.lower() for e in errors), method

    def test_invalid_parameter_nan_reported(self) -> None:
        errors = validators.validate_all_inputs(
            expression="k * y[0]",
            order=1,
            x_min=0.0,
            x_max=10.0,
            y0=[1.0],
            num_points=100,
            method="RK45",
            params={"k": float("nan")},
        )
        assert any("parameter" in e.lower() or "finite" in e.lower() for e in errors)

    def test_x0_list_outside_domain_reported(self) -> None:
        errors = validators.validate_all_inputs(
            expression="y[0]",
            order=1,
            x_min=0.0,
            x_max=10.0,
            y0=[1.0],
            num_points=100,
            method="RK45",
            x0_list=[15.0],  # outside [0, 10]
        )
        assert any("domain" in e.lower() or "0" in e for e in errors)

    def test_multiple_errors_accumulated(self) -> None:
        errors = validators.validate_all_inputs(
            expression="",
            order=1,
            x_min=10.0,
            x_max=0.0,
            y0=[],
            num_points=3,
            method="BAD",
        )
        assert len(errors) >= 2
