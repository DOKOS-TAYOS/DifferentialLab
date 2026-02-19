"""Input validation for DifferentialLab solver parameters."""

from __future__ import annotations

import math

from config import SOLVER_METHODS
from solver.equation_parser import validate_expression
from utils import get_logger

logger = get_logger(__name__)


def _validate_domain(x_min: float, x_max: float) -> list[str]:
    """Validate the integration domain.

    Args:
        x_min: Start of the domain.
        x_max: End of the domain.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []
    if x_min >= x_max:
        errors.append(f"x_min ({x_min}) must be less than x_max ({x_max})")
    if not all(map(_is_finite, (x_min, x_max))):
        errors.append("Domain bounds must be finite numbers")
    return errors


def _validate_initial_conditions(y0: list[float], expected_order: int) -> list[str]:
    """Validate the initial conditions vector.

    Args:
        y0: Initial conditions values.
        expected_order: The ODE order (determines how many ICs are needed).

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []
    if len(y0) != expected_order:
        errors.append(
            f"Expected {expected_order} initial condition(s) for a "
            f"{'1st' if expected_order == 1 else f'{expected_order}th'}-order ODE, "
            f"got {len(y0)}"
        )
    for i, val in enumerate(y0):
        if not _is_finite(val):
            errors.append(f"Initial condition y0[{i}] = {val} is not a finite number")
    return errors


def _validate_grid(num_points: int) -> list[str]:
    """Validate the number of evaluation points.

    Args:
        num_points: Requested grid size.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []
    if num_points < 10:
        errors.append("Number of points must be at least 10")
    if num_points > 1_000_000:
        errors.append("Number of points must not exceed 1,000,000")
    return errors


def _validate_method(method: str) -> list[str]:
    """Validate the solver method name.

    Args:
        method: Solver method name.

    Returns:
        List of error messages (empty if valid).
    """
    if method not in SOLVER_METHODS:
        return [f"Unknown method '{method}'. Choose from: {', '.join(SOLVER_METHODS)}"]
    return []


def _validate_ode_expression(expression: str) -> list[str]:
    """Validate a custom ODE expression string.

    Args:
        expression: Python-syntax ODE expression.

    Returns:
        List of error messages (empty if valid).
    """
    return validate_expression(expression)


def _validate_parameters(params: dict[str, float]) -> list[str]:
    """Validate parameter values.

    Args:
        params: Parameter name-value mapping.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []
    for name, value in params.items():
        if not _is_finite(value):
            errors.append(f"Parameter '{name}' = {value} is not a finite number")
    return errors


def validate_all_inputs(
    expression: str,
    order: int,
    x_min: float,
    x_max: float,
    y0: list[float],
    num_points: int,
    method: str,
    params: dict[str, float] | None = None,
) -> list[str]:
    """Run all validations and return accumulated errors.

    Args:
        expression: ODE expression.
        order: ODE order.
        x_min: Domain start.
        x_max: Domain end.
        y0: Initial conditions.
        num_points: Grid points.
        method: Solver method.
        params: Named parameters.

    Returns:
        List of all error messages (empty if everything is valid).
    """
    errors: list[str] = []
    errors.extend(_validate_ode_expression(expression))
    errors.extend(_validate_domain(x_min, x_max))
    errors.extend(_validate_initial_conditions(y0, order))
    errors.extend(_validate_grid(num_points))
    errors.extend(_validate_method(method))
    if params:
        errors.extend(_validate_parameters(params))
    if errors:
        logger.warning("Validation errors: %s", errors)
    return errors


def _is_finite(value: float) -> bool:
    """Return ``True`` if *value* is a finite number (not NaN, not ±inf)."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
