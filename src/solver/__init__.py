"""ODE, difference, and PDE solving engine.

Public symbols are loaded lazily so importing :mod:`solver` does not pull in
SciPy-heavy solver modules until a caller asks for them.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "solve_difference": ("solver.difference_solver", "solve_difference"),
    "get_difference_function": ("solver.equation_parser", "get_difference_function"),
    "get_ode_function": ("solver.equation_parser", "get_ode_function"),
    "get_vector_ode_function": ("solver.equation_parser", "get_vector_ode_function"),
    "parse_pde_rhs_expression": ("solver.equation_parser", "parse_pde_rhs_expression"),
    "compute_ode_residual_error": ("solver.error_metrics", "compute_ode_residual_error"),
    "FNotation": ("solver.notation", "FNotation"),
    "generate_derivative_labels": ("solver.notation", "generate_derivative_labels"),
    "ODESolution": ("solver.ode_solver", "ODESolution"),
    "solve_multipoint": ("solver.ode_solver", "solve_multipoint"),
    "solve_ode": ("solver.ode_solver", "solve_ode"),
    "solve_pde_2d": ("solver.pde_solver", "solve_pde_2d"),
    "is_multivariate": ("solver.predefined", "is_multivariate"),
    "load_predefined_equations": ("solver.predefined", "load_predefined_equations"),
    "compute_statistics": ("solver.statistics", "compute_statistics"),
    "compute_statistics_2d": ("solver.statistics", "compute_statistics_2d"),
    "validate_all_inputs": ("solver.validators", "validate_all_inputs"),
}

__all__ = [
    "solve_difference",
    "get_difference_function",
    "get_ode_function",
    "get_vector_ode_function",
    "parse_pde_rhs_expression",
    "compute_ode_residual_error",
    "FNotation",
    "generate_derivative_labels",
    "ODESolution",
    "solve_multipoint",
    "solve_ode",
    "solve_pde_2d",
    "is_multivariate",
    "load_predefined_equations",
    "compute_statistics",
    "compute_statistics_2d",
    "validate_all_inputs",
]

if TYPE_CHECKING:
    from solver.difference_solver import solve_difference
    from solver.equation_parser import (
        get_difference_function,
        get_ode_function,
        get_vector_ode_function,
        parse_pde_rhs_expression,
    )
    from solver.error_metrics import compute_ode_residual_error
    from solver.notation import FNotation, generate_derivative_labels
    from solver.ode_solver import ODESolution, solve_multipoint, solve_ode
    from solver.pde_solver import solve_pde_2d
    from solver.predefined import is_multivariate, load_predefined_equations
    from solver.statistics import compute_statistics, compute_statistics_2d
    from solver.validators import validate_all_inputs


def __getattr__(name: str) -> Any:
    """Load a public solver symbol on first access."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
