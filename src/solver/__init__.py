"""ODE, difference, and PDE solving engine."""

from solver.difference_solver import solve_difference
from solver.equation_parser import (
    get_difference_function,
    get_ode_function,
    get_vector_ode_function,
    parse_pde_rhs_expression,
)
from solver.error_metrics import compute_ode_residual_error
from solver.ode_solver import ODESolution, solve_multipoint, solve_ode
from solver.pde_solver import solve_pde_2d
from solver.predefined import is_multivariate, load_predefined_equations
from solver.statistics import compute_statistics, compute_statistics_2d
from solver.validators import validate_all_inputs

__all__ = [
    "solve_difference",
    "get_difference_function",
    "get_ode_function",
    "get_vector_ode_function",
    "parse_pde_rhs_expression",
    "compute_ode_residual_error",
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
