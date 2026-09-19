"""Newtonian gravitational N-body Advanced Problem plugin."""

from complex_problems.gravitational_n_body.problem import PROBLEM
from complex_problems.gravitational_n_body.solver import NBodyResult, solve_n_body

__all__ = ["NBodyResult", "PROBLEM", "solve_n_body"]
