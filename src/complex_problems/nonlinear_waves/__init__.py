"""Nonlinear wave propagation complex problem (NLSE and KdV)."""

from __future__ import annotations

from complex_problems.nonlinear_waves.problem import PROBLEM
from complex_problems.nonlinear_waves.solver import solve_nonlinear_waves

__all__ = [
    "PROBLEM",
    "solve_nonlinear_waves",
]

