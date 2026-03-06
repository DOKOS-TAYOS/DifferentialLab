"""Aerodynamics complex problem plugin."""

from __future__ import annotations

from complex_problems.aerodynamics_2d.problem import PROBLEM
from complex_problems.aerodynamics_2d.solver import solve_aerodynamics_2d

__all__ = [
    "PROBLEM",
    "solve_aerodynamics_2d",
]
