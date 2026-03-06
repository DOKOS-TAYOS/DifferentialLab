"""Time-dependent Schrodinger complex problem (1D/2D)."""

from __future__ import annotations

from complex_problems.schrodinger_td.problem import PROBLEM
from complex_problems.schrodinger_td.solver import solve_schrodinger_td

__all__ = [
    "PROBLEM",
    "solve_schrodinger_td",
]

