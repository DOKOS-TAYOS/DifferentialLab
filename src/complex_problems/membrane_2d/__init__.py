"""2D coupled-oscillator membrane complex problem."""

from __future__ import annotations

from complex_problems.membrane_2d.problem import PROBLEM
from complex_problems.membrane_2d.solver import solve_membrane_2d

__all__ = [
    "PROBLEM",
    "solve_membrane_2d",
]

