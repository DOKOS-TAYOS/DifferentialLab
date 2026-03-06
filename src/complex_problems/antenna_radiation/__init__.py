"""Antenna radiation complex problem plugin."""

from __future__ import annotations

from complex_problems.antenna_radiation.problem import PROBLEM
from complex_problems.antenna_radiation.solver import solve_antenna_radiation

__all__ = [
    "PROBLEM",
    "solve_antenna_radiation",
]
