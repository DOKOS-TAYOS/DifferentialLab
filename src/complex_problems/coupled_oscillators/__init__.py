"""Coupled harmonic oscillators plugin package."""

from __future__ import annotations

from complex_problems.coupled_oscillators.model import build_ode_function
from complex_problems.coupled_oscillators.problem import PROBLEM
from complex_problems.coupled_oscillators.solver import solve_coupled_oscillators

__all__ = [
    "PROBLEM",
    "build_ode_function",
    "solve_coupled_oscillators",
]

