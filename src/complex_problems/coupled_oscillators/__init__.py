"""Coupled harmonic oscillators — one-dimensional chain with configurable parameters."""

from __future__ import annotations

from complex_problems.coupled_oscillators.model import build_ode_function
from complex_problems.coupled_oscillators.solver import solve_coupled_oscillators

__all__ = [
    "build_ode_function",
    "solve_coupled_oscillators",
]
