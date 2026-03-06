"""Pipe flow complex problem plugin."""

from __future__ import annotations

from complex_problems.pipe_flow.problem import PROBLEM
from complex_problems.pipe_flow.solver import solve_pipe_flow

__all__ = [
    "PROBLEM",
    "solve_pipe_flow",
]
