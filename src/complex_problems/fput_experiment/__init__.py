"""Dedicated Fermi--Pasta--Ulam--Tsingou experiment plugin."""

from complex_problems.fput_experiment.solver import FPUTResult, FPUTSweepResult, solve_fput

__all__ = ("FPUTResult", "FPUTSweepResult", "solve_fput")
