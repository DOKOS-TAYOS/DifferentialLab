"""Complex problems module — specialized cases with custom solvers and visualizations."""

from __future__ import annotations

from complex_problems.problem_registry import PROBLEM_REGISTRY

__all__ = [
    "ComplexProblemsDialog",
    "PROBLEM_REGISTRY",
]


def __getattr__(name: str):
    """Lazy-load ComplexProblemsDialog on first access."""
    if name == "ComplexProblemsDialog":
        from complex_problems.complex_problems_dialog import ComplexProblemsDialog

        return ComplexProblemsDialog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
