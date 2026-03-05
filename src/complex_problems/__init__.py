"""Complex problems package with pluggable, problem-specific workflows."""

from __future__ import annotations

from complex_problems.base import ProblemDescriptor
from complex_problems.problem_registry import (
    PROBLEM_REGISTRY,
    get_problem_descriptors,
    open_problem_dialog,
)

__all__ = [
    "ComplexProblemsDialog",
    "ProblemDescriptor",
    "PROBLEM_REGISTRY",
    "get_problem_descriptors",
    "open_problem_dialog",
]


def __getattr__(name: str):
    """Lazy-load ComplexProblemsDialog on first access."""
    if name == "ComplexProblemsDialog":
        from complex_problems.complex_problems_dialog import ComplexProblemsDialog

        return ComplexProblemsDialog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

