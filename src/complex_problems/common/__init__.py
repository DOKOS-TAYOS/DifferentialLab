"""Shared helpers for complex problem plugins."""

from __future__ import annotations

from complex_problems.common.background import run_solver_with_loading
from complex_problems.common.expression import compile_scalar_expression
from complex_problems.common.problem_doc_ui import add_how_to_config_section
from complex_problems.common.validation import (
    parse_float,
    parse_int,
    parse_positive_float,
    parse_positive_int,
)

__all__ = [
    "compile_scalar_expression",
    "run_solver_with_loading",
    "add_how_to_config_section",
    "parse_float",
    "parse_int",
    "parse_positive_float",
    "parse_positive_int",
]
