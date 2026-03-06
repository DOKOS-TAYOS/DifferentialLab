"""Safe expression helpers for complex-problem user inputs."""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any

from utils import build_eval_namespace, safe_eval, validate_expression_ast


def compile_scalar_expression(
    expression: str,
    *,
    variables: tuple[str, ...],
    parameters: dict[str, float] | None = None,
) -> Callable[..., float]:
    """Compile a scalar expression into a callable.

    Args:
        expression: User expression.
        variables: Allowed variable names expected at call-time.
        parameters: Constant parameter map available in namespace.

    Returns:
        Callable returning a float from the expression evaluation.
    """
    expr = expression.strip()
    if not expr:
        raise ValueError("Expression cannot be empty.")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    validate_expression_ast(tree)
    code = compile(tree, "<expression>", "eval")
    ns = build_eval_namespace(parameters or {})

    def _evaluate(**kwargs: Any) -> float:
        local_ns = dict(ns)
        for var_name in variables:
            if var_name in kwargs:
                local_ns[var_name] = kwargs[var_name]
        return float(safe_eval(code, local_ns))

    # Quick smoke test so UI fails early.
    test_args = {name: 0.0 for name in variables}
    _evaluate(**test_args)
    return _evaluate

