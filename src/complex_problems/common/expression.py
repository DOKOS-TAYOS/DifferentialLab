"""Safe expression helpers for complex-problem user inputs."""

from __future__ import annotations

import ast
from typing import Any

import numpy as np

from utils import build_eval_namespace, safe_eval, validate_expression_ast


class CompiledScalarExpression:
    """Safe expression callable with an optional NumPy-array evaluation path."""

    def __init__(self, code: Any, namespace: dict[str, Any], variables: tuple[str, ...]) -> None:
        self._code = code
        self._namespace = namespace
        self._variables = variables

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        """Evaluate the expression for scalar arguments."""
        if len(args) > len(self._variables):
            raise TypeError("Too many positional arguments")
        values = dict(kwargs)
        values.update(dict(zip(self._variables, args, strict=False)))
        local_ns = dict(self._namespace)
        for var_name in self._variables:
            if var_name in values:
                local_ns[var_name] = values[var_name]
        return float(safe_eval(self._code, local_ns))

    def evaluate_array(self, *args: Any) -> np.ndarray:
        """Evaluate the expression directly on NumPy arrays."""
        if len(args) != len(self._variables):
            raise TypeError("Expected one value per expression variable")
        local_ns = dict(self._namespace)
        local_ns.update(zip(self._variables, args, strict=True))
        return np.asarray(safe_eval(self._code, local_ns), dtype=float)


def compile_scalar_expression(
    expression: str,
    *,
    variables: tuple[str, ...],
    parameters: dict[str, float] | None = None,
) -> CompiledScalarExpression:
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

    validate_expression_ast(expr)
    code = compile(tree, "<expression>", "eval")
    ns = build_eval_namespace(parameters or {})

    evaluator = CompiledScalarExpression(code, ns, variables)

    # Quick smoke test so UI fails early.
    test_args = {name: 0.0 for name in variables}
    evaluator(**test_args)
    return evaluator
