"""Safe parsing and evaluation of user-written scalar functions f(x)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from utils import (
    EquationParseError,
    build_eval_namespace,
    get_logger,
    normalize_params,
    normalize_unicode_escapes,
    safe_eval,
    validate_expression_ast,
)

logger = get_logger(__name__)


def parse_scalar_function(
    expression: str,
    parameters: dict[str, float] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Parse a scalar function expression f(x) into a vectorized callable.

    The expression should use ``x`` as the independent variable.
    Supports the same math functions as the ODE parser.

    Args:
        expression: Python expression string (e.g. ``"sin(x)"``, ``"exp(-a*x)"``).
        parameters: Named parameter values (e.g. ``{"a": 1.0}``).

    Returns:
        A vectorized callable ``f(x)`` that accepts a numpy array and returns
        the evaluated values.

    Raises:
        EquationParseError: If the expression is invalid.
    """
    expression = normalize_unicode_escapes(expression.strip())
    validate_expression_ast(expression, "scalar function")
    params = normalize_params(parameters)
    logger.debug("Parsing scalar function: %s, params=%s", expression, params)

    namespace = build_eval_namespace(params)
    compiled = compile(expression, "<scalar_function>", "eval")

    # Test evaluation at x=0
    try:
        safe_eval(compiled, {**namespace, "x": 0.0})
    except Exception as exc:
        raise EquationParseError(f"Expression evaluation failed: {exc}") from exc

    def scalar_func(x: np.ndarray) -> np.ndarray:
        """Evaluate the compiled expression over a vectorized array."""
        x_arr = np.asarray(x, dtype=float)
        ns = {**namespace, "x": x_arr}
        result = safe_eval(compiled, ns)
        return np.asarray(result, dtype=float)

    return scalar_func
