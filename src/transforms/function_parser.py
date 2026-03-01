"""Safe parsing and evaluation of user-written scalar functions f(x)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from utils import (
    SAFE_MATH,
    EquationParseError,
    get_logger,
    normalize_unicode_escapes,
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
    params = dict(parameters) if parameters else {}
    logger.debug("Parsing scalar function: %s, params=%s", expression, params)

    namespace: dict[str, Any] = {**SAFE_MATH, **params}
    compiled = compile(expression, "<scalar_function>", "eval")

    def _test_eval() -> None:
        test_ns = {**namespace, "x": 0.0}
        try:
            eval(compiled, {"__builtins__": {}}, test_ns)
        except Exception as exc:
            raise EquationParseError(f"Expression evaluation failed: {exc}") from exc

    _test_eval()

    def scalar_func(x: np.ndarray) -> np.ndarray:
        """Evaluate the compiled expression over a vectorized array."""
        x_arr = np.asarray(x, dtype=float)
        ns = {**namespace, "x": x_arr}
        result = eval(compiled, {"__builtins__": {}}, ns)
        return np.asarray(result, dtype=float)

    return scalar_func
