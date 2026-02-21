"""Safe parsing and evaluation of user-written scalar functions f(x)."""

from __future__ import annotations

import ast
from typing import Any, Callable

import numpy as np

from solver.equation_parser import normalize_unicode_escapes
from utils import EquationParseError, get_logger

logger = get_logger(__name__)

_SAFE_MATH: dict[str, Any] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "pi": np.pi,
    "e": np.e,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "floor": np.floor,
    "ceil": np.ceil,
    "sign": np.sign,
    "heaviside": np.heaviside,
}

_ALLOWED_NODE_TYPES = (
    ast.Module,
    ast.Expr,
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Subscript,
    ast.Attribute,
    ast.FloorDiv,
    ast.Mod,
    ast.Compare,
    ast.IfExp,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Tuple,
    ast.List,
)


def _validate_ast(expression: str) -> None:
    """Check that expression contains only allowed AST nodes.

    Args:
        expression: Python expression string.

    Raises:
        EquationParseError: If the expression contains disallowed constructs.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise EquationParseError(f"Syntax error in expression: {exc}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise EquationParseError(
                f"Disallowed construct in expression: {type(node).__name__}"
            )


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
    _validate_ast(expression)
    params = dict(parameters) if parameters else {}
    logger.debug("Parsing scalar function: %s, params=%s", expression, params)

    namespace: dict[str, Any] = {**_SAFE_MATH, **params}
    compiled = compile(expression, "<scalar_function>", "eval")

    def _test_eval() -> None:
        test_ns = {**namespace, "x": 0.0}
        try:
            eval(compiled, {"__builtins__": {}}, test_ns)
        except Exception as exc:
            raise EquationParseError(
                f"Expression evaluation failed: {exc}"
            ) from exc

    _test_eval()

    def scalar_func(x: np.ndarray) -> np.ndarray:
        return np.array(
            [float(eval(compiled, {"__builtins__": {}}, {**namespace, "x": float(xi)}))
            for xi in np.atleast_1d(x)
        ],
        dtype=float,
    )

    return scalar_func
