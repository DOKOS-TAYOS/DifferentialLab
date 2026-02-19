"""Safe parsing and evaluation of user-written ODE expressions."""

from __future__ import annotations

import ast
from typing import Any, Callable

import numpy as np

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
    ast.Index,
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
    """Check that *expression* contains only allowed AST nodes.

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


def parse_expression(
    expression: str,
    order: int,
    parameters: dict[str, float] | None = None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Parse an ODE expression into a callable ``f(x, y) -> dy/dx``.

    The expression should give the *highest derivative* in terms of ``x``,
    ``y[0]`` (the function), ``y[1]`` (first derivative), etc.

    For a first-order ODE ``y' = expr``, *order* = 1 and ``y[0]`` is ``y``.
    For a second-order ODE ``y'' = expr``, *order* = 2, ``y[0]`` is ``y``
    and ``y[1]`` is ``y'``.

    Args:
        expression: Python expression string for the highest derivative.
        order: Order of the ODE (1, 2, …).
        parameters: Named parameter values (e.g. ``{"omega": 2.0}``).

    Returns:
        A callable ``f(x, y)`` that returns ``dy/dx`` as a 1-D array
        suitable for :func:`scipy.integrate.solve_ivp`.

    Raises:
        EquationParseError: If the expression is invalid.
    """
    _validate_ast(expression)
    params = dict(parameters) if parameters else {}
    logger.debug("Parsing expression (order=%d): %s, params=%s", order, expression, params)

    namespace: dict[str, Any] = {**_SAFE_MATH, **params}

    compiled = compile(expression, "<ode_expression>", "eval")

    def _test_eval() -> None:
        test_y = np.zeros(order)
        test_ns = {**namespace, "x": 0.0, "y": test_y}
        try:
            eval(compiled, {"__builtins__": {}}, test_ns)
        except Exception as exc:
            raise EquationParseError(
                f"Expression evaluation failed: {exc}"
            ) from exc

    _test_eval()

    def ode_func(x: float, y: np.ndarray) -> np.ndarray:
        local_ns = {**namespace, "x": x, "y": y}
        highest = eval(compiled, {"__builtins__": {}}, local_ns)
        dydt = np.empty(order)
        for i in range(order - 1):
            dydt[i] = y[i + 1]
        dydt[order - 1] = float(highest)
        return dydt

    return ode_func


def validate_expression(expression: str) -> list[str]:
    """Check an expression for obvious errors without evaluating.

    Args:
        expression: Python expression string.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []
    if not expression or not expression.strip():
        errors.append("Expression is empty")
        return errors
    try:
        _validate_ast(expression.strip())
    except EquationParseError as exc:
        errors.append(str(exc))
    return errors
