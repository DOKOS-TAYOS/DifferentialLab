"""Shared constants and validation for safe expression parsing.

Used by solver.equation_parser and transforms.function_parser to avoid
duplication of _SAFE_MATH, _ALLOWED_NODE_TYPES, and AST validation logic.
"""

from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np

from utils.exceptions import EquationParseError
from utils.logger import get_logger

logger = get_logger(__name__)

_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9A-Fa-f]{4})")


def normalize_unicode_escapes(text: str) -> str:
    """Replace ``\\uXXXX`` escape sequences with their Unicode characters.

    Allows users to enter expressions like ``\\u03C9**2 * y[0]`` and have
    them treated equivalently to ``ω**2 * y[0]``.

    Args:
        text: Input string that may contain Unicode escape sequences.

    Returns:
        String with all ``\\uXXXX`` sequences replaced by the corresponding
        Unicode character.
    """
    return _UNICODE_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), text)


SAFE_MATH: dict[str, Any] = {
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

_ALLOWED_NODE_TYPES: frozenset[type[ast.AST]] = frozenset(
    (
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
)


def normalize_params(parameters: dict[str, float] | None) -> dict[str, float]:
    """Return a mutable copy of *parameters*, or an empty dict if ``None``."""
    return dict(parameters) if parameters else {}


def build_eval_namespace(params: dict[str, float]) -> dict[str, Any]:
    """Build the safe evaluation namespace combining SAFE_MATH and user params."""
    return {**SAFE_MATH, **params}


def safe_eval(compiled: Any, namespace: dict[str, Any]) -> Any:
    """Evaluate a compiled expression in a sandboxed namespace."""
    return eval(compiled, {"__builtins__": {}}, namespace)  # noqa: S307


def validate_exclusive_args(
    arg_a: object | None,
    arg_b: object | None,
    name_a: str,
    name_b: str,
) -> None:
    """Ensure exactly one of two mutually exclusive arguments is provided.

    Raises:
        ValueError: If both or neither argument is provided.
    """
    if arg_a is not None and arg_b is not None:
        raise ValueError(f"Provide either {name_a} or {name_b}, not both")
    if arg_a is None and arg_b is None:
        raise ValueError(f"Provide either {name_a} or {name_b}")


def validate_expression_ast(expression: str, context: str = "expression") -> None:
    """Check that the expression contains only allowed AST nodes.

    Args:
        expression: Python expression string to validate.
        context: Short description for error messages (e.g. "ODE expression").

    Raises:
        EquationParseError: If the expression contains disallowed constructs.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        logger.debug("%s syntax error: %s — %s", context, expression[:80], exc)
        raise EquationParseError(f"Syntax error in expression: {exc}") from exc

    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODE_TYPES:
            raise EquationParseError(f"Disallowed construct in {context}: {type(node).__name__}")
