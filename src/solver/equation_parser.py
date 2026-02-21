"""Safe parsing and evaluation of user-written ODE expressions."""

from __future__ import annotations

import ast
import re
from typing import Any, Callable

import numpy as np

from utils import EquationParseError, get_logger

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
    expression = normalize_unicode_escapes(expression)
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


def get_ode_function(
    *,
    expression: str | None = None,
    function_name: str | None = None,
    order: int,
    parameters: dict[str, float] | None = None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Resolve an ODE function from either an expression string or a Python function.

    Exactly one of expression or function_name must be provided.

    Args:
        expression: Python expression for the highest derivative.
        function_name: Name of a function in config.equations to import.
        order: ODE order (1, 2, …).
        parameters: Named parameter values.

    Returns:
        A callable ``f(x, y)`` that returns ``dy/dx`` as a 1-D array.

    Raises:
        EquationParseError: If expression is invalid or function cannot be resolved.
        ValueError: If neither or both expression and function_name are provided.
    """
    params = dict(parameters) if parameters else {}
    if expression is not None and function_name is not None:
        raise ValueError("Provide either expression or function_name, not both")
    if expression is None and function_name is None:
        raise ValueError("Provide either expression or function_name")

    if expression is not None:
        return parse_expression(expression, order, params)

    # Import function from config.equations
    try:
        from config import equations as equations_module
    except ImportError as exc:
        raise EquationParseError(
            f"Cannot import config.equations: {exc}"
        ) from exc

    if not hasattr(equations_module, function_name):
        raise EquationParseError(
            f"Function '{function_name}' not found in config.equations"
        )

    func = getattr(equations_module, function_name)
    if not callable(func):
        raise EquationParseError(
            f"'{function_name}' in config.equations is not callable"
        )

    def ode_func(x: float, y: np.ndarray) -> np.ndarray:
        return func(x, y, **params)

    return ode_func


def parse_difference_expression(
    expression: str,
    order: int,
    parameters: dict[str, float] | None = None,
) -> Callable[[int, np.ndarray], float]:
    """Parse a difference equation expression into a callable ``f(n, y) -> y_next``.

    The expression gives y_{n+order} in terms of n and y[0], y[1], ..., y[order-1].
    Use n for the index, y[0] for y_n, y[1] for y_{n+1}, etc.

    Args:
        expression: Python expression string for the next value.
        order: Order of the recurrence (1, 2, …).
        parameters: Named parameter values.

    Returns:
        A callable ``f(n, y)`` that returns the next value (scalar).

    Raises:
        EquationParseError: If the expression is invalid.
    """
    expression = normalize_unicode_escapes(expression)
    _validate_ast(expression)
    params = dict(parameters) if parameters else {}
    logger.debug(
        "Parsing difference expression (order=%d): %s, params=%s",
        order, expression, params,
    )

    namespace: dict[str, Any] = {**_SAFE_MATH, **params}
    compiled = compile(expression, "<difference_expression>", "eval")

    def _test_eval() -> None:
        test_y = np.zeros(order)
        test_ns = {**namespace, "n": 0, "y": test_y}
        try:
            eval(compiled, {"__builtins__": {}}, test_ns)
        except Exception as exc:
            raise EquationParseError(
                f"Expression evaluation failed: {exc}"
            ) from exc

    _test_eval()

    def recur_func(n: int, y: np.ndarray) -> float:
        local_ns = {**namespace, "n": n, "y": y}
        return float(eval(compiled, {"__builtins__": {}}, local_ns))

    return recur_func


def get_difference_function(
    *,
    expression: str | None = None,
    function_name: str | None = None,
    order: int,
    parameters: dict[str, float] | None = None,
) -> Callable[[int, np.ndarray], float]:
    """Resolve a difference equation function from expression or Python function.

    Exactly one of expression or function_name must be provided.

    Args:
        expression: Python expression for y_{n+order}.
        function_name: Name of a function in config.difference_equations to import.
        order: Recurrence order (1, 2, …).
        parameters: Named parameter values.

    Returns:
        A callable ``f(n, y)`` that returns the next value (scalar).

    Raises:
        EquationParseError: If expression is invalid or function cannot be resolved.
        ValueError: If neither or both expression and function_name are provided.
    """
    params = dict(parameters) if parameters else {}
    if expression is not None and function_name is not None:
        raise ValueError("Provide either expression or function_name, not both")
    if expression is None and function_name is None:
        raise ValueError("Provide either expression or function_name")

    if expression is not None:
        return parse_difference_expression(expression, order, params)

    try:
        from config import difference_equations as diff_module
    except ImportError:
        try:
            from config import equations as diff_module
        except ImportError as exc:
            raise EquationParseError(
                f"Cannot import config.difference_equations or config.equations: {exc}"
            ) from exc

    if not hasattr(diff_module, function_name):
        raise EquationParseError(
            f"Function '{function_name}' not found in config"
        )

    func = getattr(diff_module, function_name)
    if not callable(func):
        raise EquationParseError(
            f"'{function_name}' is not callable"
        )

    def recur_func(n: int, y: np.ndarray) -> float:
        return float(func(n, y, **params))

    return recur_func


def parse_pde_rhs_expression(
    expression: str,
    variables: list[str],
    parameters: dict[str, float] | None = None,
) -> Callable[..., float]:
    """Parse a PDE RHS expression into a callable f(x, y, ...) -> float.

    The expression can use variable names (x, y, z, ...) and parameters.
    Used for the RHS of Poisson-type equations -u_xx - u_yy = f(x,y).

    Args:
        expression: Python expression string (e.g. ``"k"`` or ``"x * y"``).
        variables: List of variable names (e.g. ``["x", "y"]``).
        parameters: Named parameter values.

    Returns:
        A callable that takes (x, y, ...) and returns the RHS value.

    Raises:
        EquationParseError: If the expression is invalid.
    """
    expression = normalize_unicode_escapes(expression)
    _validate_ast(expression)
    params = dict(parameters) if parameters else {}
    logger.debug(
        "Parsing PDE RHS expression: %s, variables=%s, params=%s",
        expression, variables, params,
    )

    namespace: dict[str, Any] = {**_SAFE_MATH, **params}
    compiled = compile(expression, "<pde_rhs>", "eval")

    def rhs_func(*args: float, **kwargs: Any) -> float:
        local_ns = {**namespace, **kwargs}
        for i, var in enumerate(variables):
            if i < len(args):
                local_ns[var] = args[i]
        return float(eval(compiled, {"__builtins__": {}}, local_ns))

    # Test evaluation
    test_ns = {**namespace}
    for i, var in enumerate(variables):
        test_ns[var] = 0.0
    try:
        eval(compiled, {"__builtins__": {}}, test_ns)
    except Exception as exc:
        raise EquationParseError(f"PDE expression evaluation failed: {exc}") from exc

    return rhs_func


def parse_vector_expression(
    expressions: list[str],
    order: int,
    parameters: dict[str, float] | None = None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Parse a list of ODE expressions into a vector ODE callable.

    For [f_0(x), f_1(x), ...] with f_i'' = h_i(x, f_0, f_1, ..., f_0', f_1', ...):
    - State y = [f_0, f_1, ..., f_0', f_1', ...] (size = n_components * order)
    - y[0..n-1] = f_0, f_1, ... (the functions)
    - y[n..2n-1] = f_0', f_1', ... (first derivatives, if order >= 2)
    - etc.

    Each expression i gives the highest derivative h_i for component i.
    In expression i, use y[j] for f_j and y[n+j] for f_j' (when order=2).

    Args:
        expressions: List of Python expressions, one per component.
        order: Order of each ODE (1, 2, …).
        parameters: Named parameter values.

    Returns:
        A callable f(x, y) that returns dy/dx as a 1-D array.
    """
    n_components = len(expressions)
    if n_components == 0:
        raise EquationParseError("vector_expressions must have at least one expression")

    params = dict(parameters) if parameters else {}
    namespace: dict[str, Any] = {**_SAFE_MATH, **params}

    compiled_list: list[Any] = []
    for i, expr in enumerate(expressions):
        expr = normalize_unicode_escapes(expr)
        _validate_ast(expr)
        compiled_list.append(compile(expr, f"<vector_ode_{i}>", "eval"))

    state_size = n_components * order

    def _test_eval() -> None:
        test_y = np.zeros(state_size)
        test_ns = {**namespace, "x": 0.0, "y": test_y}
        for i, compiled in enumerate(compiled_list):
            try:
                eval(compiled, {"__builtins__": {}}, test_ns)
            except Exception as exc:
                raise EquationParseError(
                    f"Expression {i} evaluation failed: {exc}"
                ) from exc

    _test_eval()

    def ode_func(x: float, y: np.ndarray) -> np.ndarray:
        dydt = np.empty(state_size)
        local_ns = {**namespace, "x": x, "y": y}

        for i in range(n_components):
            for k in range(order - 1):
                dydt[i * order + k] = y[i * order + k + 1]
            highest = eval(compiled_list[i], {"__builtins__": {}}, local_ns)
            dydt[i * order + order - 1] = float(highest)

        return dydt

    return ode_func


def get_vector_ode_function(
    *,
    vector_expressions: list[str],
    function_name: str | None = None,
    order: int,
    vector_components: int,
    parameters: dict[str, float] | None = None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Resolve a vector ODE function from expressions or Python function.

    Exactly one of vector_expressions or function_name must be provided.

    Args:
        vector_expressions: List of expressions for each component's highest derivative.
        function_name: Name of function in config.equations (returns full dydt).
        order: Order of each ODE component.
        vector_components: Number of components (f_0, f_1, ...).
        parameters: Named parameter values.

    Returns:
        A callable f(x, y) that returns dy/dx.
    """
    params = dict(parameters) if parameters else {}
    if vector_expressions and function_name:
        raise ValueError("Provide either vector_expressions or function_name, not both")
    if not vector_expressions and not function_name:
        raise ValueError("Provide either vector_expressions or function_name")

    if vector_expressions:
        if len(vector_expressions) != vector_components:
            raise EquationParseError(
                f"vector_expressions length ({len(vector_expressions)}) "
                f"must match vector_components ({vector_components})"
            )
        return parse_vector_expression(vector_expressions, order, params)

    try:
        from config import equations as equations_module
    except ImportError as exc:
        raise EquationParseError(
            f"Cannot import config.equations: {exc}"
        ) from exc

    if not hasattr(equations_module, function_name):
        raise EquationParseError(
            f"Function '{function_name}' not found in config.equations"
        )

    func = getattr(equations_module, function_name)
    if not callable(func):
        raise EquationParseError(
            f"'{function_name}' in config.equations is not callable"
        )

    def ode_func(x: float, y: np.ndarray) -> np.ndarray:
        return func(x, y, **params)

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
        _validate_ast(normalize_unicode_escapes(expression.strip()))
    except EquationParseError as exc:
        errors.append(str(exc))
    return errors
