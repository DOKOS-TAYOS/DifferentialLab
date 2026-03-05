"""Safe parsing and evaluation of user-written ODE expressions.

Expressions may use either the legacy ``y[k]`` notation or the unified
``f[...]`` notation.  When ``f`` tokens are present they are automatically
rewritten to ``y[...]`` via :mod:`solver.notation` before compilation.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np

from solver.notation import FNotation, rewrite_f_expression
from utils import (
    EquationParseError,
    build_eval_namespace,
    get_logger,
    normalize_params,
    normalize_unicode_escapes,
    safe_eval,
    validate_exclusive_args,
    validate_expression_ast,
)

logger = get_logger(__name__)


def _maybe_rewrite(expression: str, notation: FNotation | None) -> str:
    """Rewrite f-notation to y-notation if a notation context is provided.

    Args:
        expression: Expression string with f-notation.
        notation: Notation context, or None to skip rewriting.

    Returns:
        Expression with f rewritten to y if notation given, else unchanged.
    """
    if notation is not None:
        return rewrite_f_expression(expression, notation)
    return expression


def _compile_and_test(
    expression: str,
    namespace: dict[str, Any],
    var_names: str | tuple[str, ...] = ("x", "y"),
    test_values: dict[str, Any] | None = None,
) -> Any:
    """Compile an expression and test it for evaluation errors.

    Args:
        expression: Python expression string.
        namespace: Namespace dict (typically {**SAFE_MATH, **params}).
        var_names: Variable names to include in test eval (single string or tuple).
        test_values: Override test values for variables (e.g., {"x": 0.0}).

    Returns:
        Compiled code object.

    Raises:
        EquationParseError: If compilation or test evaluation fails.
    """
    compiled = compile(expression, "<expression>", "eval")

    # Build test namespace
    test_ns = {**namespace}
    if isinstance(var_names, str):
        var_names = (var_names,)
    for var_name in var_names:
        if test_values and var_name in test_values:
            test_ns[var_name] = test_values[var_name]
        elif var_name == "x":
            test_ns[var_name] = 0.0
        elif var_name == "n":
            test_ns[var_name] = 0
        elif var_name == "y":
            test_ns[var_name] = np.zeros(test_values.get("y_size", 1) if test_values else 1)

    try:
        safe_eval(compiled, test_ns)
    except Exception as exc:
        raise EquationParseError(f"Expression evaluation failed: {exc}") from exc

    return compiled


def _load_config_function(function_name: str, module_name: str = "config.equations") -> Callable:
    """Load a callable function from a config module.

    Args:
        function_name: Name of the function to load.
        module_name: Full module path (default: "config.equations").

    Returns:
        The callable function.

    Raises:
        EquationParseError: If the module cannot be imported or function not found.
    """
    try:
        import importlib

        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise EquationParseError(f"Cannot import {module_name}: {exc}") from exc

    if not hasattr(module, function_name):
        raise EquationParseError(f"Function '{function_name}' not found in {module_name}")

    func = getattr(module, function_name)
    if not callable(func):
        raise EquationParseError(f"'{function_name}' in {module_name} is not callable")

    return func


def _parse_expression(
    expression: str,
    order: int,
    parameters: dict[str, float] | None = None,
    notation: FNotation | None = None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Parse an ODE expression into a callable ``f(x, y) -> dy/dx``.

    The expression may use ``f[k]`` notation (rewritten automatically)
    or legacy ``y[k]`` notation.

    Args:
        expression: Python expression string for the highest derivative.
        order: Order of the ODE (1, 2, …).
        parameters: Named parameter values (e.g. ``{"omega": 2.0}``).
        notation: Notation context for ``f[...]`` rewriting. If ``None``,
            a default scalar ODE notation is created automatically.

    Returns:
        A callable ``f(x, y)`` that returns ``dy/dx`` as a 1-D array
        suitable for :func:`scipy.integrate.solve_ivp`.

    Raises:
        EquationParseError: If the expression is invalid.
    """
    expression = normalize_unicode_escapes(expression)
    if notation is None:
        notation = FNotation(kind="ode", n_components=1, order=order)
    expression = _maybe_rewrite(expression, notation)
    validate_expression_ast(expression, "ODE expression")
    params = normalize_params(parameters)
    logger.debug("Parsing expression (order=%d): %s, params=%s", order, expression, params)

    namespace = build_eval_namespace(params)

    compiled = _compile_and_test(
        expression,
        namespace,
        var_names=("x", "y"),
        test_values={"y_size": order},
    )

    def ode_func(x: float, y: np.ndarray) -> np.ndarray:
        local_ns = {**namespace, "x": x, "y": y}
        highest = safe_eval(compiled, local_ns)
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
    params = normalize_params(parameters)
    validate_exclusive_args(expression, function_name, "expression", "function_name")

    if expression is not None:
        return _parse_expression(expression, order, params)

    assert function_name is not None  # Guaranteed by validation above
    func = _load_config_function(function_name, "config.equations")

    def ode_func(x: float, y: np.ndarray) -> np.ndarray:
        return func(x, y, **params)

    return ode_func


def _parse_difference_expression(
    expression: str,
    order: int,
    parameters: dict[str, float] | None = None,
    notation: FNotation | None = None,
) -> Callable[[int, np.ndarray], float]:
    """Parse a difference equation expression into a callable ``f(n, y) -> y_next``.

    The expression may use ``f[k]`` notation (rewritten automatically)
    or legacy ``y[k]`` notation.

    Args:
        expression: Python expression string for the next value.
        order: Order of the recurrence (1, 2, …).
        parameters: Named parameter values.
        notation: Notation context for ``f[...]`` rewriting.

    Returns:
        A callable ``f(n, y)`` that returns the next value (scalar).

    Raises:
        EquationParseError: If the expression is invalid.
    """
    expression = normalize_unicode_escapes(expression)
    if notation is None:
        notation = FNotation(kind="difference", n_components=1, order=order)
    expression = _maybe_rewrite(expression, notation)
    validate_expression_ast(expression, "difference expression")
    params = normalize_params(parameters)
    logger.debug(
        "Parsing difference expression (order=%d): %s, params=%s",
        order,
        expression,
        params,
    )

    namespace = build_eval_namespace(params)
    compiled = _compile_and_test(
        expression,
        namespace,
        var_names=("n", "y"),
        test_values={"y_size": order},
    )

    def recur_func(n: int, y: np.ndarray) -> float:
        local_ns = {**namespace, "n": n, "y": y}
        return float(safe_eval(compiled, local_ns))

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
    params = normalize_params(parameters)
    validate_exclusive_args(expression, function_name, "expression", "function_name")

    if expression is not None:
        return _parse_difference_expression(expression, order, params)

    assert function_name is not None  # Guaranteed by validation above
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
        raise EquationParseError(f"Function '{function_name}' not found in config")

    func = getattr(diff_module, function_name)
    if not callable(func):
        raise EquationParseError(f"'{function_name}' is not callable")

    def recur_func(n: int, y: np.ndarray) -> float:
        return float(func(n, y, **params))

    return recur_func


_INDEXED_VAR_NAMES = ["x", "y", "z", "w"]

_INDEXED_VAR_RE = re.compile(r"\bx\[([0-3])\]")

# PDE RHS notation: f[k] = f_{x[k]}, f[i,j] = f_{x[i],x[j]}
# x[0]=x, x[1]=y. So f[0]=fx, f[1]=fy, f[0,0]=fxx, f[0,1]=fxy, f[1,0]=fxy, f[1,1]=fyy
# Bare f (no brackets) = solution value
_PDE_F_SINGLE: dict[int, str] = {0: "fx", 1: "fy"}
_PDE_F_DOUBLE: dict[tuple[int, int], str] = {
    (0, 0): "fxx",
    (0, 1): "fxy",
    (1, 0): "fxy",
    (1, 1): "fyy",
}
_PDE_F_DOUBLE_RE = re.compile(r"\bf\[([0-1]),([0-1])\]")
_PDE_F_SINGLE_RE = re.compile(r"\bf\[([0-1])\]")


def _rewrite_pde_f_notation(expression: str) -> str:
    """Rewrite f[k], f[i,j] to fx, fy, fxx, fxy, fyy in PDE RHS context.

    Notation: f[k] = f_{x[k]}, f[i,j] = f_{x[i],x[j]}.
    Bare f (no brackets) = solution value.

    Args:
        expression: PDE RHS expression string.

    Returns:
        Expression with f-notation rewritten to derivative names.
    """

    # Replace f[i,j] first (longer pattern)
    def _replace_double(m: re.Match) -> str:
        i, j = int(m.group(1)), int(m.group(2))
        return _PDE_F_DOUBLE.get((i, j), m.group(0))

    def _replace_single(m: re.Match) -> str:
        idx = int(m.group(1))
        return _PDE_F_SINGLE.get(idx, m.group(0))

    expr = _PDE_F_DOUBLE_RE.sub(_replace_double, expression)
    return _PDE_F_SINGLE_RE.sub(_replace_single, expr)


def _rewrite_indexed_vars(expression: str) -> str:
    """Rewrite indexed variable notation ``x[0]``, ``x[1]``, ... to named variables.

    Maps ``x[0]`` -> ``x``, ``x[1]`` -> ``y``, ``x[2]`` -> ``z``, ``x[3]`` -> ``w``.
    This allows users to write PDE expressions using indexed notation while
    the internal solver still uses named variables.

    Args:
        expression: Expression string with indexed variables.

    Returns:
        Expression with indexed vars replaced by names.
    """
    return _INDEXED_VAR_RE.sub(
        lambda m: _INDEXED_VAR_NAMES[int(m.group(1))],
        expression,
    )


def parse_pde_rhs_expression(
    expression: str,
    variables: list[str],
    parameters: dict[str, float] | None = None,
) -> Callable[..., float]:
    """Parse a PDE RHS expression into a callable f(x, y, ...) -> float.

    The expression can use variable names (x, y, z, ...) or indexed notation
    (x[0], x[1], ...) and parameters.
    Used for the RHS of Poisson-type equations -u_xx - u_yy = f(x,y).

    Args:
        expression: Python expression string (e.g. ``"k"`` or ``"x[0] * x[1]"``).
        variables: List of variable names (e.g. ``["x", "y"]`` or ``["x[0]", "x[1]"]``).
        parameters: Named parameter values.

    Returns:
        A callable that takes (x, y, ...) and returns the RHS value.

    Raises:
        EquationParseError: If the expression is invalid.
    """
    expression = normalize_unicode_escapes(expression)
    expression = _rewrite_indexed_vars(expression)
    expression = _rewrite_pde_f_notation(expression)
    # Ensure internal variable names are plain (x, y, ...) for evaluation
    internal_vars = [
        _INDEXED_VAR_NAMES[i] if v.startswith("x[") else v
        for i, v in enumerate(variables)
        if i < len(_INDEXED_VAR_NAMES)
    ]
    if not internal_vars:
        internal_vars = list(variables)
    validate_expression_ast(expression, "PDE RHS")
    params = normalize_params(parameters)
    logger.debug(
        "Parsing PDE RHS expression: %s, variables=%s, internal_vars=%s, params=%s",
        expression,
        variables,
        internal_vars,
        params,
    )

    namespace = build_eval_namespace(params)
    pde_solution_vars = ("f", "fx", "fy", "fxx", "fxy", "fyy")
    test_values: dict[str, Any] = {var: 0.0 for var in internal_vars}
    test_values.update({v: 0.0 for v in pde_solution_vars})
    compiled = _compile_and_test(
        expression,
        namespace,
        var_names=tuple(internal_vars) + pde_solution_vars,
        test_values=test_values,
    )

    def rhs_func(*args: float, **kwargs: Any) -> float:
        local_ns = {**namespace, **kwargs}
        for i, var in enumerate(internal_vars):
            if i < len(args):
                local_ns[var] = args[i]
        return float(safe_eval(compiled, local_ns))

    return rhs_func


def _parse_vector_expression(
    expressions: list[str],
    order: int,
    parameters: dict[str, float] | None = None,
    notation: FNotation | None = None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Parse a list of ODE expressions into a vector ODE callable.

    Expressions may use ``f[i,k]`` notation (rewritten automatically)
    or legacy ``y[j]`` flat indexing.

    Args:
        expressions: List of Python expressions, one per component.
        order: Order of each ODE (1, 2, …).
        parameters: Named parameter values.
        notation: Notation context for ``f[...]`` rewriting.

    Returns:
        A callable f(x, y) that returns dy/dx as a 1-D array.
    """
    n_components = len(expressions)
    if n_components == 0:
        raise EquationParseError("vector_expressions must have at least one expression")

    if notation is None:
        notation = FNotation(kind="vector_ode", n_components=n_components, order=order)

    params = normalize_params(parameters)
    namespace = build_eval_namespace(params)

    compiled_list: list[Any] = []
    for i, expr in enumerate(expressions):
        expr = normalize_unicode_escapes(expr)
        expr = _maybe_rewrite(expr, notation)
        validate_expression_ast(expr, f"vector expression {i}")
        compiled_list.append(compile(expr, f"<vector_ode_{i}>", "eval"))

    state_size = n_components * order

    # Test each compiled expression
    test_y = np.zeros(state_size)
    test_ns = {**namespace, "x": 0.0, "y": test_y}
    for i, compiled in enumerate(compiled_list):
        try:
            safe_eval(compiled, test_ns)
        except Exception as exc:
            raise EquationParseError(f"Expression {i} evaluation failed: {exc}") from exc

    def ode_func(x: float, y: np.ndarray) -> np.ndarray:
        dydt = np.empty(state_size)
        local_ns = {**namespace, "x": x, "y": y}

        for i in range(n_components):
            for k in range(order - 1):
                dydt[i * order + k] = y[i * order + k + 1]
            highest = safe_eval(compiled_list[i], local_ns)
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

    Raises:
        ValueError: If both or neither of vector_expressions and function_name provided.
        EquationParseError: If expressions are invalid or function not found.
    """
    params = normalize_params(parameters)
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
        return _parse_vector_expression(vector_expressions, order, params)

    assert function_name is not None  # Guaranteed by validation above
    func = _load_config_function(function_name, "config.equations")

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
    from solver.notation import preprocess_prime_notation

    errors: list[str] = []
    if not expression or not expression.strip():
        errors.append("Expression is empty")
        return errors
    try:
        # Preprocess f'/f'' notation before AST validation so that
        # Python's parser doesn't confuse f' with an f-string literal.
        expr = preprocess_prime_notation(normalize_unicode_escapes(expression.strip()))
        validate_expression_ast(expr, "expression")
    except EquationParseError as exc:
        errors.append(str(exc))
    return errors
