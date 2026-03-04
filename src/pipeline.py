"""Solver pipeline — orchestrates validation, solving, and statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from config import get_env_from_schema
from solver import (
    FNotation,
    ODESolution,
    compute_ode_residual_error,
    compute_statistics,
    compute_statistics_2d,
    get_difference_function,
    get_ode_function,
    get_vector_ode_function,
    is_multivariate,
    parse_pde_rhs_expression,
    solve_difference,
    solve_multipoint,
    solve_ode,
    solve_pde_2d,
    validate_all_inputs,
)
from solver.predefined import EquationType
from utils import ValidationError, get_logger

logger = get_logger(__name__)


@dataclass
class _DispatchResult:
    """Intermediate result from a solver dispatch function."""

    x: np.ndarray
    y: np.ndarray
    success: bool
    message: str
    n_eval: int
    error_metrics: dict[str, float] = field(default_factory=dict)
    solver_quality: dict[str, Any] = field(default_factory=dict)
    y_grid: np.ndarray | None = None
    ode_func: Callable | None = None


def _build_solver_quality(solution: ODESolution) -> dict[str, Any]:
    """Build solver quality metadata from an ODE solution."""
    quality: dict[str, Any] = {
        "rtol": get_env_from_schema("SOLVER_RTOL"),
        "atol": get_env_from_schema("SOLVER_ATOL"),
    }
    if solution.raw is not None:
        njev = getattr(solution.raw, "njev", None)
        if njev is not None:
            quality["n_jacobian_evals"] = int(njev)
    return quality


def _build_bc_array(
    bc_expressions: list[str],
    variables: list[str],
    parameters: dict[str, float],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Build a (ny, nx) boundary values array from function expressions.

    ``bc_expressions`` order: [bottom(y=y_min), top(y=y_max),
    left(x=x_min), right(x=x_max)].

    Horizontal boundaries (bottom/top) are written first. Vertical
    boundaries (left/right) overwrite corner values where they overlap.
    """
    bc = np.zeros((ny, nx))

    # Bottom (row 0): f(x) along x at y = y_min
    if bc_expressions[0].strip() not in ("0", ""):
        func = parse_pde_rhs_expression(bc_expressions[0], [variables[0]], parameters)
        for i in range(nx):
            bc[0, i] = func(x_grid[i])

    # Top (row ny-1): f(x) along x at y = y_max
    if len(bc_expressions) > 1 and bc_expressions[1].strip() not in ("0", ""):
        func = parse_pde_rhs_expression(bc_expressions[1], [variables[0]], parameters)
        for i in range(nx):
            bc[ny - 1, i] = func(x_grid[i])

    # Left (col 0): f(y) along y at x = x_min
    if len(bc_expressions) > 2 and bc_expressions[2].strip() not in ("0", ""):
        func = parse_pde_rhs_expression(bc_expressions[2], [variables[1]], parameters)
        for j in range(ny):
            bc[j, 0] = func(y_grid[j])

    # Right (col nx-1): f(y) along y at x = x_max
    if len(bc_expressions) > 3 and bc_expressions[3].strip() not in ("0", ""):
        func = parse_pde_rhs_expression(bc_expressions[3], [variables[1]], parameters)
        for j in range(ny):
            bc[j, nx - 1] = func(y_grid[j])

    return bc


def _dispatch_2d_pde(
    *,
    expression: str | None,
    vars_list: list[str],
    parameters: dict[str, float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n_points: int,
    n_points_y: int | None,
    pde_operator: str,
    bc_expressions: list[str] | None,
) -> _DispatchResult:
    """Dispatch a 2D PDE solve."""
    ny = n_points_y if n_points_y is not None else n_points
    rhs_func = parse_pde_rhs_expression(expression or "0", vars_list, parameters)

    def residual(
        x: float,
        y: float,
        f: float,
        fx: float,
        fy: float,
        fxx: float,
        fxy: float,
        fyy: float,
        **kw: Any,
    ) -> float:
        rhs = rhs_func(x, y, **kw)
        if pde_operator == "neg_laplacian":
            return -fxx - fyy - rhs
        if pde_operator == "laplacian":
            return fxx + fyy - rhs
        if pde_operator == "fxx":
            return fxx - rhs
        if pde_operator == "fyy":
            return fyy - rhs
        if pde_operator == "fx":
            return fx - rhs
        if pde_operator == "fy":
            return fy - rhs
        if pde_operator == "fxy":
            return fxy - rhs
        return -fxx - fyy - rhs

    bc_values: np.ndarray | None = None
    if bc_expressions and any(e.strip() not in ("0", "") for e in bc_expressions):
        x_grid = np.linspace(x_min, x_max, n_points)
        y_grid_bc = np.linspace(y_min, y_max, ny)
        bc_values = _build_bc_array(
            bc_expressions,
            vars_list,
            parameters,
            x_grid,
            y_grid_bc,
            n_points,
            ny,
        )

    pde_sol = solve_pde_2d(
        residual,
        x_min,
        x_max,
        y_min,
        y_max,
        n_points,
        ny,
        bc_values=bc_values,
        parameters=parameters,
    )
    return _DispatchResult(
        x=pde_sol.grid[0],
        y=pde_sol.u,
        success=pde_sol.success,
        message=pde_sol.message,
        n_eval=pde_sol.n_eval,
        y_grid=pde_sol.grid[1],
    )


def _dispatch_difference(
    *,
    expression: str | None,
    function_name: str | None,
    order: int,
    parameters: dict[str, float],
    x_min: float,
    x_max: float,
    y0: list[float],
) -> _DispatchResult:
    """Dispatch a difference equation solve."""
    recur_func = get_difference_function(
        expression=expression,
        function_name=function_name,
        order=order,
        parameters=parameters,
    )
    diff_sol = solve_difference(recur_func, int(x_min), int(x_max), y0, order)
    if not diff_sol.success:
        from utils import SolverFailedError

        logger.error("Difference equation solver failed: %s", diff_sol.message)
        raise SolverFailedError(diff_sol.message)
    return _DispatchResult(
        x=diff_sol.n,
        y=diff_sol.y,
        success=diff_sol.success,
        message=diff_sol.message,
        n_eval=0,
    )


def _dispatch_vector_ode(
    *,
    vector_expressions: list[str] | None,
    function_name: str | None,
    order: int,
    vector_components: int,
    parameters: dict[str, float],
    x_min: float,
    x_max: float,
    y0: list[float],
    n_points: int,
    method: str,
) -> _DispatchResult:
    """Dispatch a vector ODE solve."""
    vec_exprs = vector_expressions if vector_expressions else None
    ode_func = get_vector_ode_function(
        vector_expressions=vec_exprs,
        function_name=function_name if not vec_exprs else None,
        order=order,
        vector_components=vector_components,
        parameters=parameters,
    )
    t_eval = np.linspace(x_min, x_max, n_points)
    solution = solve_ode(ode_func, (x_min, x_max), y0, method=method, t_eval=t_eval)
    return _DispatchResult(
        x=solution.x,
        y=solution.y,
        success=solution.success,
        message=solution.message,
        n_eval=solution.n_eval,
        error_metrics=compute_ode_residual_error(ode_func, solution.x, solution.y),
        solver_quality=_build_solver_quality(solution),
        ode_func=ode_func,
    )


def _dispatch_scalar_ode(
    *,
    expression: str | None,
    function_name: str | None,
    order: int,
    parameters: dict[str, float],
    x_min: float,
    x_max: float,
    y0: list[float],
    n_points: int,
    method: str,
    x0_list: list[float] | None,
) -> _DispatchResult:
    """Dispatch a scalar ODE solve (IVP or multipoint BVP)."""
    ode_func = get_ode_function(
        expression=expression,
        function_name=function_name,
        order=order,
        parameters=parameters,
    )
    t_eval = np.linspace(x_min, x_max, n_points)
    use_multipoint = x0_list is not None and any(abs(xi - x_min) > 1e-12 for xi in x0_list)
    if use_multipoint:
        conditions = [(k, xi, ai) for k, (xi, ai) in enumerate(zip(x0_list, y0))]  # type: ignore[arg-type]
        solution = solve_multipoint(
            ode_func,
            conditions=conditions,
            order=order,
            x_min=x_min,
            x_max=x_max,
            method=method,
            t_eval=t_eval,
        )
    else:
        solution = solve_ode(ode_func, (x_min, x_max), y0, method=method, t_eval=t_eval)
    return _DispatchResult(
        x=solution.x,
        y=solution.y,
        success=solution.success,
        message=solution.message,
        n_eval=solution.n_eval,
        error_metrics=compute_ode_residual_error(ode_func, solution.x, solution.y),
        solver_quality=_build_solver_quality(solution),
        ode_func=ode_func,
    )


@dataclass
class SolverResult:
    """Data-only bundle produced by a solver run (no pre-generated plots).

    Attributes:
        x: Independent variable values (1D) or x grid for 2D PDE.
        y: Solution array — shape ``(n_vars, n_points)`` or ``(ny, nx)`` for 2D.
        statistics: Computed statistics dict.
        metadata: Equation info, solver parameters, domain, etc.
        equation_type: ``"ode"``, ``"difference"``, ``"pde"``, or ``"vector_ode"``.
        y_grid: For 2D PDE, the y-axis grid. ``None`` otherwise.
        is_vector: Whether the equation is a vector ODE.
        vector_components: Number of components for vector ODE.
        vector_order: Display order (derivatives per component).
        notation: F-notation context for labels.
    """

    x: np.ndarray
    y: np.ndarray
    statistics: dict[str, Any]
    metadata: dict[str, Any]
    equation_type: str = "ode"
    y_grid: np.ndarray | None = None  # For 2D PDE: y-axis grid
    is_vector: bool = False
    vector_components: int = 1
    vector_order: int = 1
    notation: FNotation | None = None


def run_solver_pipeline(
    *,
    expression: str | None = None,
    function_name: str | None = None,
    order: int,
    parameters: dict[str, float],
    equation_name: str,
    x_min: float,
    x_max: float,
    y0: list[float],
    n_points: int,
    method: str,
    selected_stats: set[str],
    x0_list: list[float] | None = None,
    equation_type: EquationType = "ode",
    variables: list[str] | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    n_points_y: int | None = None,
    vector_expressions: list[str] | None = None,
    vector_components: int = 1,
    pde_operator: str = "neg_laplacian",
    component_orders: tuple[int, ...] | None = None,
    bc_expressions: list[str] | None = None,
) -> SolverResult:
    """Execute the full solve workflow and return data results.

    Stages: validate → resolve function → solve → statistics.
    Plot generation is deferred to the ResultDialog for interactive control.

    Args:
        expression: ODE expression string (optional).
        function_name: Name of function in config.equations (optional).
        order: ODE order.
        parameters: Named parameter values.
        equation_name: Display name for plots/metadata.
        x_min: Domain start.
        x_max: Domain end.
        y0: Initial condition values ``[f(x₀), f'(x₁), …]``.
        n_points: Number of evaluation points.
        method: Solver method name.
        selected_stats: Set of statistic keys to compute.
        x0_list: Per-derivative condition points ``[x₀, x₁, …]``.
            If ``None`` or all equal to ``x_min``, uses standard IVP.
        equation_type: ``"ode"``, ``"difference"``, ``"pde"``, or ``"vector_ode"``.
        variables: Independent variable names (e.g. ``["x"]`` or ``["x", "y"]``).
        y_min: For 2D PDE, domain y start.
        y_max: For 2D PDE, domain y end.
        n_points_y: For 2D PDE, number of y grid points.
        vector_expressions: For vector ODE, list of expressions per component.
        vector_components: Number of components for vector ODE.
        pde_operator: PDE operator type (e.g. ``"neg_laplacian"``).
        component_orders: For vector ODE, order per component (optional).

    Returns:
        A :class:`SolverResult` with solution data, statistics, and metadata.

    Raises:
        ValidationError: If inputs fail validation.
        EquationParseError: If the expression cannot be parsed or function not found.
        DifferentialLabError: If the solver fails.
    """
    vars_list = variables if variables else ["x"]
    is_pde = equation_type == "pde" or is_multivariate(vars_list)
    is_2d_pde = is_pde and len(vars_list) >= 2
    is_vector = (
        vector_expressions is not None and len(vector_expressions) > 0
    ) or equation_type == "vector_ode"

    # ── Validate ──────────────────────────────────────────────────────
    if not is_pde:
        errors = validate_all_inputs(
            expression=expression if not is_vector else None,
            function_name=function_name,
            order=order,
            x_min=x_min,
            x_max=x_max,
            y0=y0,
            num_points=n_points,
            method=method,
            params=parameters,
            x0_list=x0_list,
            equation_type=equation_type,
            vector_expressions=vector_expressions if is_vector else None,
            vector_components=vector_components if is_vector else 1,
        )
        if errors:
            msg = "\n".join(errors)
            logger.warning("Validation failed: %s", msg)
            raise ValidationError(msg)

    # ── Dispatch to equation-type-specific solver ─────────────────────
    if is_2d_pde:
        if y_min is None or y_max is None:
            logger.warning("PDE validation failed: y_min and y_max required")
            raise ValidationError("PDE requires y_min and y_max for 2D domain")
        dr = _dispatch_2d_pde(
            expression=expression,
            vars_list=vars_list,
            parameters=parameters,
            x_min=x_min,
            x_max=x_max,
            y_min=float(y_min),
            y_max=float(y_max),
            n_points=n_points,
            n_points_y=n_points_y,
            pde_operator=pde_operator,
            bc_expressions=bc_expressions,
        )
    elif equation_type == "difference":
        dr = _dispatch_difference(
            expression=expression,
            function_name=function_name,
            order=order,
            parameters=parameters,
            x_min=x_min,
            x_max=x_max,
            y0=y0,
        )
    elif is_vector:
        dr = _dispatch_vector_ode(
            vector_expressions=vector_expressions,
            function_name=function_name,
            order=order,
            vector_components=vector_components,
            parameters=parameters,
            x_min=x_min,
            x_max=x_max,
            y0=y0,
            n_points=n_points,
            method=method,
        )
    else:
        dr = _dispatch_scalar_ode(
            expression=expression,
            function_name=function_name,
            order=order,
            parameters=parameters,
            x_min=x_min,
            x_max=x_max,
            y0=y0,
            n_points=n_points,
            method=method,
            x0_list=x0_list,
        )

    solution_x = dr.x
    solution_y = dr.y

    # ── Compute highest derivative and augment y ──────────────────────
    display_order = order
    if not is_2d_pde and equation_type != "difference" and dr.ode_func is not None:
        try:
            y_2d = np.atleast_2d(solution_y)
            if y_2d.shape[1] != len(solution_x):
                y_2d = y_2d.T
            n_pts = len(solution_x)
            n_comp = vector_components if is_vector else 1

            dydt_all = np.column_stack(
                [dr.ode_func(solution_x[j], y_2d[:, j]) for j in range(n_pts)]
            )

            new_order = order + 1
            augmented = np.empty((n_comp * new_order, n_pts))
            for comp_i in range(n_comp):
                for k in range(order):
                    augmented[comp_i * new_order + k] = y_2d[comp_i * order + k]
                augmented[comp_i * new_order + order] = dydt_all[comp_i * order + order - 1]

            solution_y = augmented
            display_order = new_order
        except Exception:
            logger.debug("Could not compute highest derivative; using raw y", exc_info=True)

    # ── Statistics ────────────────────────────────────────────────────
    if is_2d_pde:
        stats = compute_statistics_2d(solution_x, dr.y_grid, solution_y, selected_stats)
    else:
        stats = compute_statistics(solution_x, solution_y, selected_stats)

    # ── Metadata ──────────────────────────────────────────────────────
    metadata: dict[str, Any] = {
        "equation_name": equation_name,
        "equation_type": equation_type,
        "expression": expression if expression else f"<function:{function_name}>",
        "order": order,
        "parameters": parameters,
        "domain": (
            [x_min, x_max, y_min, y_max]
            if (is_pde and y_min is not None and y_max is not None)
            else [x_min, x_max]
        ),
        "initial_conditions": y0,
        "ic_points": x0_list if x0_list is not None else [x_min] * order,
        "method": method if equation_type == "ode" else "fdm",
        "num_points": n_points,
        "solver_success": dr.success,
        "solver_message": dr.message,
        "n_evaluations": dr.n_eval,
        "rtol": dr.solver_quality.get("rtol"),
        "atol": dr.solver_quality.get("atol"),
        "residual_max": dr.error_metrics.get("residual_max"),
        "residual_mean": dr.error_metrics.get("residual_mean"),
        "residual_rms": dr.error_metrics.get("residual_rms"),
        "n_jacobian_evals": dr.solver_quality.get("n_jacobian_evals"),
        "variables": vars_list,
        "boundary_conditions": bc_expressions,
    }

    # ── Notation ──────────────────────────────────────────────────────
    if equation_type == "difference":
        notation = FNotation(kind="difference", order=order)
    elif is_vector:
        display_comp_orders: tuple[int, ...] | None = None
        if component_orders:
            display_comp_orders = tuple(co + 1 for co in component_orders)
        notation = FNotation(
            kind="vector_ode",
            n_components=vector_components,
            order=display_order,
            component_orders=display_comp_orders or (),
        )
    elif is_pde:
        notation = FNotation(kind="pde", n_independent_vars=len(vars_list), order=order)
    else:
        notation = FNotation(kind="ode", order=display_order)

    logger.info("Pipeline complete for '%s'", equation_name)

    return SolverResult(
        x=solution_x,
        y=solution_y,
        statistics=stats,
        metadata=metadata,
        equation_type=equation_type,
        y_grid=dr.y_grid if is_2d_pde else None,
        is_vector=is_vector,
        vector_components=vector_components if is_vector else 1,
        vector_order=display_order,
        notation=notation,
    )
