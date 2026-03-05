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
from solver.pde_solver import BC_DIRICHLET, BC_NEUMANN
from solver.predefined import EquationType
from utils import ValidationError, build_eval_namespace, get_logger, safe_eval

logger = get_logger(__name__)


@dataclass
class _DispatchResult:
    """Intermediate result from a solver dispatch function.

    Attributes:
        x: Independent variable or grid values.
        y: Solution array.
        success: Whether the solver converged.
        message: Solver status message.
        n_eval: Number of function evaluations.
        error_metrics: Residual error metrics (ODE only).
        solver_quality: Solver parameters and metadata.
        y_grid: For 2D PDE, the y-axis grid. None otherwise.
        ode_func: ODE function used (for residual computation). None for PDE/difference.
    """

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
    """Build solver quality metadata from an ODE solution.

    Args:
        solution: ODE solution from :func:`solve_ode` or :func:`solve_multipoint`.

    Returns:
        Dict with rtol, atol, and optionally n_jacobian_evals.
    """
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

    Order of ``bc_expressions``: [bottom(y=y_min), top(y=y_max),
    left(x=x_min), right(x=x_max)]. Horizontal boundaries (bottom/top)
    are written first. Vertical boundaries (left/right) overwrite corner
    values where they overlap.

    Args:
        bc_expressions: List of 4 expressions for boundary values.
        variables: Independent variable names (e.g. ``["x", "y"]``).
        parameters: Named parameter values for expression evaluation.
        x_grid: 1D array of x values.
        y_grid: 1D array of y values.
        nx: Number of x grid points.
        ny: Number of y grid points.

    Returns:
        Boundary values array of shape (ny, nx).
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


def _build_mask(
    mask_expression: str | None,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray | None:
    """Evaluate a mask expression on the grid, or return None for rectangular.

    Args:
        mask_expression: Python expression using x, y (or X, Y for meshgrid).
        x_grid: 1D array of x values.
        y_grid: 1D array of y values.
        parameters: Named parameters for expression evaluation.

    Returns:
        Boolean array (ny, nx) where True = inside domain, or None if
        no mask (rectangular domain).
    """
    if not mask_expression or not mask_expression.strip():
        return None
    X, Y = np.meshgrid(x_grid, y_grid)
    ns = build_eval_namespace(parameters)
    ns.update({"x": X, "y": Y, "X": X, "Y": Y})
    compiled = compile(mask_expression.strip(), "<mask>", "eval")
    result = safe_eval(compiled, ns)
    return np.asarray(result, dtype=bool)


def _build_bc_type_array(
    bc_types: list[str] | None,
    nx: int,
    ny: int,
    mask: np.ndarray | None,
    contour_bc_type: str | None,
) -> np.ndarray | None:
    """Build a (ny, nx) BC type array from per-edge types or contour type.

    Order of bc_types: [bottom, top, left, right] — each "dirichlet" or
    "neumann". For custom contour domains, contour_bc_type is used for
    all boundary points.

    Args:
        bc_types: Per-edge BC types [bottom, top, left, right].
        nx: Number of x grid points.
        ny: Number of y grid points.
        mask: Boolean mask (ny, nx). None for rectangular.
        contour_bc_type: BC type for custom contour (all boundary).

    Returns:
        String array (ny, nx) with "dirichlet" or "neumann", or None.
    """
    if bc_types is None and contour_bc_type is None:
        return None

    arr = np.full((ny, nx), BC_DIRICHLET, dtype=object)

    if mask is not None and contour_bc_type:
        # Custom contour: uniform BC type on all boundary
        arr[:] = contour_bc_type
    elif bc_types:
        # Rectangular: per-edge types
        types = bc_types + [BC_DIRICHLET] * (4 - len(bc_types))
        arr[0, :] = types[0]  # bottom
        arr[ny - 1, :] = types[1]  # top
        arr[:, 0] = types[2]  # left
        arr[:, nx - 1] = types[3]  # right

    return arr


def _build_neumann_array(
    bc_types: list[str] | None,
    bc_expressions: list[str] | None,
    variables: list[str],
    parameters: dict[str, float],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    nx: int,
    ny: int,
    mask: np.ndarray | None,
    contour_bc_type: str | None,
    contour_bc_expression: str | None,
) -> np.ndarray | None:
    """Build a (ny, nx) Neumann derivative values array.

    Only fills values where bc_type is "neumann". Returns None if no Neumann.

    Args:
        bc_types: Per-edge BC types.
        bc_expressions: Boundary value expressions.
        variables: Independent variable names.
        parameters: Parameter dict for expression evaluation.
        x_grid: 1D x grid.
        y_grid: 1D y grid.
        nx: Number of x points.
        ny: Number of y points.
        mask: Domain mask (ny, nx) or None.
        contour_bc_type: BC type for custom contour.
        contour_bc_expression: Expression for contour Neumann values.

    Returns:
        Neumann values array (ny, nx) or None if no Neumann BCs.
    """
    has_neumann = False
    if bc_types and any(t == BC_NEUMANN for t in bc_types):
        has_neumann = True
    if contour_bc_type == BC_NEUMANN:
        has_neumann = True
    if not has_neumann:
        return None

    arr = np.zeros((ny, nx))

    if mask is not None and contour_bc_type == BC_NEUMANN and contour_bc_expression:
        # Custom contour: evaluate expression on full grid
        func = parse_pde_rhs_expression(contour_bc_expression, variables, parameters)
        for j in range(ny):
            for i in range(nx):
                arr[j, i] = func(x_grid[i], y_grid[j])
    elif bc_types and bc_expressions:
        types = bc_types + [BC_DIRICHLET] * (4 - len(bc_types))
        # Bottom (row 0)
        if types[0] == BC_NEUMANN and len(bc_expressions) > 0:
            func = parse_pde_rhs_expression(bc_expressions[0], [variables[0]], parameters)
            for i in range(nx):
                arr[0, i] = func(x_grid[i])
        # Top (row ny-1)
        if types[1] == BC_NEUMANN and len(bc_expressions) > 1:
            func = parse_pde_rhs_expression(bc_expressions[1], [variables[0]], parameters)
            for i in range(nx):
                arr[ny - 1, i] = func(x_grid[i])
        # Left (col 0)
        if types[2] == BC_NEUMANN and len(bc_expressions) > 2:
            func = parse_pde_rhs_expression(bc_expressions[2], [variables[1]], parameters)
            for j in range(ny):
                arr[j, 0] = func(y_grid[j])
        # Right (col nx-1)
        if types[3] == BC_NEUMANN and len(bc_expressions) > 3:
            func = parse_pde_rhs_expression(bc_expressions[3], [variables[1]], parameters)
            for j in range(ny):
                arr[j, nx - 1] = func(y_grid[j])

    return arr


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
    bc_types: list[str] | None = None,
    mask_expression: str | None = None,
    contour_bc_expression: str | None = None,
    contour_bc_type: str | None = None,
) -> _DispatchResult:
    """Dispatch a 2D PDE solve.

    Returns:
        :class:`_DispatchResult` with grid, solution, and metadata.
    """
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
        rhs = rhs_func(
            x,
            y,
            f=f,
            fx=fx,
            fy=fy,
            fxx=fxx,
            fxy=fxy,
            fyy=fyy,
            **kw,
        )
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

    x_grid = np.linspace(x_min, x_max, n_points)
    y_grid_bc = np.linspace(y_min, y_max, ny)

    # Build mask
    mask = _build_mask(mask_expression, x_grid, y_grid_bc, parameters)

    # Build Dirichlet BC values
    bc_values: np.ndarray | None = None
    if mask is not None and contour_bc_type != BC_NEUMANN and contour_bc_expression:
        # Custom contour with Dirichlet BC: evaluate expression on grid
        bc_values = np.zeros((ny, n_points))
        func = parse_pde_rhs_expression(contour_bc_expression, vars_list, parameters)
        for j in range(ny):
            for i in range(n_points):
                bc_values[j, i] = func(x_grid[i], y_grid_bc[j])
    elif bc_expressions and any(e.strip() not in ("0", "") for e in bc_expressions):
        bc_values = _build_bc_array(
            bc_expressions,
            vars_list,
            parameters,
            x_grid,
            y_grid_bc,
            n_points,
            ny,
        )

    # Build BC type and Neumann arrays
    bc_type_arr = _build_bc_type_array(bc_types, n_points, ny, mask, contour_bc_type)
    neumann_arr = _build_neumann_array(
        bc_types,
        bc_expressions,
        vars_list,
        parameters,
        x_grid,
        y_grid_bc,
        n_points,
        ny,
        mask,
        contour_bc_type,
        contour_bc_expression,
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
        mask=mask,
        bc_type=bc_type_arr,
        bc_neumann_value=neumann_arr,
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
    """Dispatch a difference equation solve.

    Returns:
        :class:`_DispatchResult` with n, y, and metadata.
    """
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
    """Dispatch a vector ODE solve.

    Returns:
        :class:`_DispatchResult` with solution and error metrics.
    """
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
    """Dispatch a scalar ODE solve (IVP or multipoint BVP).

    Returns:
        :class:`_DispatchResult` with solution and error metrics.
    """
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
    bc_types: list[str] | None = None,
    mask_expression: str | None = None,
    contour_bc_expression: str | None = None,
    contour_bc_type: str | None = None,
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
            bc_types=bc_types,
            mask_expression=mask_expression,
            contour_bc_expression=contour_bc_expression,
            contour_bc_type=contour_bc_type,
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
