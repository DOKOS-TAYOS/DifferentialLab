"""Solver pipeline — orchestrates validation, solving, and statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


@dataclass
class SolverResult:
    """Data-only bundle produced by a solver run (no pre-generated plots)."""

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

    Raises:
        ValidationError: If inputs fail validation.
        EquationParseError: If the expression cannot be parsed or function not found.
        DifferentialLabError: If the solver fails.
    """
    vars_list = variables if variables else ["x"]
    is_pde = equation_type == "pde" or is_multivariate(vars_list)
    is_2d_pde = is_pde and len(vars_list) >= 2
    error_metrics: dict[str, float] = {}
    solver_quality: dict[str, Any] = {}
    is_vector = (
        vector_expressions is not None and len(vector_expressions) > 0
    ) or equation_type == "vector_ode"

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

    if is_2d_pde:
        # PDE path: 2D (or more) variables
        if y_min is None or y_max is None:
            logger.warning("PDE validation failed: y_min and y_max required")
            raise ValidationError("PDE requires y_min and y_max for 2D domain")
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
            # Residual = LHS_operator(f) - rhs = 0
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
            # Default: neg_laplacian
            return -fxx - fyy - rhs

        pde_sol = solve_pde_2d(
            residual,
            x_min,
            x_max,
            float(y_min),
            float(y_max),
            n_points,
            ny,
            parameters=parameters,
        )
        solution_x = pde_sol.grid[0]
        solution_y_grid = pde_sol.grid[1]
        solution_y = pde_sol.u  # shape (ny, nx)
        solution_success = pde_sol.success
        solution_message = pde_sol.message
        solution_n_eval = pde_sol.n_eval
    elif equation_type == "difference":
        recur_func = get_difference_function(
            expression=expression,
            function_name=function_name,
            order=order,
            parameters=parameters,
        )
        n_min = int(x_min)
        n_max = int(x_max)
        diff_sol = solve_difference(recur_func, n_min, n_max, y0, order)
        if not diff_sol.success:
            from utils import SolverFailedError

            logger.error("Difference equation solver failed: %s", diff_sol.message)
            raise SolverFailedError(diff_sol.message)
        solution_x = diff_sol.n
        solution_y = diff_sol.y
        solution_success = diff_sol.success
        solution_message = diff_sol.message
        solution_n_eval = 0
    elif is_vector:
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
        solution_x = solution.x
        solution_y = solution.y
        solution_success = solution.success
        solution_message = solution.message
        solution_n_eval = solution.n_eval
        error_metrics = compute_ode_residual_error(ode_func, solution_x, solution_y)
        solver_quality = _build_solver_quality(solution)
    else:
        ode_func = get_ode_function(
            expression=expression,
            function_name=function_name,
            order=order,
            parameters=parameters,
        )
        t_eval = np.linspace(x_min, x_max, n_points)
        use_multipoint = x0_list is not None and any(abs(xi - x_min) > 1e-12 for xi in x0_list)
        if use_multipoint:
            conditions = list(enumerate(zip(x0_list, y0)))  # type: ignore[arg-type]
            conditions_flat = [(k, xi, ai) for k, (xi, ai) in conditions]
            solution = solve_multipoint(
                ode_func,
                conditions=conditions_flat,
                order=order,
                x_min=x_min,
                x_max=x_max,
                method=method,
                t_eval=t_eval,
            )
        else:
            solution = solve_ode(ode_func, (x_min, x_max), y0, method=method, t_eval=t_eval)
        solution_x = solution.x
        solution_y = solution.y
        solution_success = solution.success
        solution_message = solution.message
        solution_n_eval = solution.n_eval
        error_metrics = compute_ode_residual_error(ode_func, solution_x, solution_y)
        solver_quality = _build_solver_quality(solution)

    # ── Compute highest derivative and augment y ──────────────────────
    # For ODE/vector_ode, evaluate the ODE function to get f^(n) and
    # interleave it into y so the notation can display it.
    display_order = order
    if not is_2d_pde and equation_type != "difference":
        try:
            y_2d = np.atleast_2d(solution_y)
            if y_2d.shape[1] != len(solution_x):
                y_2d = y_2d.T
            n_pts = len(solution_x)
            n_comp = vector_components if is_vector else 1

            # Evaluate ODE function at each time point
            dydt_all = np.column_stack(
                [ode_func(solution_x[j], y_2d[:, j]) for j in range(n_pts)]
            )  # shape (n_state, n_pts)

            # Build augmented y with order+1 per component
            new_order = order + 1
            new_rows = n_comp * new_order
            augmented = np.empty((new_rows, n_pts))
            for comp_i in range(n_comp):
                for k in range(order):
                    augmented[comp_i * new_order + k] = y_2d[comp_i * order + k]
                # Highest derivative: dydt[comp_i * order + order - 1]
                augmented[comp_i * new_order + order] = dydt_all[comp_i * order + order - 1]

            solution_y = augmented
            display_order = new_order
        except Exception:
            logger.debug("Could not compute highest derivative; using raw y", exc_info=True)

    if is_2d_pde:
        stats = compute_statistics_2d(solution_x, solution_y_grid, solution_y, selected_stats)
    else:
        stats = compute_statistics(solution_x, solution_y, selected_stats)

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
        "solver_success": solution_success,
        "solver_message": solution_message,
        "n_evaluations": solution_n_eval,
        "rtol": solver_quality.get("rtol"),
        "atol": solver_quality.get("atol"),
        "residual_max": error_metrics.get("residual_max"),
        "residual_mean": error_metrics.get("residual_mean"),
        "residual_rms": error_metrics.get("residual_rms"),
        "n_jacobian_evals": solver_quality.get("n_jacobian_evals"),
    }

    # Build notation descriptor for f-notation labels
    if equation_type == "difference":
        notation = FNotation(kind="difference", order=order)
    elif is_vector:
        # If component_orders provided, augment each with +1 for display
        display_comp_orders: tuple[int, ...] | None = None
        if component_orders:
            display_comp_orders = tuple(co + 1 for co in component_orders)
        notation = FNotation(
            kind="vector_ode", n_components=vector_components, order=display_order,
            component_orders=display_comp_orders or (),
        )
    elif is_pde:
        notation = FNotation(
            kind="pde", n_independent_vars=len(vars_list), order=order
        )
    else:
        notation = FNotation(kind="ode", order=display_order)

    logger.info("Pipeline complete for '%s'", equation_name)

    y_grid_result: np.ndarray | None = solution_y_grid if is_2d_pde else None
    return SolverResult(
        x=solution_x,
        y=solution_y,
        statistics=stats,
        metadata=metadata,
        equation_type=equation_type,
        y_grid=y_grid_result,
        is_vector=is_vector,
        vector_components=vector_components if is_vector else 1,
        vector_order=display_order,
        notation=notation,
    )
