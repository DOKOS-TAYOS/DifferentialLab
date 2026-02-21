"""Solver pipeline — orchestrates validation, solving, plotting, and export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from matplotlib.figure import Figure

from config import generate_output_basename, get_csv_path, get_json_path, get_plot_path
from plotting import (
    create_contour_plot,
    create_phase_plot,
    create_solution_plot,
    create_surface_plot,
    save_plot,
)
from solver import (
    compute_statistics,
    compute_statistics_2d,
    get_difference_function,
    get_ode_function,
    is_multivariate,
    parse_pde_rhs_expression,
    solve_difference,
    solve_multipoint,
    solve_ode,
    solve_pde_2d,
    validate_all_inputs,
)
from utils import ValidationError, export_all_results, get_logger

logger = get_logger(__name__)

EquationType = Literal["ode", "difference", "pde"]


@dataclass
class SolverResult:
    """Bundles every artefact produced by a solver run."""

    x: np.ndarray
    y: np.ndarray
    statistics: dict[str, Any]
    metadata: dict[str, Any]
    fig: Figure
    phase_fig: Figure | None
    csv_path: Path
    json_path: Path
    plot_path: Path


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
    selected_derivatives: list[int] | None = None,
    x0_list: list[float] | None = None,
    equation_type: EquationType = "ode",
    variables: list[str] | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    n_points_y: int | None = None,
    plot_3d: bool = True,
) -> SolverResult:
    """Execute the full solve workflow and return all results.

    Stages: validate → resolve ODE function → solve → statistics → plot → export.

    Args:
        expression: ODE expression string (optional).
        function_name: Name of function in config.equations (optional).
        order: ODE order.
        parameters: Named parameter values.
        equation_name: Display name for plots/metadata.
        x_min: Domain start.
        x_max: Domain end.
        y0: Initial condition values ``[y(x_0), y'(x_1), …]``.
        n_points: Number of evaluation points.
        method: Solver method name.
        selected_stats: Set of statistic keys to compute.
        selected_derivatives: Indices of derivatives to plot.
        x0_list: Per-derivative condition points ``[x_0, x_1, …]``.
            If ``None`` or all equal to ``x_min``, uses standard IVP.

    Raises:
        ValidationError: If inputs fail validation.
        EquationParseError: If the expression cannot be parsed or function not found.
        DifferentialLabError: If the solver fails.
    """
    vars_list = variables if variables else ["x"]
    is_pde = equation_type == "pde" or is_multivariate(vars_list)

    if not is_pde:
        errors = validate_all_inputs(
            expression=expression,
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
        )
        if errors:
            raise ValidationError("\n".join(errors))

    if is_pde and len(vars_list) >= 2:
        # PDE path: 2D (or more) variables
        if y_min is None or y_max is None:
            raise ValidationError("PDE requires y_min and y_max for 2D domain")
        ny = n_points_y if n_points_y is not None else n_points
        rhs_func = parse_pde_rhs_expression(
            expression or "0", vars_list, parameters
        )

        def residual(x: float, y: float, f: float, fx: float, fy: float,
                     fxx: float, fxy: float, fyy: float, **kw: Any) -> float:
            return rhs_func(x, y, **kw)

        pde_sol = solve_pde_2d(
            residual,
            x_min, x_max, float(y_min), float(y_max),
            n_points, ny,
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
            raise SolverFailedError(diff_sol.message)
        solution_x = diff_sol.n
        solution_y = diff_sol.y
        solution_success = diff_sol.success
        solution_message = diff_sol.message
        solution_n_eval = 0
    else:
        ode_func = get_ode_function(
            expression=expression,
            function_name=function_name,
            order=order,
            parameters=parameters,
        )
        t_eval = np.linspace(x_min, x_max, n_points)
        use_multipoint = (
            x0_list is not None
            and any(abs(xi - x_min) > 1e-12 for xi in x0_list)
        )
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
            solution = solve_ode(
                ode_func, (x_min, x_max), y0, method=method, t_eval=t_eval
            )
        solution_x = solution.x
        solution_y = solution.y
        solution_success = solution.success
        solution_message = solution.message
        solution_n_eval = solution.n_eval

    if is_pde and len(vars_list) >= 2:
        stats = compute_statistics_2d(
            solution_x, solution_y_grid, solution_y, selected_stats
        )
    else:
        stats = compute_statistics(solution_x, solution_y, selected_stats)

    basename = generate_output_basename()
    csv_path = get_csv_path(basename)
    json_path = get_json_path(basename)
    plot_path = get_plot_path(basename)

    xlabel = "n" if equation_type == "difference" else "x"
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
    }

    if is_pde and len(vars_list) >= 2:
        if plot_3d:
            fig = create_surface_plot(
                solution_x, solution_y_grid, solution_y,
                title=equation_name, xlabel="x", ylabel="y", zlabel="u",
            )
        else:
            fig = create_contour_plot(
                solution_x, solution_y_grid, solution_y,
                title=equation_name, xlabel="x", ylabel="y",
            )
        phase_fig = None
        export_all_results(
            solution_x, solution_y, stats, metadata, csv_path, json_path,
            y_grid=solution_y_grid,
        )
    else:
        fig = create_solution_plot(
            solution_x, solution_y,
            title=equation_name, xlabel=xlabel, ylabel="y",
            selected_derivatives=selected_derivatives,
        )
        phase_fig = None
        if order >= 2 and not is_pde:
            phase_fig = create_phase_plot(
                solution_y, title=f"{equation_name} — Phase",
            )
        export_all_results(solution_x, solution_y, stats, metadata, csv_path, json_path)

    save_plot(fig, plot_path)

    logger.info("Pipeline complete for '%s'", equation_name)

    return SolverResult(
        x=solution_x,
        y=solution_y,
        statistics=stats,
        metadata=metadata,
        fig=fig,
        phase_fig=phase_fig,
        csv_path=csv_path,
        json_path=json_path,
        plot_path=plot_path,
    )
