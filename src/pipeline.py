"""Solver pipeline — orchestrates validation, solving, plotting, and export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from config import generate_output_basename, get_csv_path, get_json_path, get_plot_path
from plotting import create_phase_plot, create_solution_plot, save_plot
from solver import compute_statistics, parse_expression, solve_ode, validate_all_inputs
from utils import ValidationError, export_all_results, get_logger

logger = get_logger(__name__)


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
    expression: str,
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
) -> SolverResult:
    """Execute the full solve workflow and return all results.

    Stages: validate → parse → solve → statistics → plot → export.

    Raises:
        ValidationError: If inputs fail validation.
        EquationParseError: If the expression cannot be parsed.
        DifferentialLabError: If the solver fails.
    """
    errors = validate_all_inputs(
        expression, order, x_min, x_max, y0, n_points, method, parameters,
    )
    if errors:
        raise ValidationError("\n".join(errors))

    ode_func = parse_expression(expression, order, parameters)

    t_eval = np.linspace(x_min, x_max, n_points)
    solution = solve_ode(ode_func, (x_min, x_max), y0, method=method, t_eval=t_eval)

    stats = compute_statistics(solution.x, solution.y, selected_stats)

    basename = generate_output_basename()
    csv_path = get_csv_path(basename)
    json_path = get_json_path(basename)
    plot_path = get_plot_path(basename)

    metadata: dict[str, Any] = {
        "equation_name": equation_name,
        "expression": expression,
        "order": order,
        "parameters": parameters,
        "domain": [x_min, x_max],
        "initial_conditions": y0,
        "method": method,
        "num_points": n_points,
        "solver_success": solution.success,
        "solver_message": solution.message,
        "n_evaluations": solution.n_eval,
    }

    fig = create_solution_plot(
        solution.x, solution.y,
        title=equation_name, xlabel="x", ylabel="y",
        selected_derivatives=selected_derivatives,
    )
    save_plot(fig, plot_path)

    phase_fig: Figure | None = None
    if order >= 2:
        phase_fig = create_phase_plot(
            solution.y, title=f"{equation_name} — Phase",
        )

    export_all_results(solution.x, solution.y, stats, metadata, csv_path, json_path)

    logger.info("Pipeline complete for '%s'", equation_name)

    return SolverResult(
        x=solution.x,
        y=solution.y,
        statistics=stats,
        metadata=metadata,
        fig=fig,
        phase_fig=phase_fig,
        csv_path=csv_path,
        json_path=json_path,
        plot_path=plot_path,
    )
