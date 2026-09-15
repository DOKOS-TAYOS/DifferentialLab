"""Backwards-compatible public facade for scalar linear elliptic 2D PDEs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from solver.pde_assembly import assemble_scalar_pde, solve_sparse_pde
from solver.pde_boundary import apply_boundary_values, classify_mask, prepare_boundary
from solver.pde_types import (
    BC_DIRICHLET,
    BC_NEUMANN,
    BC_ROBIN,
    PDEBoundaryCondition,
    PDEBoundaryConditions,
    PDECoefficientProvider,
    PDECoefficients,
    PDEDiagnostics,
    PDESolution,
)
from solver.pde_validation import probe_coefficients, validate_coefficients, validate_grid_inputs
from utils import SolverFailedError, get_logger, normalize_params

logger = get_logger(__name__)


def _classify_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper for the established non-periodic mask classifier."""
    return classify_mask(mask)


def solve_pde_2d(
    residual_func: Callable[..., float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
    bc_values: np.ndarray | None = None,
    parameters: dict[str, float] | None = None,
    mask: np.ndarray | None = None,
    bc_type: np.ndarray | None = None,
    bc_neumann_value: np.ndarray | None = None,
    coefficient_provider: PDECoefficientProvider | None = None,
    boundary_conditions: PDEBoundaryConditions | None = None,
) -> PDESolution:
    """Solve a real scalar linear elliptic PDE on a two-dimensional grid.

    The established positional and keyword arguments remain supported. New
    callers may pass ``boundary_conditions`` for structured Dirichlet,
    Neumann, Robin, and axis-level periodic conditions. Structured and legacy
    boundary inputs cannot be mixed.

    Periodic axes use a non-duplicated upper endpoint. Arbitrary masks cannot
    be periodic because wrapped mask topology is not implemented. On masked
    contours, Neumann and Robin derivatives use a one-sided grid normal rather
    than an inferred geometric normal.
    """
    params = normalize_params(parameters)
    x_min, x_max, y_min, y_max, nx, ny = validate_grid_inputs(
        x_min,
        x_max,
        y_min,
        y_max,
        nx,
        ny,
    )
    if boundary_conditions is not None and not isinstance(
        boundary_conditions, PDEBoundaryConditions
    ):
        raise SolverFailedError("boundary_conditions must be a PDEBoundaryConditions object")
    periodic_x = boundary_conditions.periodic_x if boundary_conditions is not None else False
    periodic_y = boundary_conditions.periodic_y if boundary_conditions is not None else False
    x = np.linspace(x_min, x_max, nx, endpoint=not periodic_x)
    y = np.linspace(y_min, y_max, ny, endpoint=not periodic_y)
    hx = (x_max - x_min) / (nx if periodic_x else nx - 1)
    hy = (y_max - y_min) / (ny if periodic_y else ny - 1)
    boundary = prepare_boundary(
        nx=nx,
        ny=ny,
        x=x,
        y=y,
        mask=mask,
        bc_values=bc_values,
        bc_type=bc_type,
        bc_neumann_value=bc_neumann_value,
        boundary_conditions=boundary_conditions,
    )

    def coefficient_at(xi: float, yj: float) -> tuple[PDECoefficients, int]:
        """Evaluate one coefficient tuple with a coordinate-aware failure."""
        try:
            raw = (
                probe_coefficients(residual_func, xi, yj, params)
                if coefficient_provider is None
                else coefficient_provider(xi, yj, params)
            )
            return validate_coefficients(raw, xi, yj)
        except SolverFailedError:
            raise
        except Exception as exc:
            logger.error("PDE coefficient probe failed at (%g, %g): %s", xi, yj, exc)
            raise SolverFailedError(
                f"Coefficient probe failed at ({xi:.12g}, {yj:.12g}): {exc}"
            ) from exc

    n_unknown = int(np.count_nonzero(boundary.unknown))
    u = np.full((ny, nx), np.nan)
    if n_unknown == 0:
        if np.any(boundary.boundary & (boundary.kind != BC_DIRICHLET)):
            raise SolverFailedError(
                "Neumann/Robin boundary values cannot be reconstructed because the domain "
                "contains no unknown grid points"
            )
        apply_boundary_values(u, boundary, hx=hx, hy=hy)
        diagnostics = PDEDiagnostics(
            0.0,
            0.0,
            0.0,
            (0, 0),
            0,
            None,
            boundary.warnings + ("No interior points were assembled.",),
        )
        return PDESolution(
            (x, y),
            u,
            True,
            "No interior points",
            0,
            boundary.mask,
            diagnostics,
        )

    assembled = assemble_scalar_pde(
        x,
        y,
        hx=hx,
        hy=hy,
        boundary=boundary,
        coefficient_at=coefficient_at,
    )
    solution, diagnostics = solve_sparse_pde(assembled, warnings=boundary.warnings)
    u[boundary.unknown] = solution[boundary.index_grid[boundary.unknown]]
    apply_boundary_values(u, boundary, hx=hx, hy=hy)
    logger.info("PDE 2D solved: %dx%d grid, %d unknown points", nx, ny, n_unknown)
    return PDESolution((x, y), u, True, "OK", 0, boundary.mask, diagnostics)


__all__ = [
    "BC_DIRICHLET",
    "BC_NEUMANN",
    "BC_ROBIN",
    "PDEBoundaryCondition",
    "PDEBoundaryConditions",
    "PDECoefficientProvider",
    "PDECoefficients",
    "PDEDiagnostics",
    "PDESolution",
    "solve_pde_2d",
]
