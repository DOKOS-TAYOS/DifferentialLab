"""Linear strongly elliptic vector PDE systems on two-dimensional grids."""

from __future__ import annotations

import itertools

import numpy as np

from solver.pde_assembly import assemble_vector_pde, solve_sparse_vector_pde
from solver.pde_boundary import apply_boundary_values, prepare_vector_boundaries
from solver.pde_types import (
    BC_DIRICHLET,
    VectorPDEBoundaryConditions,
    VectorPDECoefficientProvider,
    VectorPDECoefficients,
    VectorPDEDiagnostics,
    VectorPDEResidual,
    VectorPDESolution,
)
from solver.pde_validation import (
    probe_vector_coefficients,
    validate_component_count,
    validate_grid_inputs,
    validate_vector_coefficients,
)
from utils import SolverFailedError, get_logger, normalize_params

logger = get_logger(__name__)


def solve_vector_pde_2d(
    residual_func: VectorPDEResidual | None,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
    *,
    components: int,
    parameters: dict[str, float] | None = None,
    mask: np.ndarray | None = None,
    coefficient_provider: VectorPDECoefficientProvider | None = None,
    boundary_conditions: VectorPDEBoundaryConditions | None = None,
) -> VectorPDESolution:
    """Solve a real linear strongly elliptic ``m``-component system in 2D.

    Exactly one coefficient path is required. ``residual_func`` receives
    ``(x, y, f, fx, fy, fxx, fxy, fyy, **parameters)`` and must return shape
    ``(m,)``; its complete matrix affinity is probed at every unknown point.
    Alternatively, ``coefficient_provider`` returns a
    :class:`VectorPDECoefficients` object directly.

    The public solution has shape ``(m, ny, nx)``. Sparse indexing is stable
    component-major: ``component * n_unknown_points + point_index``. All
    components share the same mask and periodic topology. Periodic axes use
    the scalar solver's non-duplicated upper endpoint and exact wrapping.

    Strong ellipticity is screened numerically using the symmetric principal
    symbol at 32 deterministic unit directions. This finite direction sample
    is validation evidence, not a general mathematical proof.
    """
    components = validate_component_count(components)
    if (residual_func is None) == (coefficient_provider is None):
        raise SolverFailedError(
            "Provide exactly one of residual_func or coefficient_provider for a vector PDE"
        )
    if boundary_conditions is not None and not isinstance(
        boundary_conditions, VectorPDEBoundaryConditions
    ):
        raise SolverFailedError(
            "boundary_conditions must be a VectorPDEBoundaryConditions object or None"
        )
    params = normalize_params(parameters)
    x_min, x_max, y_min, y_max, nx, ny = validate_grid_inputs(
        x_min,
        x_max,
        y_min,
        y_max,
        nx,
        ny,
    )
    periodic_x = boundary_conditions.periodic_x if boundary_conditions is not None else False
    periodic_y = boundary_conditions.periodic_y if boundary_conditions is not None else False
    x = np.linspace(x_min, x_max, nx, endpoint=not periodic_x)
    y = np.linspace(y_min, y_max, ny, endpoint=not periodic_y)
    hx = (x_max - x_min) / (nx if periodic_x else nx - 1)
    hy = (y_max - y_min) / (ny if periodic_y else ny - 1)
    boundaries = prepare_vector_boundaries(
        boundary_conditions,
        components=components,
        mask=mask,
        x=x,
        y=y,
    )
    reference_boundary = boundaries[0]

    def coefficient_at(xi: float, yj: float) -> tuple[VectorPDECoefficients, int]:
        """Evaluate and validate one system coefficient set."""
        try:
            if coefficient_provider is not None:
                raw = coefficient_provider(xi, yj, params)
            else:
                assert residual_func is not None
                raw = probe_vector_coefficients(residual_func, xi, yj, components, params)
            return validate_vector_coefficients(raw, xi, yj, components)
        except SolverFailedError:
            raise
        except Exception as exc:
            logger.error("Vector PDE coefficient probe failed at (%g, %g): %s", xi, yj, exc)
            raise SolverFailedError(
                f"Vector PDE coefficient probe failed at ({xi:.12g}, {yj:.12g}): {exc}"
            ) from exc

    n_unknown = int(np.count_nonzero(reference_boundary.unknown))
    u = np.full((components, ny, nx), np.nan, dtype=float)
    warnings = tuple(
        dict.fromkeys(itertools.chain.from_iterable(boundary.warnings for boundary in boundaries))
    )
    if n_unknown == 0:
        for boundary in boundaries:
            if np.any(boundary.boundary & (boundary.kind != BC_DIRICHLET)):
                raise SolverFailedError(
                    "Vector Neumann/Robin boundary values cannot be reconstructed because "
                    "the domain contains no unknown grid points"
                )
        for component, boundary in enumerate(boundaries):
            apply_boundary_values(u[component], boundary, hx=hx, hy=hy)
        diagnostics = VectorPDEDiagnostics(
            discrete_residual_l2=0.0,
            discrete_residual_linf=0.0,
            relative_residual_l2=0.0,
            component_residual_l2=(0.0,) * components,
            component_residual_linf=(0.0,) * components,
            component_relative_residual_l2=(0.0,) * components,
            matrix_shape=(0, 0),
            nnz=0,
            condition_estimate=None,
            warnings=warnings + ("No interior points were assembled.",),
        )
        return VectorPDESolution(
            grid=(x, y),
            u=u,
            success=True,
            message="No interior points",
            mask=reference_boundary.mask,
            diagnostics=diagnostics,
        )

    assembled = assemble_vector_pde(
        x,
        y,
        hx=hx,
        hy=hy,
        boundaries=boundaries,
        coefficient_at=coefficient_at,
    )
    solution, diagnostics = solve_sparse_vector_pde(
        assembled,
        components=components,
        points_per_component=n_unknown,
        warnings=warnings,
    )
    point_indices = reference_boundary.index_grid[reference_boundary.unknown]
    for component, boundary in enumerate(boundaries):
        component_solution = solution[component * n_unknown : (component + 1) * n_unknown]
        u[component, boundary.unknown] = component_solution[point_indices]
        apply_boundary_values(u[component], boundary, hx=hx, hy=hy)

    logger.info(
        "Vector PDE 2D solved: %d components, %dx%d grid, %d unknown points",
        components,
        nx,
        ny,
        n_unknown,
    )
    return VectorPDESolution(
        grid=(x, y),
        u=u,
        success=True,
        message="OK",
        mask=reference_boundary.mask,
        diagnostics=diagnostics,
    )


__all__ = [
    "VectorPDEBoundaryConditions",
    "VectorPDECoefficientProvider",
    "VectorPDECoefficients",
    "VectorPDEDiagnostics",
    "VectorPDEResidual",
    "VectorPDESolution",
    "solve_vector_pde_2d",
]
