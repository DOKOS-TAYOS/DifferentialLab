"""Sparse assembly and linear-solve helpers for scalar 2D PDEs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from solver.pde_boundary import PreparedBoundary, boundary_substitution, wrapped_neighbor
from solver.pde_types import (
    BC_DIRICHLET,
    PDECoefficients,
    PDEDiagnostics,
    VectorPDECoefficients,
    VectorPDEDiagnostics,
)
from utils import SolverFailedError

_LINEAR_RESIDUAL_TOLERANCE = 1.0e-8
_MAX_CONDITION_ESTIMATE_SIZE = 256


@dataclass(frozen=True)
class AssembledPDE:
    """One assembled sparse scalar PDE system."""

    matrix: Any
    rhs: np.ndarray


def assemble_scalar_pde(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hx: float,
    hy: float,
    boundary: PreparedBoundary,
    coefficient_at: Callable[[float, float], tuple[PDECoefficients, int]],
) -> AssembledPDE:
    """Assemble the finite-difference system, including periodic mixed wraps."""
    from scipy import sparse

    n_unknown = int(np.count_nonzero(boundary.unknown))
    max_entries = max(1, 9 * n_unknown)
    rows = np.empty(max_entries, dtype=np.int64)
    cols = np.empty(max_entries, dtype=np.int64)
    data = np.empty(max_entries, dtype=float)
    rhs = np.zeros(n_unknown)
    entry_count = 0
    component_orientations: dict[int, tuple[int, float, float]] = {}

    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)
    inv_2hx = 1.0 / (2.0 * hx)
    inv_2hy = 1.0 / (2.0 * hy)
    inv_4hxhy = 1.0 / (4.0 * hx * hy)

    def append_entry(row: int, col: int, value: float) -> None:
        nonlocal entry_count
        rows[entry_count] = row
        cols[entry_count] = col
        data[entry_count] = value
        entry_count += 1

    grid_width = boundary.unknown.shape[1]
    for row, flat_index in enumerate(np.flatnonzero(boundary.unknown)):
        j, i = divmod(int(flat_index), grid_width)
        xi, yj = float(x[i]), float(y[j])
        coefficients, orientation = coefficient_at(xi, yj)
        component = int(boundary.component_labels[j, i])
        reference = component_orientations.get(component)
        if reference is None:
            component_orientations[component] = (orientation, xi, yj)
        elif reference[0] != orientation:
            expected = "positive definite" if reference[0] > 0 else "negative definite"
            actual = "positive definite" if orientation > 0 else "negative definite"
            raise SolverFailedError(
                "PDE ellipticity orientation changes within connected domain component "
                f"{component}: expected {expected} from ({reference[1]:.12g}, "
                f"{reference[2]:.12g}), got {actual} at ({xi:.12g}, {yj:.12g})"
            )

        a_c, bxy, c_c, d_c, e_c, g_c, r0 = coefficients
        rhs[row] = -r0
        append_entry(row, row, -2.0 * a_c * inv_hx2 - 2.0 * c_c * inv_hy2 + g_c)

        def add_neighbor(
            raw_i: int,
            raw_j: int,
            coefficient: float,
            *,
            mixed_neighbor: bool = False,
        ) -> None:
            """Add an unknown, wrapped, Dirichlet, or substituted neighbor."""
            normalized = wrapped_neighbor(raw_i, raw_j, boundary)
            if normalized is None:
                return
            neighbor_i, neighbor_j = normalized
            neighbor_index = int(boundary.index_grid[neighbor_j, neighbor_i])
            if neighbor_index >= 0:
                append_entry(row, neighbor_index, coefficient)
                return
            if not boundary.boundary[neighbor_j, neighbor_i]:
                return
            if boundary.kind[neighbor_j, neighbor_i] == BC_DIRICHLET:
                rhs[row] -= coefficient * boundary.gamma[neighbor_j, neighbor_i]
                return
            inward_i, inward_j, factor, offset = boundary_substitution(
                boundary,
                boundary_i=neighbor_i,
                boundary_j=neighbor_j,
                source_i=i,
                source_j=j,
                hx=hx,
                hy=hy,
                mixed_neighbor=mixed_neighbor,
            )
            inward_index = int(boundary.index_grid[inward_j, inward_i])
            if inward_index < 0:
                raise SolverFailedError(
                    "Neumann/Robin boundary elimination did not resolve to an unknown point"
                )
            append_entry(row, inward_index, coefficient * factor)
            rhs[row] -= coefficient * offset

        add_neighbor(i - 1, j, a_c * inv_hx2 - d_c * inv_2hx)
        add_neighbor(i + 1, j, a_c * inv_hx2 + d_c * inv_2hx)
        add_neighbor(i, j - 1, c_c * inv_hy2 - e_c * inv_2hy)
        add_neighbor(i, j + 1, c_c * inv_hy2 + e_c * inv_2hy)
        if abs(bxy) > 1.0e-15:
            add_neighbor(i + 1, j + 1, bxy * inv_4hxhy, mixed_neighbor=True)
            add_neighbor(i - 1, j + 1, -bxy * inv_4hxhy, mixed_neighbor=True)
            add_neighbor(i + 1, j - 1, -bxy * inv_4hxhy, mixed_neighbor=True)
            add_neighbor(i - 1, j - 1, bxy * inv_4hxhy, mixed_neighbor=True)

    matrix = sparse.coo_matrix(
        (data[:entry_count], (rows[:entry_count], cols[:entry_count])),
        shape=(n_unknown, n_unknown),
    ).tocsr()
    if not np.all(np.isfinite(matrix.data)):
        raise SolverFailedError("Assembled sparse matrix contains non-finite values")
    if not np.all(np.isfinite(rhs)):
        raise SolverFailedError("Assembled right-hand side contains non-finite values")
    return AssembledPDE(matrix, rhs)


def assemble_vector_pde(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hx: float,
    hy: float,
    boundaries: tuple[PreparedBoundary, ...],
    coefficient_at: Callable[[float, float], tuple[VectorPDECoefficients, int]],
) -> AssembledPDE:
    """Assemble a component-major sparse block system for a vector 2D PDE.

    The COO workspace reserves the maximum of nine stencil contributions for
    every solution component in each system row.
    """
    from scipy import sparse

    if not boundaries:
        raise SolverFailedError("Vector PDE assembly requires at least one component boundary")
    reference_boundary = boundaries[0]
    for boundary in boundaries[1:]:
        if not (
            np.array_equal(boundary.mask, reference_boundary.mask)
            and np.array_equal(boundary.unknown, reference_boundary.unknown)
            and np.array_equal(boundary.index_grid, reference_boundary.index_grid)
            and boundary.periodic_x == reference_boundary.periodic_x
            and boundary.periodic_y == reference_boundary.periodic_y
        ):
            raise SolverFailedError("All vector PDE components must share one grid topology")

    components = len(boundaries)
    n_points = int(np.count_nonzero(reference_boundary.unknown))
    system_size = components * n_points
    max_entries = 9 * components * system_size
    rows = np.empty(max_entries, dtype=np.int64)
    cols = np.empty(max_entries, dtype=np.int64)
    data = np.empty(max_entries, dtype=float)
    entry_count = 0
    rhs = np.zeros(system_size, dtype=float)
    component_orientations: dict[int, tuple[int, float, float]] = {}

    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)
    inv_2hx = 1.0 / (2.0 * hx)
    inv_2hy = 1.0 / (2.0 * hy)
    inv_4hxhy = 1.0 / (4.0 * hx * hy)

    def append_entry(row: int, col: int, value: float) -> None:
        """Append a nonzero matrix contribution."""
        nonlocal entry_count
        if value != 0.0:
            rows[entry_count] = row
            cols[entry_count] = col
            data[entry_count] = value
            entry_count += 1

    grid_width = reference_boundary.unknown.shape[1]
    for point_index, flat_index in enumerate(np.flatnonzero(reference_boundary.unknown)):
        j, i = divmod(int(flat_index), grid_width)
        xi, yj = float(x[i]), float(y[j])
        coefficients, orientation = coefficient_at(xi, yj)
        domain_component = int(reference_boundary.component_labels[j, i])
        orientation_reference = component_orientations.get(domain_component)
        if orientation_reference is None:
            component_orientations[domain_component] = (orientation, xi, yj)
        elif orientation_reference[0] != orientation:
            expected = "positive definite" if orientation_reference[0] > 0 else "negative definite"
            actual = "positive definite" if orientation > 0 else "negative definite"
            raise SolverFailedError(
                "Vector PDE ellipticity orientation changes within connected domain component "
                f"{domain_component}: expected {expected} from "
                f"({orientation_reference[1]:.12g}, {orientation_reference[2]:.12g}), "
                f"got {actual} at ({xi:.12g}, {yj:.12g})"
            )

        for equation in range(components):
            row = equation * n_points + point_index
            rhs[row] = -coefficients.constant[equation]

            for solution_component in range(components):
                boundary = boundaries[solution_component]
                central = (
                    -2.0 * coefficients.fxx[equation, solution_component] * inv_hx2
                    - 2.0 * coefficients.fyy[equation, solution_component] * inv_hy2
                    + coefficients.f[equation, solution_component]
                )
                append_entry(
                    row,
                    solution_component * n_points + point_index,
                    central,
                )

                def add_neighbor(
                    raw_i: int,
                    raw_j: int,
                    coefficient: float,
                    *,
                    mixed_neighbor: bool = False,
                ) -> None:
                    """Add one component-aware neighbor or boundary substitution."""
                    if coefficient == 0.0:
                        return
                    normalized = wrapped_neighbor(raw_i, raw_j, boundary)
                    if normalized is None:
                        return
                    neighbor_i, neighbor_j = normalized
                    neighbor_index = int(boundary.index_grid[neighbor_j, neighbor_i])
                    if neighbor_index >= 0:
                        append_entry(
                            row,
                            solution_component * n_points + neighbor_index,
                            coefficient,
                        )
                        return
                    if not boundary.boundary[neighbor_j, neighbor_i]:
                        return
                    if boundary.kind[neighbor_j, neighbor_i] == BC_DIRICHLET:
                        rhs[row] -= coefficient * boundary.gamma[neighbor_j, neighbor_i]
                        return
                    inward_i, inward_j, factor, offset = boundary_substitution(
                        boundary,
                        boundary_i=neighbor_i,
                        boundary_j=neighbor_j,
                        source_i=i,
                        source_j=j,
                        hx=hx,
                        hy=hy,
                        mixed_neighbor=mixed_neighbor,
                    )
                    inward_index = int(boundary.index_grid[inward_j, inward_i])
                    if inward_index < 0:
                        raise SolverFailedError(
                            "Vector Neumann/Robin boundary elimination did not resolve "
                            "to an unknown point"
                        )
                    append_entry(
                        row,
                        solution_component * n_points + inward_index,
                        coefficient * factor,
                    )
                    rhs[row] -= coefficient * offset

                a_xx = coefficients.fxx[equation, solution_component]
                a_xy = coefficients.fxy[equation, solution_component]
                a_yy = coefficients.fyy[equation, solution_component]
                a_x = coefficients.fx[equation, solution_component]
                a_y = coefficients.fy[equation, solution_component]
                add_neighbor(i - 1, j, a_xx * inv_hx2 - a_x * inv_2hx)
                add_neighbor(i + 1, j, a_xx * inv_hx2 + a_x * inv_2hx)
                add_neighbor(i, j - 1, a_yy * inv_hy2 - a_y * inv_2hy)
                add_neighbor(i, j + 1, a_yy * inv_hy2 + a_y * inv_2hy)
                if a_xy != 0.0:
                    add_neighbor(
                        i + 1,
                        j + 1,
                        a_xy * inv_4hxhy,
                        mixed_neighbor=True,
                    )
                    add_neighbor(
                        i - 1,
                        j + 1,
                        -a_xy * inv_4hxhy,
                        mixed_neighbor=True,
                    )
                    add_neighbor(
                        i + 1,
                        j - 1,
                        -a_xy * inv_4hxhy,
                        mixed_neighbor=True,
                    )
                    add_neighbor(
                        i - 1,
                        j - 1,
                        a_xy * inv_4hxhy,
                        mixed_neighbor=True,
                    )

    matrix = sparse.coo_matrix(
        (data[:entry_count], (rows[:entry_count], cols[:entry_count])),
        shape=(system_size, system_size),
    ).tocsr()
    if not np.all(np.isfinite(matrix.data)):
        raise SolverFailedError("Assembled vector PDE matrix contains non-finite values")
    if not np.all(np.isfinite(rhs)):
        raise SolverFailedError("Assembled vector PDE right-hand side contains non-finite values")
    return AssembledPDE(matrix, rhs)


def solve_sparse_pde(
    assembled: AssembledPDE,
    *,
    warnings: tuple[str, ...],
) -> tuple[np.ndarray, PDEDiagnostics]:
    """Solve a sparse PDE system and compute bounded algebraic diagnostics."""
    import warnings as warning_control

    from scipy.sparse.linalg import MatrixRankWarning, spsolve

    if assembled.matrix.shape[0] > 0:
        constant_probe = np.ones(assembled.matrix.shape[1], dtype=float)
        constant_image = np.asarray(assembled.matrix @ constant_probe, dtype=float)
        row_scale = float(np.max(np.asarray(np.abs(assembled.matrix).sum(axis=1))))
        null_tolerance = 100.0 * np.finfo(float).eps * max(1.0, row_scale)
        if float(np.linalg.norm(constant_image, ord=np.inf)) <= null_tolerance:
            raise SolverFailedError(
                "Linear system is singular: the constant field is in its numerical nullspace"
            )

    try:
        with warning_control.catch_warnings():
            warning_control.simplefilter("error", MatrixRankWarning)
            solution = np.asarray(spsolve(assembled.matrix, assembled.rhs), dtype=float)
    except Exception as exc:
        raise SolverFailedError(f"Linear solver failed: {exc}") from exc
    if solution.shape != (assembled.matrix.shape[0],) or not np.all(np.isfinite(solution)):
        raise SolverFailedError("Linear solver returned a non-finite or invalid solution")

    applied = np.asarray(assembled.matrix @ solution, dtype=float)
    residual = applied - assembled.rhs
    residual_l2 = float(np.linalg.norm(residual))
    residual_linf = float(np.linalg.norm(residual, ord=np.inf)) if residual.size else 0.0
    rhs_scale = float(np.linalg.norm(assembled.rhs))
    applied_scale = float(np.linalg.norm(applied))
    relative_l2 = residual_l2 / max(rhs_scale, applied_scale, np.finfo(float).tiny)
    condition_estimate: float | None = None
    if assembled.matrix.shape[0] <= _MAX_CONDITION_ESTIMATE_SIZE:
        condition_estimate = float(np.linalg.cond(assembled.matrix.toarray()))
        rank_limit = 1.0 / (max(1, assembled.matrix.shape[0]) * np.finfo(float).eps)
        if not np.isfinite(condition_estimate) or condition_estimate >= rank_limit:
            raise SolverFailedError("Linear system is singular or numerically rank deficient")
    residual_scale = max(1.0, rhs_scale, applied_scale)
    if residual_l2 > _LINEAR_RESIDUAL_TOLERANCE * residual_scale:
        raise SolverFailedError(
            "Linear solve produced an excessive algebraic residual: "
            f"relative L2 residual={relative_l2:.3g}"
        )
    diagnostics = PDEDiagnostics(
        residual_l2,
        residual_linf,
        relative_l2,
        (int(assembled.matrix.shape[0]), int(assembled.matrix.shape[1])),
        int(assembled.matrix.nnz),
        condition_estimate,
        warnings,
    )
    return solution, diagnostics


def solve_sparse_vector_pde(
    assembled: AssembledPDE,
    *,
    components: int,
    points_per_component: int,
    warnings: tuple[str, ...],
) -> tuple[np.ndarray, VectorPDEDiagnostics]:
    """Solve a component-major block system and add per-equation residual evidence."""
    solution, global_diagnostics = solve_sparse_pde(assembled, warnings=warnings)
    applied = np.asarray(assembled.matrix @ solution, dtype=float)
    residual = applied - assembled.rhs
    component_l2: list[float] = []
    component_linf: list[float] = []
    component_relative: list[float] = []
    for component in range(components):
        component_slice = slice(
            component * points_per_component,
            (component + 1) * points_per_component,
        )
        component_residual = residual[component_slice]
        component_rhs = assembled.rhs[component_slice]
        component_applied = applied[component_slice]
        residual_l2 = float(np.linalg.norm(component_residual))
        component_l2.append(residual_l2)
        component_linf.append(
            float(np.linalg.norm(component_residual, ord=np.inf))
            if component_residual.size
            else 0.0
        )
        component_relative.append(
            residual_l2
            / max(
                float(np.linalg.norm(component_rhs)),
                float(np.linalg.norm(component_applied)),
                np.finfo(float).tiny,
            )
        )
    diagnostics = VectorPDEDiagnostics(
        discrete_residual_l2=global_diagnostics.discrete_residual_l2,
        discrete_residual_linf=global_diagnostics.discrete_residual_linf,
        relative_residual_l2=global_diagnostics.relative_residual_l2,
        component_residual_l2=tuple(component_l2),
        component_residual_linf=tuple(component_linf),
        component_relative_residual_l2=tuple(component_relative),
        matrix_shape=global_diagnostics.matrix_shape,
        nnz=global_diagnostics.nnz,
        condition_estimate=global_diagnostics.condition_estimate,
        warnings=global_diagnostics.warnings,
    )
    return solution, diagnostics
