"""Boundary validation, adaptation, substitution, and reconstruction for 2D PDEs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solver.pde_types import (
    BC_DIRICHLET,
    BC_NEUMANN,
    BC_ROBIN,
    BoundaryData,
    PDEBoundaryCondition,
    PDEBoundaryConditions,
)
from solver.pde_validation import connected_component_labels, finite_scalar
from utils import SolverFailedError

_CORNER_RTOL = 1.0e-10
_CORNER_ATOL = 1.0e-12


@dataclass(frozen=True)
class PreparedBoundary:
    """Validated pointwise boundary representation used by assembly."""

    mask: np.ndarray
    unknown: np.ndarray
    boundary: np.ndarray
    index_grid: np.ndarray
    component_labels: np.ndarray
    kind: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    periodic_x: bool
    periodic_y: bool
    structured: bool
    warnings: tuple[str, ...]


def classify_mask(
    mask: np.ndarray,
    *,
    periodic_x: bool = False,
    periodic_y: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify active points as unknown or boundary with optional axis wraps."""
    active = np.asarray(mask, dtype=bool)
    padded = np.pad(active, pad_width=1, mode="constant", constant_values=False)
    north = padded[:-2, 1:-1]
    south = padded[2:, 1:-1]
    west = padded[1:-1, :-2]
    east = padded[1:-1, 2:]
    if periodic_x:
        west[:, 0] = active[:, -1]
        east[:, -1] = active[:, 0]
    if periodic_y:
        north[0, :] = active[-1, :]
        south[-1, :] = active[0, :]
    unknown = active & north & south & west & east
    boundary = active & ~unknown
    index_grid = np.full(active.shape, -1, dtype=np.int64)
    index_grid[unknown] = np.arange(int(np.count_nonzero(unknown)), dtype=np.int64)
    return unknown, boundary, index_grid


def _float_array(value: object, name: str, shape: tuple[int, int]) -> np.ndarray:
    """Return a float array after validating its exact shape."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverFailedError(f"{name} must contain real numeric values") from exc
    if array.shape != shape:
        raise SolverFailedError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _validate_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """Validate a domain mask or create a full rectangular mask."""
    if mask is None:
        return np.ones(shape, dtype=bool)
    raw_mask = np.asarray(mask)
    if raw_mask.shape != shape:
        raise SolverFailedError(f"mask must have shape {shape}, got {raw_mask.shape}")
    if np.issubdtype(raw_mask.dtype, np.number) and not np.all(np.isfinite(raw_mask)):
        raise SolverFailedError("mask must not contain non-finite values")
    return np.asarray(raw_mask, dtype=bool)


def _evaluate_data(
    data: BoundaryData,
    *,
    name: str,
    i: int,
    j: int,
    x: np.ndarray,
    y: np.ndarray,
    shape: tuple[int, int],
) -> float:
    """Evaluate scalar, array, or callable boundary data at one grid point."""
    if callable(data):
        raw = data(float(x[i]), float(y[j]))
    else:
        array = np.asarray(data)
        if array.ndim == 0:
            raw = array
        elif array.shape == shape:
            raw = array[j, i]
        else:
            raise SolverFailedError(f"{name} must be scalar, callable, or have shape {shape}")
    return finite_scalar(raw, f"{name} at grid index ({i}, {j})")


def _condition_values(
    condition: PDEBoundaryCondition,
    *,
    i: int,
    j: int,
    x: np.ndarray,
    y: np.ndarray,
    shape: tuple[int, int],
) -> tuple[str, float, float, float]:
    """Materialize one structured condition into canonical coefficients."""
    if condition.kind not in (BC_DIRICHLET, BC_NEUMANN, BC_ROBIN):
        raise SolverFailedError(
            "Structured boundary kind must be 'dirichlet', 'neumann', or 'robin'; "
            f"got {condition.kind!r}"
        )
    gamma = _evaluate_data(
        condition.value,
        name=f"{condition.kind} value",
        i=i,
        j=j,
        x=x,
        y=y,
        shape=shape,
    )
    if condition.kind == BC_DIRICHLET:
        return condition.kind, 1.0, 0.0, gamma
    if condition.kind == BC_NEUMANN:
        return condition.kind, 0.0, 1.0, gamma
    alpha = _evaluate_data(
        condition.alpha,
        name="Robin alpha",
        i=i,
        j=j,
        x=x,
        y=y,
        shape=shape,
    )
    beta = _evaluate_data(
        condition.beta,
        name="Robin beta",
        i=i,
        j=j,
        x=x,
        y=y,
        shape=shape,
    )
    if alpha == 0.0 and beta == 0.0:
        raise SolverFailedError(
            f"Robin alpha and beta cannot both be zero at grid index ({i}, {j})"
        )
    return condition.kind, alpha, beta, gamma


def _merge_structured_corner(
    candidates: list[tuple[str, float, float, float]],
    *,
    i: int,
    j: int,
) -> tuple[str, float, float, float]:
    """Resolve explicitly supported structured corner combinations."""
    if len(candidates) == 1:
        return candidates[0]
    dirichlet = [candidate for candidate in candidates if candidate[0] == BC_DIRICHLET]
    if dirichlet:
        values = [candidate[3] for candidate in dirichlet]
        if not np.allclose(values, values[0], rtol=_CORNER_RTOL, atol=_CORNER_ATOL):
            raise SolverFailedError(
                f"Incompatible Dirichlet values at structured boundary corner ({i}, {j}): {values}"
            )
        return dirichlet[0]
    raise SolverFailedError(
        "Ambiguous structured boundary corner "
        f"({i}, {j}): two non-Dirichlet edges do not define one grid normal"
    )


def _prepare_structured(
    conditions: PDEBoundaryConditions,
    *,
    mask: np.ndarray | None,
    x: np.ndarray,
    y: np.ndarray,
) -> PreparedBoundary:
    """Validate and materialize structured edge or contour conditions."""
    shape = (len(y), len(x))
    if not isinstance(conditions.periodic_x, bool) or not isinstance(conditions.periodic_y, bool):
        raise SolverFailedError("periodic_x and periodic_y must be booleans")
    configured = (
        conditions.left,
        conditions.right,
        conditions.bottom,
        conditions.top,
        conditions.contour,
    )
    if any(
        condition is not None and not isinstance(condition, PDEBoundaryCondition)
        for condition in configured
    ):
        raise SolverFailedError("Structured edges and contour must be PDEBoundaryCondition values")
    if conditions.periodic_x and (conditions.left is not None or conditions.right is not None):
        raise SolverFailedError("A periodic x-axis cannot also define left/right conditions")
    if conditions.periodic_y and (conditions.bottom is not None or conditions.top is not None):
        raise SolverFailedError("A periodic y-axis cannot also define bottom/top conditions")
    if mask is not None and (conditions.periodic_x or conditions.periodic_y):
        raise SolverFailedError(
            "Periodic axes cannot be combined with an arbitrary mask; masked wrap topology "
            "is not implemented"
        )
    if mask is not None:
        if conditions.contour is None:
            raise SolverFailedError("Structured conditions on an arbitrary mask require contour")
        if any(
            edge is not None
            for edge in (conditions.left, conditions.right, conditions.bottom, conditions.top)
        ):
            raise SolverFailedError(
                "Structured masked domains use contour, not rectangular edge conditions"
            )
    elif conditions.contour is not None:
        raise SolverFailedError("contour is only valid when an arbitrary mask is supplied")

    mask_array = _validate_mask(mask, shape)
    unknown, boundary, index_grid = classify_mask(
        mask_array,
        periodic_x=conditions.periodic_x,
        periodic_y=conditions.periodic_y,
    )
    kinds = np.full(shape, BC_DIRICHLET, dtype=object)
    alpha = np.ones(shape, dtype=float)
    beta = np.zeros(shape, dtype=float)
    gamma = np.zeros(shape, dtype=float)

    if mask is not None:
        assert conditions.contour is not None
        for j, i in np.argwhere(boundary):
            values = _condition_values(
                conditions.contour,
                i=int(i),
                j=int(j),
                x=x,
                y=y,
                shape=shape,
            )
            kinds[j, i], alpha[j, i], beta[j, i], gamma[j, i] = values
    else:
        default = PDEBoundaryCondition.dirichlet()
        edges: list[tuple[PDEBoundaryCondition, list[tuple[int, int]]]] = []
        if not conditions.periodic_x:
            edges.extend(
                (
                    (conditions.left or default, [(0, j) for j in range(shape[0])]),
                    (
                        conditions.right or default,
                        [(shape[1] - 1, j) for j in range(shape[0])],
                    ),
                )
            )
        if not conditions.periodic_y:
            edges.extend(
                (
                    (conditions.bottom or default, [(i, 0) for i in range(shape[1])]),
                    (
                        conditions.top or default,
                        [(i, shape[0] - 1) for i in range(shape[1])],
                    ),
                )
            )
        candidates: dict[tuple[int, int], list[tuple[str, float, float, float]]] = {}
        for condition, points in edges:
            for i, j in points:
                values = _condition_values(
                    condition,
                    i=i,
                    j=j,
                    x=x,
                    y=y,
                    shape=shape,
                )
                candidates.setdefault((i, j), []).append(values)
        for (i, j), point_candidates in candidates.items():
            values = _merge_structured_corner(point_candidates, i=i, j=j)
            kinds[j, i], alpha[j, i], beta[j, i], gamma[j, i] = values

    flux = boundary & np.isin(kinds, (BC_NEUMANN, BC_ROBIN))
    warnings = (
        (
            "Neumann/Robin conditions on masked boundaries use grid-normal, "
            "not geometric-normal, derivatives.",
        )
        if mask is not None and np.any(flux)
        else ()
    )
    return PreparedBoundary(
        mask_array,
        unknown,
        boundary,
        index_grid,
        connected_component_labels(
            mask_array,
            periodic_x=conditions.periodic_x,
            periodic_y=conditions.periodic_y,
        ),
        kinds,
        alpha,
        beta,
        gamma,
        conditions.periodic_x,
        conditions.periodic_y,
        True,
        warnings,
    )


def _prepare_legacy(
    *,
    nx: int,
    ny: int,
    mask: np.ndarray | None,
    bc_values: np.ndarray | None,
    bc_type: np.ndarray | None,
    bc_neumann_value: np.ndarray | None,
) -> PreparedBoundary:
    """Adapt legacy point arrays to canonical Robin coefficients."""
    shape = (ny, nx)
    mask_array = _validate_mask(mask, shape)
    values = np.zeros(shape) if bc_values is None else _float_array(bc_values, "bc_values", shape)
    neumann = (
        np.zeros(shape)
        if bc_neumann_value is None
        else _float_array(bc_neumann_value, "bc_neumann_value", shape)
    )
    if bc_type is None:
        kinds = np.full(shape, BC_DIRICHLET, dtype=object)
    else:
        kinds = np.asarray(bc_type, dtype=object)
        if kinds.shape != shape:
            raise SolverFailedError(f"bc_type must have shape {shape}, got {kinds.shape}")
    unknown, boundary, index_grid = classify_mask(mask_array)
    invalid = [label for label in kinds[boundary] if label not in (BC_DIRICHLET, BC_NEUMANN)]
    if invalid:
        raise SolverFailedError(
            f"bc_type boundary labels must be 'dirichlet' or 'neumann'; got {invalid[0]!r}"
        )
    dirichlet = boundary & (kinds == BC_DIRICHLET)
    neumann_mask = boundary & (kinds == BC_NEUMANN)
    if not np.all(np.isfinite(values[dirichlet])):
        raise SolverFailedError("bc_values must be finite at Dirichlet boundary points")
    if not np.all(np.isfinite(neumann[neumann_mask])):
        raise SolverFailedError("bc_neumann_value must be finite at Neumann boundary points")
    alpha = np.where(kinds == BC_DIRICHLET, 1.0, 0.0)
    beta = np.where(kinds == BC_NEUMANN, 1.0, 0.0)
    gamma = np.where(kinds == BC_NEUMANN, neumann, values)
    warnings = (
        (
            "Neumann conditions on masked boundaries use grid-normal, "
            "not geometric-normal, derivatives.",
        )
        if mask is not None and np.any(neumann_mask)
        else ()
    )
    return PreparedBoundary(
        mask_array,
        unknown,
        boundary,
        index_grid,
        connected_component_labels(mask_array, periodic_x=False, periodic_y=False),
        kinds,
        alpha,
        beta,
        gamma,
        False,
        False,
        False,
        warnings,
    )


def prepare_boundary(
    *,
    nx: int,
    ny: int,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray | None,
    bc_values: np.ndarray | None,
    bc_type: np.ndarray | None,
    bc_neumann_value: np.ndarray | None,
    boundary_conditions: PDEBoundaryConditions | None,
) -> PreparedBoundary:
    """Prepare structured boundary data or adapt the backwards-compatible arrays."""
    if boundary_conditions is not None:
        if any(value is not None for value in (bc_values, bc_type, bc_neumann_value)):
            raise SolverFailedError(
                "boundary_conditions cannot be combined with legacy bc_values, bc_type, "
                "or bc_neumann_value"
            )
        return _prepare_structured(boundary_conditions, mask=mask, x=x, y=y)
    return _prepare_legacy(
        nx=nx,
        ny=ny,
        mask=mask,
        bc_values=bc_values,
        bc_type=bc_type,
        bc_neumann_value=bc_neumann_value,
    )


def wrapped_neighbor(
    i: int,
    j: int,
    boundary: PreparedBoundary,
) -> tuple[int, int] | None:
    """Return an in-grid neighbor, wrapping configured periodic axes."""
    ny, nx = boundary.mask.shape
    if boundary.periodic_x:
        i %= nx
    if boundary.periodic_y:
        j %= ny
    if 0 <= i < nx and 0 <= j < ny:
        return i, j
    return None


def _substitution_coefficients(
    boundary: PreparedBoundary,
    *,
    i: int,
    j: int,
    step: float,
) -> tuple[float, float]:
    """Return factor/offset in ``u_boundary = factor*u_inward + offset``."""
    alpha = float(boundary.alpha[j, i])
    beta_over_h = float(boundary.beta[j, i]) / step
    denominator = alpha + beta_over_h
    scale = max(abs(alpha), abs(beta_over_h), np.finfo(float).tiny)
    if abs(denominator) <= 100.0 * np.finfo(float).eps * scale:
        raise SolverFailedError(
            f"Robin boundary substitution denominator is numerically zero at grid index ({i}, {j})"
        )
    return beta_over_h / denominator, float(boundary.gamma[j, i]) / denominator


def boundary_substitution(
    boundary: PreparedBoundary,
    *,
    boundary_i: int,
    boundary_j: int,
    source_i: int,
    source_j: int,
    hx: float,
    hy: float,
    mixed_neighbor: bool,
) -> tuple[int, int, float, float]:
    """Express a Neumann/Robin boundary value using one grid-normal neighbor."""
    ny, nx = boundary.mask.shape
    physical: list[tuple[int, int, float]] = []
    if boundary_i == 0 and not boundary.periodic_x:
        physical.append((boundary_i + 1, boundary_j, hx))
    if boundary_i == nx - 1 and not boundary.periodic_x:
        physical.append((boundary_i - 1, boundary_j, hx))
    if boundary_j == 0 and not boundary.periodic_y:
        physical.append((boundary_i, boundary_j + 1, hy))
    if boundary_j == ny - 1 and not boundary.periodic_y:
        physical.append((boundary_i, boundary_j - 1, hy))
    valid = [candidate for candidate in physical if boundary.unknown[candidate[1], candidate[0]]]
    if len(valid) == 1:
        inward_i, inward_j, step = valid[0]
    else:
        delta_i = boundary_i - source_i
        delta_j = boundary_j - source_j
        if mixed_neighbor or abs(delta_i) + abs(delta_j) != 1:
            kind = str(boundary.kind[boundary_j, boundary_i]).capitalize()
            raise SolverFailedError(
                f"A mixed-derivative stencil adjacent to a {kind} boundary requires "
                "a unique rectangular grid normal; masked-boundary and corner diagonals "
                f"are not supported at grid index ({boundary_i}, {boundary_j})"
            )
        inward_i, inward_j = source_i, source_j
        step = hx if delta_i != 0 else hy
    factor, offset = _substitution_coefficients(
        boundary,
        i=boundary_i,
        j=boundary_j,
        step=step,
    )
    return inward_i, inward_j, factor, offset


def apply_boundary_values(
    u: np.ndarray,
    boundary: PreparedBoundary,
    *,
    hx: float,
    hy: float,
) -> None:
    """Fill boundary output values, rejecting unresolved or incompatible normals."""
    dirichlet = boundary.boundary & (boundary.kind == BC_DIRICHLET)
    u[dirichlet] = boundary.gamma[dirichlet]
    unresolved = {
        (int(i), int(j))
        for j, i in np.argwhere(boundary.boundary & np.isin(boundary.kind, (BC_NEUMANN, BC_ROBIN)))
    }
    ny, nx = u.shape
    while unresolved:
        progressed = False
        for i, j in tuple(unresolved):
            candidates: list[tuple[int, int, float]] = []
            if i == 0 and not boundary.periodic_x:
                candidates.append((i + 1, j, hx))
            if i == nx - 1 and not boundary.periodic_x:
                candidates.append((i - 1, j, hx))
            if j == 0 and not boundary.periodic_y:
                candidates.append((i, j + 1, hy))
            if j == ny - 1 and not boundary.periodic_y:
                candidates.append((i, j - 1, hy))
            if len(candidates) > 1:
                if boundary.structured:
                    raise SolverFailedError(
                        "Ambiguous structured boundary corner "
                        f"({i}, {j}): no unique grid normal is available"
                    )
                matching_normals = [
                    candidate
                    for index, candidate in enumerate(candidates)
                    if boundary.kind[candidates[1 - index][1], candidates[1 - index][0]]
                    == boundary.kind[j, i]
                ]
                if len(matching_normals) != 1:
                    raise SolverFailedError(
                        "Legacy boundary corner cannot be assigned a unique grid normal at "
                        f"grid index ({i}, {j}); use open-edge labels or structured conditions"
                    )
                candidates = matching_normals
            if not candidates:
                for di, dj, step in ((1, 0, hx), (-1, 0, hx), (0, 1, hy), (0, -1, hy)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < nx and 0 <= nj < ny and boundary.unknown[nj, ni]:
                        candidates.append((ni, nj, step))
            predictions: list[float] = []
            for ni, nj, step in candidates:
                if np.isfinite(u[nj, ni]):
                    factor, offset = _substitution_coefficients(
                        boundary,
                        i=i,
                        j=j,
                        step=step,
                    )
                    predictions.append(factor * float(u[nj, ni]) + offset)
            if not predictions:
                continue
            if not np.allclose(
                predictions,
                predictions[0],
                rtol=_CORNER_RTOL,
                atol=_CORNER_ATOL,
            ):
                raise SolverFailedError(
                    "Incompatible grid-normal boundary reconstructions at corner/masked point "
                    f"({i}, {j}): {predictions}"
                )
            u[j, i] = float(np.mean(predictions))
            unresolved.remove((i, j))
            progressed = True
        if not progressed:
            i, j = min(unresolved)
            raise SolverFailedError(
                "Boundary value cannot be reconstructed unambiguously from an interior "
                f"grid-normal neighbor at grid index ({i}, {j})"
            )
