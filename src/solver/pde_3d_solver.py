"""Scalar linear elliptic finite-difference solver on rectangular 3D grids."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np

from solver.pde_assembly import AssembledPDE, solve_sparse_pde
from solver.pde_types import (
    BC_DIRICHLET,
    BC_NEUMANN,
    BC_ROBIN,
    PDEBoundaryCondition3D,
    PDEBoundaryConditions3D,
    PDECoefficientProvider3D,
    PDECoefficients3D,
    PDEDiagnostics,
    PDESolution3D,
)
from solver.pde_validation import finite_scalar
from utils import SolverFailedError, get_logger, normalize_params

logger = get_logger(__name__)

_AFFINITY_PROBE_3D = (0.37, -0.61, 1.19, -0.83, 0.47, 1.31, -1.07, 0.73, -0.29, 0.91)
_AFFINITY_TOLERANCE = 1.0e-9
_BOUNDARY_RTOL = 1.0e-10
_BOUNDARY_ATOL = 1.0e-12


@dataclass(frozen=True)
class _PreparedBoundary3D:
    """Materialized rectangular boundary topology and values."""

    unknown: np.ndarray
    boundary: np.ndarray
    index_grid: np.ndarray
    kind: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    inward_i: np.ndarray
    inward_j: np.ndarray
    inward_k: np.ndarray
    step: np.ndarray
    periodic_x: bool
    periodic_y: bool
    periodic_z: bool


def _validate_grid_inputs_3d(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[float, float, float, float, float, float, int, int, int]:
    """Validate and normalize a rectangular 3D grid."""
    bounds = tuple(
        finite_scalar(value, name)
        for value, name in (
            (x_min, "x_min"),
            (x_max, "x_max"),
            (y_min, "y_min"),
            (y_max, "y_max"),
            (z_min, "z_min"),
            (z_max, "z_max"),
        )
    )
    x_start, x_end, y_start, y_end, z_start, z_end = bounds
    for start, end, name in (
        (x_start, x_end, "x"),
        (y_start, y_end, "y"),
        (z_start, z_end, "z"),
    ):
        if start >= end:
            raise SolverFailedError(f"{name}_min must be strictly less than {name}_max")
    counts: list[int] = []
    for value, name in ((nx, "nx"), (ny, "ny"), (nz, "nz")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise SolverFailedError(f"{name} must be an integer")
        if int(value) < 3:
            raise SolverFailedError("3D grids must have at least 3 points per dimension")
        counts.append(int(value))
    return (
        x_start,
        x_end,
        y_start,
        y_end,
        z_start,
        z_end,
        counts[0],
        counts[1],
        counts[2],
    )


def _probe_coefficients_3d(
    residual_func: Callable[..., float],
    xi: float,
    yj: float,
    zk: float,
    params: dict[str, float],
) -> PDECoefficients3D:
    """Recover coefficients and verify residual affinity in the complete 3D state."""
    coordinate = f"({xi:.12g}, {yj:.12g}, {zk:.12g})"

    def evaluate(state: tuple[float, ...]) -> float:
        try:
            raw = residual_func(xi, yj, zk, *state, **params)
        except SolverFailedError:
            raise
        except Exception as exc:
            raise SolverFailedError(
                f"3D PDE residual evaluation failed at {coordinate}: {exc}"
            ) from exc
        return finite_scalar(raw, f"3D PDE residual at {coordinate}")

    zero = (0.0,) * 10
    constant = evaluate(zero)
    responses: list[float] = []
    for index in range(10):
        basis = [0.0] * 10
        basis[index] = 1.0
        responses.append(evaluate(tuple(basis)) - constant)
    actual = evaluate(_AFFINITY_PROBE_3D)
    reconstructed = constant + sum(
        coefficient * value
        for coefficient, value in zip(responses, _AFFINITY_PROBE_3D, strict=True)
    )
    scale = max(
        1.0,
        abs(actual),
        abs(reconstructed),
        abs(constant)
        + sum(
            abs(coefficient * value)
            for coefficient, value in zip(responses, _AFFINITY_PROBE_3D, strict=True)
        ),
    )
    mismatch = abs(actual - reconstructed)
    if mismatch > _AFFINITY_TOLERANCE * scale:
        raise SolverFailedError(
            "3D PDE residual is not affine in "
            "(f, fx, fy, fz, fxx, fxy, fxz, fyy, fyz, fzz) "
            f"at {coordinate}: probe mismatch {mismatch:.3g}"
        )
    f, fx, fy, fz, fxx, fxy, fxz, fyy, fyz, fzz = responses
    return (fxx, fyy, fzz, fxy, fxz, fyz, fx, fy, fz, f, constant)


def _validate_coefficients_3d(
    coefficients: object,
    xi: float,
    yj: float,
    zk: float,
) -> tuple[PDECoefficients3D, int]:
    """Validate coefficient shape, finiteness, and strict 3D ellipticity."""
    coordinate = f"({xi:.12g}, {yj:.12g}, {zk:.12g})"
    try:
        array = np.asarray(coefficients, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverFailedError(
            f"3D PDE coefficients at {coordinate} must be eleven real scalars"
        ) from exc
    if array.shape != (11,):
        raise SolverFailedError(
            f"3D PDE coefficients at {coordinate} must have shape (11,), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise SolverFailedError(f"3D PDE coefficients at {coordinate} must be finite")

    fxx, fyy, fzz, fxy, fxz, fyz = (float(value) for value in array[:6])
    principal = np.array(
        (
            (fxx, 0.5 * fxy, 0.5 * fxz),
            (0.5 * fxy, fyy, 0.5 * fyz),
            (0.5 * fxz, 0.5 * fyz, fzz),
        )
    )
    principal_scale = float(np.max(np.abs(principal)))
    if principal_scale <= np.finfo(float).tiny:
        raise SolverFailedError(f"3D PDE principal operator is degenerate at {coordinate}")
    eigenvalues = np.linalg.eigvalsh(principal)
    margin = 100.0 * np.finfo(float).eps * principal_scale
    positive = bool(np.all(eigenvalues > margin))
    negative = bool(np.all(eigenvalues < -margin))
    if not (positive or negative):
        raise SolverFailedError(
            "3D PDE principal operator is not strictly elliptic "
            f"at {coordinate}: eigenvalues={eigenvalues.tolist()}"
        )
    normalized = cast(PDECoefficients3D, tuple(float(value) for value in array))
    return normalized, 1 if positive else -1


def _evaluate_boundary_data(
    data: object,
    *,
    name: str,
    i: int,
    j: int,
    k: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    shape: tuple[int, int, int],
) -> float:
    """Evaluate scalar, callable, or full-grid 3D boundary data."""
    if callable(data):
        raw = data(float(x[i]), float(y[j]), float(z[k]))
    else:
        array = np.asarray(data)
        if array.ndim == 0:
            raw = array
        elif array.shape == shape:
            raw = array[k, j, i]
        else:
            raise SolverFailedError(f"{name} must be scalar, callable, or have shape {shape}")
    return finite_scalar(raw, f"{name} at grid index ({i}, {j}, {k})")


def _condition_values_3d(
    condition: PDEBoundaryCondition3D,
    *,
    i: int,
    j: int,
    k: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[str, float, float, float]:
    """Materialize one 3D face condition at one grid point."""
    if condition.kind not in (BC_DIRICHLET, BC_NEUMANN, BC_ROBIN):
        raise SolverFailedError(
            f"3D boundary kind must be 'dirichlet', 'neumann', or 'robin'; got {condition.kind!r}"
        )
    gamma = _evaluate_boundary_data(
        condition.value,
        name=f"{condition.kind} value",
        i=i,
        j=j,
        k=k,
        x=x,
        y=y,
        z=z,
        shape=shape,
    )
    if condition.kind == BC_DIRICHLET:
        return condition.kind, 1.0, 0.0, gamma
    if condition.kind == BC_NEUMANN:
        return condition.kind, 0.0, 1.0, gamma
    alpha = _evaluate_boundary_data(
        condition.alpha,
        name="Robin alpha",
        i=i,
        j=j,
        k=k,
        x=x,
        y=y,
        z=z,
        shape=shape,
    )
    beta = _evaluate_boundary_data(
        condition.beta,
        name="Robin beta",
        i=i,
        j=j,
        k=k,
        x=x,
        y=y,
        z=z,
        shape=shape,
    )
    if alpha == 0.0 and beta == 0.0:
        raise SolverFailedError(
            f"Robin alpha and beta cannot both be zero at grid index ({i}, {j}, {k})"
        )
    return condition.kind, alpha, beta, gamma


def _prepare_boundary_3d(
    conditions: PDEBoundaryConditions3D | None,
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    hx: float,
    hy: float,
    hz: float,
) -> _PreparedBoundary3D:
    """Validate six rectangular faces and build pointwise substitutions."""
    conditions = conditions or PDEBoundaryConditions3D()
    if not isinstance(conditions, PDEBoundaryConditions3D):
        raise SolverFailedError("boundary_conditions must be a PDEBoundaryConditions3D object")
    periodic = (conditions.periodic_x, conditions.periodic_y, conditions.periodic_z)
    if any(not isinstance(value, bool) for value in periodic):
        raise SolverFailedError("periodic_x, periodic_y, and periodic_z must be booleans")
    faces = (
        conditions.x_min,
        conditions.x_max,
        conditions.y_min,
        conditions.y_max,
        conditions.z_min,
        conditions.z_max,
    )
    if any(face is not None and not isinstance(face, PDEBoundaryCondition3D) for face in faces):
        raise SolverFailedError("3D boundary faces must be PDEBoundaryCondition3D values")
    for axis, enabled, pair in (
        ("x", conditions.periodic_x, (conditions.x_min, conditions.x_max)),
        ("y", conditions.periodic_y, (conditions.y_min, conditions.y_max)),
        ("z", conditions.periodic_z, (conditions.z_min, conditions.z_max)),
    ):
        if enabled and any(face is not None for face in pair):
            raise SolverFailedError(
                f"A periodic {axis}-axis cannot also define {axis}_min/{axis}_max conditions"
            )

    nz, ny, nx = len(z), len(y), len(x)
    shape = (nz, ny, nx)
    unknown = np.ones(shape, dtype=bool)
    if not conditions.periodic_x:
        unknown[:, :, (0, -1)] = False
    if not conditions.periodic_y:
        unknown[:, (0, -1), :] = False
    if not conditions.periodic_z:
        unknown[(0, -1), :, :] = False
    boundary = ~unknown
    index_grid = np.full(shape, -1, dtype=np.int64)
    index_grid[unknown] = np.arange(int(np.count_nonzero(unknown)), dtype=np.int64)
    kinds = np.full(shape, BC_DIRICHLET, dtype=object)
    alpha = np.ones(shape, dtype=float)
    beta = np.zeros(shape, dtype=float)
    gamma = np.zeros(shape, dtype=float)
    inward_i = np.full(shape, -1, dtype=np.int64)
    inward_j = np.full(shape, -1, dtype=np.int64)
    inward_k = np.full(shape, -1, dtype=np.int64)
    steps = np.zeros(shape, dtype=float)
    default = PDEBoundaryCondition3D.dirichlet()

    for k_raw, j_raw, i_raw in np.argwhere(boundary):
        i, j, k = int(i_raw), int(j_raw), int(k_raw)
        candidates: list[tuple[tuple[str, float, float, float], tuple[int, int, int], float]] = []
        if not conditions.periodic_x and i == 0:
            candidates.append(
                (
                    _condition_values_3d(
                        conditions.x_min or default,
                        i=i,
                        j=j,
                        k=k,
                        x=x,
                        y=y,
                        z=z,
                        shape=shape,
                    ),
                    (i + 1, j, k),
                    hx,
                )
            )
        if not conditions.periodic_x and i == nx - 1:
            candidates.append(
                (
                    _condition_values_3d(
                        conditions.x_max or default,
                        i=i,
                        j=j,
                        k=k,
                        x=x,
                        y=y,
                        z=z,
                        shape=shape,
                    ),
                    (i - 1, j, k),
                    hx,
                )
            )
        if not conditions.periodic_y and j == 0:
            candidates.append(
                (
                    _condition_values_3d(
                        conditions.y_min or default,
                        i=i,
                        j=j,
                        k=k,
                        x=x,
                        y=y,
                        z=z,
                        shape=shape,
                    ),
                    (i, j + 1, k),
                    hy,
                )
            )
        if not conditions.periodic_y and j == ny - 1:
            candidates.append(
                (
                    _condition_values_3d(
                        conditions.y_max or default,
                        i=i,
                        j=j,
                        k=k,
                        x=x,
                        y=y,
                        z=z,
                        shape=shape,
                    ),
                    (i, j - 1, k),
                    hy,
                )
            )
        if not conditions.periodic_z and k == 0:
            candidates.append(
                (
                    _condition_values_3d(
                        conditions.z_min or default,
                        i=i,
                        j=j,
                        k=k,
                        x=x,
                        y=y,
                        z=z,
                        shape=shape,
                    ),
                    (i, j, k + 1),
                    hz,
                )
            )
        if not conditions.periodic_z and k == nz - 1:
            candidates.append(
                (
                    _condition_values_3d(
                        conditions.z_max or default,
                        i=i,
                        j=j,
                        k=k,
                        x=x,
                        y=y,
                        z=z,
                        shape=shape,
                    ),
                    (i, j, k - 1),
                    hz,
                )
            )
        if not candidates:
            raise SolverFailedError(
                f"3D boundary topology is inconsistent at grid index ({i}, {j}, {k})"
            )
        dirichlet = [candidate for candidate in candidates if candidate[0][0] == BC_DIRICHLET]
        if dirichlet:
            values = [candidate[0][3] for candidate in dirichlet]
            if not np.allclose(values, values[0], rtol=_BOUNDARY_RTOL, atol=_BOUNDARY_ATOL):
                raise SolverFailedError(
                    "Incompatible Dirichlet values at 3D boundary edge/corner "
                    f"({i}, {j}, {k}): {values}"
                )
            selected = dirichlet[0]
        elif len(candidates) == 1:
            selected = candidates[0]
        else:
            raise SolverFailedError(
                "Ambiguous 3D boundary edge/corner "
                f"({i}, {j}, {k}): multiple non-Dirichlet faces do not define one grid normal"
            )
        values, inward, step = selected
        kinds[k, j, i], alpha[k, j, i], beta[k, j, i], gamma[k, j, i] = values
        inward_i[k, j, i], inward_j[k, j, i], inward_k[k, j, i] = inward
        steps[k, j, i] = step
        if values[0] != BC_DIRICHLET:
            denominator = values[1] + values[2] / step
            scale = max(1.0, abs(values[1]), abs(values[2] / step))
            if abs(denominator) <= 100.0 * np.finfo(float).eps * scale:
                raise SolverFailedError(
                    "Robin boundary substitution denominator is numerically zero at "
                    f"grid index ({i}, {j}, {k})"
                )

    return _PreparedBoundary3D(
        unknown=unknown,
        boundary=boundary,
        index_grid=index_grid,
        kind=kinds,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        inward_i=inward_i,
        inward_j=inward_j,
        inward_k=inward_k,
        step=steps,
        periodic_x=conditions.periodic_x,
        periodic_y=conditions.periodic_y,
        periodic_z=conditions.periodic_z,
    )


def _boundary_substitution_3d(
    boundary: _PreparedBoundary3D,
    i: int,
    j: int,
    k: int,
) -> tuple[int, int, int, float, float]:
    """Return inward point and affine boundary substitution coefficients."""
    step = float(boundary.step[k, j, i])
    alpha = float(boundary.alpha[k, j, i])
    beta = float(boundary.beta[k, j, i])
    gamma = float(boundary.gamma[k, j, i])
    denominator = alpha + beta / step
    factor = (beta / step) / denominator
    offset = gamma / denominator
    return (
        int(boundary.inward_i[k, j, i]),
        int(boundary.inward_j[k, j, i]),
        int(boundary.inward_k[k, j, i]),
        factor,
        offset,
    )


def _wrapped_neighbor_3d(
    i: int,
    j: int,
    k: int,
    boundary: _PreparedBoundary3D,
) -> tuple[int, int, int] | None:
    """Normalize a neighbor across periodic seams or reject outside points."""
    nz, ny, nx = boundary.unknown.shape
    if boundary.periodic_x:
        i %= nx
    if boundary.periodic_y:
        j %= ny
    if boundary.periodic_z:
        k %= nz
    if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
        return None
    return i, j, k


def _assemble_pde_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    hx: float,
    hy: float,
    hz: float,
    boundary: _PreparedBoundary3D,
    coefficient_at: Callable[[float, float, float], tuple[PDECoefficients3D, int]],
) -> AssembledPDE:
    """Assemble the 19-point 3D stencil, including mixed periodic wraps."""
    from scipy import sparse

    n_unknown = int(np.count_nonzero(boundary.unknown))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(n_unknown, dtype=float)
    orientation_reference: tuple[int, float, float, float] | None = None
    inv_hx2 = 1.0 / hx**2
    inv_hy2 = 1.0 / hy**2
    inv_hz2 = 1.0 / hz**2
    inv_2hx = 1.0 / (2.0 * hx)
    inv_2hy = 1.0 / (2.0 * hy)
    inv_2hz = 1.0 / (2.0 * hz)
    inv_4hxhy = 1.0 / (4.0 * hx * hy)
    inv_4hxhz = 1.0 / (4.0 * hx * hz)
    inv_4hyhz = 1.0 / (4.0 * hy * hz)

    def append_entry(row: int, col: int, value: float) -> None:
        """Append a nonzero sparse contribution."""
        if value != 0.0:
            rows.append(row)
            cols.append(col)
            data.append(value)

    for row, (k_raw, j_raw, i_raw) in enumerate(np.argwhere(boundary.unknown)):
        i, j, k = int(i_raw), int(j_raw), int(k_raw)
        xi, yj, zk = float(x[i]), float(y[j]), float(z[k])
        coefficients, orientation = coefficient_at(xi, yj, zk)
        if orientation_reference is None:
            orientation_reference = (orientation, xi, yj, zk)
        elif orientation_reference[0] != orientation:
            expected = "positive definite" if orientation_reference[0] > 0 else "negative definite"
            actual = "positive definite" if orientation > 0 else "negative definite"
            raise SolverFailedError(
                "3D PDE ellipticity orientation changes within the rectangular domain: "
                f"expected {expected} from ({orientation_reference[1]:.12g}, "
                f"{orientation_reference[2]:.12g}, {orientation_reference[3]:.12g}), "
                f"got {actual} at ({xi:.12g}, {yj:.12g}, {zk:.12g})"
            )
        fxx, fyy, fzz, fxy, fxz, fyz, fx, fy, fz, f, constant = coefficients
        rhs[row] = -constant
        append_entry(
            row,
            row,
            -2.0 * fxx * inv_hx2 - 2.0 * fyy * inv_hy2 - 2.0 * fzz * inv_hz2 + f,
        )

        def add_neighbor(raw_i: int, raw_j: int, raw_k: int, coefficient: float) -> None:
            """Add an unknown, wrapped, or eliminated face neighbor."""
            if coefficient == 0.0:
                return
            normalized = _wrapped_neighbor_3d(raw_i, raw_j, raw_k, boundary)
            if normalized is None:
                return
            neighbor_i, neighbor_j, neighbor_k = normalized
            neighbor_index = int(boundary.index_grid[neighbor_k, neighbor_j, neighbor_i])
            if neighbor_index >= 0:
                append_entry(row, neighbor_index, coefficient)
                return
            if not boundary.boundary[neighbor_k, neighbor_j, neighbor_i]:
                return
            if boundary.kind[neighbor_k, neighbor_j, neighbor_i] == BC_DIRICHLET:
                rhs[row] -= coefficient * boundary.gamma[neighbor_k, neighbor_j, neighbor_i]
                return
            inward_i, inward_j, inward_k, factor, offset = _boundary_substitution_3d(
                boundary,
                neighbor_i,
                neighbor_j,
                neighbor_k,
            )
            inward_index = int(boundary.index_grid[inward_k, inward_j, inward_i])
            if inward_index < 0:
                raise SolverFailedError(
                    "3D Neumann/Robin boundary elimination did not resolve to an unknown point"
                )
            append_entry(row, inward_index, coefficient * factor)
            rhs[row] -= coefficient * offset

        add_neighbor(i - 1, j, k, fxx * inv_hx2 - fx * inv_2hx)
        add_neighbor(i + 1, j, k, fxx * inv_hx2 + fx * inv_2hx)
        add_neighbor(i, j - 1, k, fyy * inv_hy2 - fy * inv_2hy)
        add_neighbor(i, j + 1, k, fyy * inv_hy2 + fy * inv_2hy)
        add_neighbor(i, j, k - 1, fzz * inv_hz2 - fz * inv_2hz)
        add_neighbor(i, j, k + 1, fzz * inv_hz2 + fz * inv_2hz)
        for di, dj, sign in ((1, 1, 1.0), (-1, 1, -1.0), (1, -1, -1.0), (-1, -1, 1.0)):
            add_neighbor(i + di, j + dj, k, sign * fxy * inv_4hxhy)
        for di, dk, sign in ((1, 1, 1.0), (-1, 1, -1.0), (1, -1, -1.0), (-1, -1, 1.0)):
            add_neighbor(i + di, j, k + dk, sign * fxz * inv_4hxhz)
        for dj, dk, sign in ((1, 1, 1.0), (-1, 1, -1.0), (1, -1, -1.0), (-1, -1, 1.0)):
            add_neighbor(i, j + dj, k + dk, sign * fyz * inv_4hyhz)

    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_unknown, n_unknown)).tocsr()
    if not np.all(np.isfinite(matrix.data)):
        raise SolverFailedError("Assembled 3D PDE matrix contains non-finite values")
    if not np.all(np.isfinite(rhs)):
        raise SolverFailedError("Assembled 3D PDE right-hand side contains non-finite values")
    return AssembledPDE(matrix=matrix, rhs=rhs)


def _apply_boundary_values_3d(u: np.ndarray, boundary: _PreparedBoundary3D) -> None:
    """Fill Dirichlet faces and reconstruct Neumann/Robin faces."""
    dirichlet = boundary.boundary & (boundary.kind == BC_DIRICHLET)
    u[dirichlet] = boundary.gamma[dirichlet]
    for k_raw, j_raw, i_raw in np.argwhere(boundary.boundary & (boundary.kind != BC_DIRICHLET)):
        i, j, k = int(i_raw), int(j_raw), int(k_raw)
        inward_i, inward_j, inward_k, factor, offset = _boundary_substitution_3d(boundary, i, j, k)
        inward_value = float(u[inward_k, inward_j, inward_i])
        if not np.isfinite(inward_value):
            raise SolverFailedError(
                "3D boundary value cannot be reconstructed from an interior neighbor at "
                f"grid index ({i}, {j}, {k})"
            )
        u[k, j, i] = factor * inward_value + offset


def solve_pde_3d(
    residual_func: Callable[..., float] | None,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    nx: int,
    ny: int,
    nz: int,
    *,
    parameters: dict[str, float] | None = None,
    coefficient_provider: PDECoefficientProvider3D | None = None,
    boundary_conditions: PDEBoundaryConditions3D | None = None,
) -> PDESolution3D:
    """Solve a real scalar linear second-order elliptic PDE in three dimensions.

    Exactly one coefficient path is required. A residual receives
    ``(x, y, z, f, fx, fy, fz, fxx, fxy, fxz, fyy, fyz, fzz, **parameters)``
    and is checked for affinity at every unknown point. The direct provider
    returns the eleven coefficients documented by :class:`PDECoefficients3D`.

    Only rectangular domains are supported in this release. The solution has
    shape ``(nz, ny, nx)``. Periodic axes omit the duplicated upper endpoint.
    """
    if (residual_func is None) == (coefficient_provider is None):
        raise SolverFailedError(
            "Provide exactly one of residual_func or coefficient_provider for a 3D PDE"
        )
    (
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        nx,
        ny,
        nz,
    ) = _validate_grid_inputs_3d(
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        nx,
        ny,
        nz,
    )
    params = normalize_params(parameters)
    conditions = boundary_conditions or PDEBoundaryConditions3D()
    periodic_x = conditions.periodic_x
    periodic_y = conditions.periodic_y
    periodic_z = conditions.periodic_z
    x = np.linspace(x_min, x_max, nx, endpoint=not periodic_x)
    y = np.linspace(y_min, y_max, ny, endpoint=not periodic_y)
    z = np.linspace(z_min, z_max, nz, endpoint=not periodic_z)
    hx = (x_max - x_min) / (nx if periodic_x else nx - 1)
    hy = (y_max - y_min) / (ny if periodic_y else ny - 1)
    hz = (z_max - z_min) / (nz if periodic_z else nz - 1)
    boundary = _prepare_boundary_3d(
        conditions,
        x=x,
        y=y,
        z=z,
        hx=hx,
        hy=hy,
        hz=hz,
    )

    def coefficient_at(xi: float, yj: float, zk: float) -> tuple[PDECoefficients3D, int]:
        """Evaluate and validate one 3D coefficient tuple."""
        try:
            if coefficient_provider is not None:
                raw = coefficient_provider(xi, yj, zk, params)
            else:
                assert residual_func is not None
                raw = _probe_coefficients_3d(residual_func, xi, yj, zk, params)
            return _validate_coefficients_3d(raw, xi, yj, zk)
        except SolverFailedError:
            raise
        except Exception as exc:
            logger.error("3D PDE coefficient probe failed at (%g, %g, %g): %s", xi, yj, zk, exc)
            raise SolverFailedError(
                f"3D PDE coefficient probe failed at ({xi:.12g}, {yj:.12g}, {zk:.12g}): {exc}"
            ) from exc

    n_unknown = int(np.count_nonzero(boundary.unknown))
    assembled = _assemble_pde_3d(
        x,
        y,
        z,
        hx=hx,
        hy=hy,
        hz=hz,
        boundary=boundary,
        coefficient_at=coefficient_at,
    )
    solution, diagnostics = solve_sparse_pde(assembled, warnings=())
    u = np.full((nz, ny, nx), np.nan, dtype=float)
    point_indices = boundary.index_grid[boundary.unknown]
    u[boundary.unknown] = solution[point_indices]
    _apply_boundary_values_3d(u, boundary)
    logger.info("PDE 3D solved: %dx%dx%d grid, %d unknown points", nx, ny, nz, n_unknown)
    return PDESolution3D(
        grid=(x, y, z),
        u=u,
        success=True,
        message="OK",
        diagnostics=diagnostics,
    )


__all__ = [
    "PDEBoundaryCondition3D",
    "PDEBoundaryConditions3D",
    "PDECoefficientProvider3D",
    "PDECoefficients3D",
    "PDEDiagnostics",
    "PDESolution3D",
    "solve_pde_3d",
]
