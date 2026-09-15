"""PDE solver for multivariate scalar fields using finite differences.

Supports 2D elliptic PDEs of the general form:

    a(x,y)*f_xx + b(x,y)*f_xy + c(x,y)*f_yy + d(x,y)*f_x + e(x,y)*f_y + g(x,y)*f = rhs(x,y)

The solver probes the user-supplied residual function at each grid point to
extract local coefficients, then assembles a sparse linear system using central
differences (5-point stencil for the Laplacian, central differences for first
derivatives and mixed derivative).

Supports:
- Arbitrary domain shapes via boolean masks
- Dirichlet and outward-normal Neumann boundary conditions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np

from utils import SolverFailedError, get_logger, normalize_params

logger = get_logger(__name__)

# Boundary condition type constants
BC_DIRICHLET = "dirichlet"
BC_NEUMANN = "neumann"

PDECoefficients = tuple[float, float, float, float, float, float, float]
PDECoefficientProvider = Callable[[float, float, dict[str, float]], PDECoefficients]

_AFFINITY_PROBE = (0.37, -0.61, 1.19, -0.83, 0.47, 1.31)
_AFFINITY_TOLERANCE = 1.0e-9
_LINEAR_RESIDUAL_TOLERANCE = 1.0e-8
_MAX_CONDITION_ESTIMATE_SIZE = 256


@dataclass(frozen=True)
class PDEDiagnostics:
    """Numerical evidence for an assembled scalar PDE solve.

    The residual values describe the interior sparse system ``A @ u = b``.
    ``condition_estimate`` is populated only for systems small enough to
    densify within the fixed diagnostic bound.
    """

    discrete_residual_l2: float
    discrete_residual_linf: float
    relative_residual_l2: float
    matrix_shape: tuple[int, int]
    nnz: int
    condition_estimate: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass
class PDESolution:
    """Container for PDE solution data.

    Attributes:
        grid: Tuple of 1D arrays, one per variable (e.g. (x_vals, y_vals)).
        u: Solution array. For 2D: shape (ny, nx). Exterior points are NaN.
        success: Whether the solver converged.
        message: Solver status message.
        n_eval: Number of iterations (if applicable).
        mask: Optional boolean mask (ny, nx). True = inside domain.
        diagnostics: Optional structured algebraic diagnostics.
    """

    grid: tuple[np.ndarray, ...]
    u: np.ndarray
    success: bool
    message: str
    n_eval: int = 0
    mask: np.ndarray | None = None
    diagnostics: PDEDiagnostics | None = None


def _classify_mask(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify grid points as interior, boundary, or exterior.

    A point (i, j) where mask[j, i] is True is *boundary* if any of its 4
    direct neighbors is False or lies outside the grid. Otherwise it is
    *interior*.

    Args:
        mask: Boolean array of shape (ny, nx).

    Returns:
        (interior_mask, boundary_mask, index_grid)
        - interior_mask: bool (ny, nx)
        - boundary_mask: bool (ny, nx)
        - index_grid: int (ny, nx), -1 outside the interior and a compact
          row-major index on interior points
    """
    mask_bool = np.asarray(mask, dtype=bool)
    padded = np.pad(mask_bool, pad_width=1, mode="constant", constant_values=False)
    north = padded[:-2, 1:-1]
    south = padded[2:, 1:-1]
    west = padded[1:-1, :-2]
    east = padded[1:-1, 2:]

    interior = mask_bool & north & south & west & east
    boundary = mask_bool & ~interior

    index_grid = np.full(mask_bool.shape, -1, dtype=np.int64)
    index_grid[interior] = np.arange(int(np.count_nonzero(interior)), dtype=np.int64)
    return interior, boundary, index_grid


def _probe_coefficients(
    residual_func: Callable[..., float],
    xi: float,
    yj: float,
    params: dict[str, float],
) -> PDECoefficients:
    """Probe the residual function to extract linear PDE coefficients.

    Given residual R(x,y,f,fx,fy,fxx,fxy,fyy) which should equal zero,
    we assume R is affine in (f, fx, fy, fxx, fxy, fyy):
        R = a*fxx + b*fxy + c*fyy + d*fx + e*fy + g*f + rhs_const
    We extract a,b,c,d,e,g by probing with unit perturbations.

    Args:
        residual_func: Callable returning residual at (x,y) with derivatives.
        xi: x-coordinate of the grid point.
        yj: y-coordinate of the grid point.
        params: Parameter dict passed to residual_func.

    Returns:
        Tuple of (a_fxx, b_fxy, c_fyy, d_fx, e_fy, g_f, rhs_const).
    """
    coordinate = f"({xi:.12g}, {yj:.12g})"

    def evaluate(state: tuple[float, float, float, float, float, float]) -> float:
        raw_value = residual_func(xi, yj, *state, **params)
        return _finite_scalar(raw_value, f"Residual at {coordinate}")

    # R(all zeros) gives the constant part (negated RHS).
    zero = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    r0 = evaluate(zero)

    # Probe each derivative direction
    unit_states = (
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    unit_responses: list[float] = []
    for state in unit_states:
        unit_responses.append(evaluate(state) - r0)

    g_f, d_fx, e_fy, a_fxx, b_fxy, c_fyy = unit_responses
    reconstructed = r0 + sum(
        coefficient * value for coefficient, value in zip(unit_responses, _AFFINITY_PROBE)
    )
    actual = evaluate(_AFFINITY_PROBE)
    scale = max(
        1.0,
        abs(actual),
        abs(reconstructed),
        abs(r0)
        + sum(
            abs(coefficient * value) for coefficient, value in zip(unit_responses, _AFFINITY_PROBE)
        ),
    )
    if not np.isfinite(reconstructed) or abs(actual - reconstructed) > _AFFINITY_TOLERANCE * scale:
        raise SolverFailedError(
            "PDE residual is not affine in (f, fx, fy, fxx, fxy, fyy) "
            f"at {coordinate}: probe mismatch {abs(actual - reconstructed):.3g}"
        )

    return (a_fxx, b_fxy, c_fyy, d_fx, e_fy, g_f, r0)


def _finite_scalar(value: object, name: str) -> float:
    """Convert a scalar-like value to a finite real float."""
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array):
        raise SolverFailedError(f"{name} must be a finite real scalar")
    try:
        result = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverFailedError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise SolverFailedError(f"{name} must be finite")
    return result


def _validate_grid_inputs(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
) -> tuple[float, float, float, float, int, int]:
    """Validate and normalize rectangular grid inputs."""
    bounds = tuple(
        _finite_scalar(value, name)
        for value, name in (
            (x_min, "x_min"),
            (x_max, "x_max"),
            (y_min, "y_min"),
            (y_max, "y_max"),
        )
    )
    x_start, x_end, y_start, y_end = bounds
    if x_start >= x_end:
        raise SolverFailedError("x_min must be strictly less than x_max")
    if y_start >= y_end:
        raise SolverFailedError("y_min must be strictly less than y_max")

    for value, name in ((nx, "nx"), (ny, "ny")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise SolverFailedError(f"{name} must be an integer")
        if int(value) < 3:
            raise SolverFailedError("Grid must have at least 3 points per dimension")
    return x_start, x_end, y_start, y_end, int(nx), int(ny)


def _float_array(value: object, name: str, shape: tuple[int, int]) -> np.ndarray:
    """Return a float array after validating its exact shape."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverFailedError(f"{name} must contain real numeric values") from exc
    if array.shape != shape:
        raise SolverFailedError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _validate_boundary_inputs(
    *,
    nx: int,
    ny: int,
    mask: np.ndarray | None,
    bc_values: np.ndarray | None,
    bc_type: np.ndarray | None,
    bc_neumann_value: np.ndarray | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Validate masks and boundary arrays and classify the domain."""
    shape = (ny, nx)
    if mask is None:
        mask_array = _default_rectangular_mask(nx, ny)
    else:
        raw_mask = np.asarray(mask)
        if raw_mask.shape != shape:
            raise SolverFailedError(f"mask must have shape {shape}, got {raw_mask.shape}")
        if np.issubdtype(raw_mask.dtype, np.number) and not np.all(np.isfinite(raw_mask)):
            raise SolverFailedError("mask must not contain non-finite values")
        mask_array = np.asarray(raw_mask, dtype=bool)

    values = (
        np.zeros(shape, dtype=float)
        if bc_values is None
        else _float_array(bc_values, "bc_values", shape)
    )
    neumann_values = (
        np.zeros(shape, dtype=float)
        if bc_neumann_value is None
        else _float_array(bc_neumann_value, "bc_neumann_value", shape)
    )
    if bc_type is None:
        types = np.full(shape, BC_DIRICHLET, dtype=object)
    else:
        types = np.asarray(bc_type, dtype=object)
        if types.shape != shape:
            raise SolverFailedError(f"bc_type must have shape {shape}, got {types.shape}")

    interior, boundary, index_grid = _classify_mask(mask_array)
    boundary_types = types[boundary]
    invalid = [label for label in boundary_types if label not in (BC_DIRICHLET, BC_NEUMANN)]
    if invalid:
        raise SolverFailedError(
            f"bc_type boundary labels must be 'dirichlet' or 'neumann'; got {invalid[0]!r}"
        )

    neumann = boundary & (types == BC_NEUMANN)
    dirichlet = boundary & ~neumann
    if not np.all(np.isfinite(values[dirichlet])):
        raise SolverFailedError("bc_values must be finite at Dirichlet boundary points")
    if not np.all(np.isfinite(neumann_values[neumann])):
        raise SolverFailedError("bc_neumann_value must be finite at Neumann boundary points")
    return mask_array, values, types, neumann_values, interior, boundary, index_grid


def _validate_coefficients(
    coefficients: object,
    xi: float,
    yj: float,
) -> PDECoefficients:
    """Validate coefficient shape, finiteness, and scalar ellipticity."""
    coordinate = f"({xi:.12g}, {yj:.12g})"
    try:
        array = np.asarray(coefficients, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverFailedError(
            f"PDE coefficients at {coordinate} must be seven real scalars"
        ) from exc
    if array.shape != (7,):
        raise SolverFailedError(
            f"PDE coefficients at {coordinate} must have shape (7,), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise SolverFailedError(f"PDE coefficients at {coordinate} must be finite")

    a_c, bxy, c_c = (float(value) for value in array[:3])
    principal = np.array(((a_c, 0.5 * bxy), (0.5 * bxy, c_c)))
    principal_scale = float(np.max(np.abs(principal)))
    if principal_scale <= np.finfo(float).tiny:
        raise SolverFailedError(f"PDE principal operator is degenerate at {coordinate}")
    eigenvalues = np.linalg.eigvalsh(principal)
    margin = 100.0 * np.finfo(float).eps * principal_scale
    positive = bool(np.all(eigenvalues > margin))
    negative = bool(np.all(eigenvalues < -margin))
    if not (positive or negative):
        raise SolverFailedError(
            "PDE principal operator is not strictly elliptic "
            f"at {coordinate}: eigenvalues={eigenvalues.tolist()}"
        )
    return cast(PDECoefficients, tuple(float(value) for value in array))


def _compute_diagnostics(
    matrix: Any,
    rhs: np.ndarray,
    solution: np.ndarray,
    diagnostic_warnings: tuple[str, ...],
) -> PDEDiagnostics:
    """Compute bounded diagnostics for the solved sparse linear system."""
    applied = np.asarray(matrix @ solution, dtype=float)
    residual = applied - rhs
    residual_l2 = float(np.linalg.norm(residual))
    residual_linf = float(np.linalg.norm(residual, ord=np.inf)) if residual.size else 0.0
    rhs_scale = float(np.linalg.norm(rhs))
    applied_scale = float(np.linalg.norm(applied))
    relative_l2 = residual_l2 / max(rhs_scale, applied_scale, np.finfo(float).tiny)

    condition_estimate: float | None = None
    if matrix.shape[0] <= _MAX_CONDITION_ESTIMATE_SIZE:
        condition_estimate = float(np.linalg.cond(matrix.toarray()))
        rank_limit = 1.0 / (max(1, matrix.shape[0]) * np.finfo(float).eps)
        if not np.isfinite(condition_estimate) or condition_estimate >= rank_limit:
            raise SolverFailedError("Linear system is singular or numerically rank deficient")

    residual_scale = max(1.0, rhs_scale, applied_scale)
    if residual_l2 > _LINEAR_RESIDUAL_TOLERANCE * residual_scale:
        raise SolverFailedError(
            "Linear solve produced an excessive algebraic residual: "
            f"relative L2 residual={relative_l2:.3g}"
        )

    return PDEDiagnostics(
        discrete_residual_l2=residual_l2,
        discrete_residual_linf=residual_linf,
        relative_residual_l2=relative_l2,
        matrix_shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        nnz=int(matrix.nnz),
        condition_estimate=condition_estimate,
        warnings=diagnostic_warnings,
    )


def _default_rectangular_mask(nx: int, ny: int) -> np.ndarray:
    """Create a mask where every point is inside the domain.

    Args:
        nx: Number of x grid points.
        ny: Number of y grid points.

    Returns:
        Boolean array (ny, nx) with all True.
    """
    return np.ones((ny, nx), dtype=bool)


def _neumann_boundary_substitution(
    boundary_i: int,
    boundary_j: int,
    source_i: int,
    source_j: int,
    nx: int,
    ny: int,
    hx: float,
    hy: float,
    interior_mask: np.ndarray,
    normal_derivative: float,
    *,
    mixed_neighbor: bool,
) -> tuple[int, int, float]:
    """Express a Neumann boundary value using an inward grid neighbor.

    For the outward-normal convention ``du/dn = q``, the one-sided relation is
    ``u_boundary = u_inward + h*q`` on every rectangular edge. A mixed-derivative
    diagonal must therefore map to the boundary point's inward neighbor rather
    than to the stencil center. Masked diagonals and Neumann corners do not have
    a unique rectangular grid normal in the current boundary representation.
    """
    inward_candidates: list[tuple[int, int, float]] = []
    if boundary_i == 0:
        inward_candidates.append((boundary_i + 1, boundary_j, hx))
    if boundary_i == nx - 1:
        inward_candidates.append((boundary_i - 1, boundary_j, hx))
    if boundary_j == 0:
        inward_candidates.append((boundary_i, boundary_j + 1, hy))
    if boundary_j == ny - 1:
        inward_candidates.append((boundary_i, boundary_j - 1, hy))

    if len(inward_candidates) == 1:
        inward_i, inward_j, step = inward_candidates[0]
        if interior_mask[inward_j, inward_i]:
            return inward_i, inward_j, step * normal_derivative

    if mixed_neighbor:
        raise SolverFailedError(
            "A mixed-derivative stencil adjacent to a Neumann boundary requires "
            "a unique rectangular grid normal; masked-boundary and corner "
            f"diagonals are not supported at grid index ({boundary_i}, {boundary_j})"
        )

    delta_i = boundary_i - source_i
    delta_j = boundary_j - source_j
    if abs(delta_i) + abs(delta_j) != 1:
        raise SolverFailedError("Neumann boundary elimination requires an axial interior neighbor")
    step = hx if delta_i != 0 else hy
    return source_i, source_j, step * normal_derivative


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
) -> PDESolution:
    """Solve a general 2D linear elliptic PDE using finite differences.

    The residual_func is called as
    ``residual_func(x, y, f, f_x, f_y, f_xx, f_xy, f_yy, **params)``
    and should return the value that must be zero at the solution.

    For example, for the equation ``-f_xx - f_yy = sin(pi*x)*sin(pi*y)``,
    the residual is ``-f_xx - f_yy - sin(pi*x)*sin(pi*y)``.

    Args:
        residual_func: Callable returning residual at (x,y) with derivatives.
        x_min: Domain x start.
        x_max: Domain x end.
        y_min: Domain y start.
        y_max: Domain y end.
        nx: Number of x grid points.
        ny: Number of y grid points.
        bc_values: Dirichlet boundary values, shape (ny, nx). Default: zeros.
        parameters: Optional dict of parameters for residual_func.
        mask: Boolean array (ny, nx). True = inside domain. None = full rectangle.
        bc_type: String array (ny, nx) with "dirichlet" or "neumann" per point.
            Only boundary points are used. Default: all Dirichlet.
        bc_neumann_value: Float array (ny, nx) with outward-normal derivative
            values ``du/dn`` for Neumann boundary points. The solver uses a
            one-sided grid-normal relation. Default: zeros.
        coefficient_provider: Optional direct provider for linear residual
            coefficients ``(f_xx, f_xy, f_yy, f_x, f_y, f, constant)``.
            When provided, the solver skips finite-difference probing.

    Returns:
        PDESolution with grid, solution array, and mask.

    Raises:
        SolverFailedError: If inputs are invalid, the residual is not affine,
            the principal operator is not strictly elliptic, or the sparse
            linear solve is singular, non-finite, or has excessive residual.
    """
    from scipy import sparse
    from scipy.sparse.linalg import MatrixRankWarning, spsolve

    params = normalize_params(parameters)

    x_min, x_max, y_min, y_max, nx, ny = _validate_grid_inputs(
        x_min,
        x_max,
        y_min,
        y_max,
        nx,
        ny,
    )

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    hx = (x_max - x_min) / (nx - 1)
    hy = (y_max - y_min) / (ny - 1)

    (
        mask_array,
        bc_value_array,
        _bc_type_array,
        neumann_value_array,
        interior_mask,
        boundary_mask,
        index_grid,
    ) = _validate_boundary_inputs(
        nx=nx,
        ny=ny,
        mask=mask,
        bc_values=bc_values,
        bc_type=bc_type,
        bc_neumann_value=bc_neumann_value,
    )
    neumann_mask = boundary_mask & (_bc_type_array == BC_NEUMANN)
    n_interior = int(np.count_nonzero(interior_mask))
    diagnostic_warnings = (
        (
            (
                "Neumann conditions on masked boundaries use grid-normal, "
                "not geometric-normal, derivatives."
            ),
        )
        if mask is not None and np.any(neumann_mask)
        else ()
    )

    if n_interior <= 0:
        u = np.full((ny, nx), np.nan)
        u[mask_array] = 0.0
        dirichlet_boundary = boundary_mask & ~neumann_mask
        u[dirichlet_boundary] = bc_value_array[dirichlet_boundary]
        return PDESolution(
            grid=(x, y),
            u=u,
            success=True,
            message="No interior points",
            n_eval=0,
            mask=mask_array,
            diagnostics=PDEDiagnostics(
                discrete_residual_l2=0.0,
                discrete_residual_linf=0.0,
                relative_residual_l2=0.0,
                matrix_shape=(0, 0),
                nnz=0,
                condition_estimate=None,
                warnings=diagnostic_warnings + ("No interior points were assembled.",),
            ),
        )

    # Finite difference weights
    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)
    inv_2hx = 1.0 / (2.0 * hx)
    inv_2hy = 1.0 / (2.0 * hy)
    inv_4hxhy = 1.0 / (4.0 * hx * hy)

    max_entries = max(1, 9 * n_interior)
    rows = np.empty(max_entries, dtype=np.int64)
    cols = np.empty(max_entries, dtype=np.int64)
    data = np.empty(max_entries, dtype=float)
    b_vec = np.zeros(n_interior)
    entry_count = 0
    principal_orientation: int | None = None
    orientation_coordinate: tuple[float, float] | None = None

    def _append_entry(row: int, col: int, value: float) -> None:
        nonlocal entry_count
        rows[entry_count] = row
        cols[entry_count] = col
        data[entry_count] = value
        entry_count += 1

    for k, (j, i) in enumerate(np.argwhere(interior_mask)):
        try:
            xi = float(x[i])
            yj = float(y[j])
            if coefficient_provider is None:
                raw_coefficients = _probe_coefficients(
                    residual_func,
                    xi,
                    yj,
                    params,
                )
            else:
                raw_coefficients = coefficient_provider(xi, yj, params)
            a_c, bxy, c_c, d_c, e_c, g_c, r0 = _validate_coefficients(
                raw_coefficients,
                xi,
                yj,
            )
            current_orientation = 1 if a_c > 0.0 else -1
            if principal_orientation is None:
                principal_orientation = current_orientation
                orientation_coordinate = (xi, yj)
            elif current_orientation != principal_orientation:
                expected = "positive definite" if principal_orientation > 0 else "negative definite"
                actual = "positive definite" if current_orientation > 0 else "negative definite"
                assert orientation_coordinate is not None
                raise SolverFailedError(
                    "PDE ellipticity orientation changes across the connected domain: "
                    f"expected {expected} from "
                    f"({orientation_coordinate[0]:.12g}, {orientation_coordinate[1]:.12g}), "
                    f"got {actual} at ({xi:.12g}, {yj:.12g})"
                )
        except SolverFailedError:
            raise
        except Exception as exc:
            logger.error("PDE coefficient probe failed at (%g, %g): %s", x[i], y[j], exc)
            raise SolverFailedError(
                f"Coefficient probe failed at ({x[i]:.12g}, {y[j]:.12g}): {exc}"
            ) from exc

        # RHS: -r0 (since R = operator(f) + r0 = 0  =>  operator(f) = -r0)
        b_vec[k] = -r0

        # Central point coefficient
        center_coeff = -2.0 * a_c * inv_hx2 - 2.0 * c_c * inv_hy2 + g_c
        _append_entry(k, k, center_coeff)

        def _add_neighbor(
            ni: int,
            nj: int,
            coeff: float,
            *,
            mixed_neighbor: bool = False,
        ) -> None:
            """Add matrix entry for neighbor or move to RHS for boundary."""
            if not (0 <= ni < nx and 0 <= nj < ny):
                return

            neighbor_idx = int(index_grid[nj, ni])
            if neighbor_idx >= 0:
                _append_entry(k, neighbor_idx, coeff)
                return

            if not boundary_mask[nj, ni]:
                return

            if neumann_mask[nj, ni]:
                inward_i, inward_j, boundary_offset = _neumann_boundary_substitution(
                    ni,
                    nj,
                    int(i),
                    int(j),
                    nx,
                    ny,
                    hx,
                    hy,
                    interior_mask,
                    float(neumann_value_array[nj, ni]),
                    mixed_neighbor=mixed_neighbor,
                )
                inward_idx = int(index_grid[inward_j, inward_i])
                if inward_idx < 0:
                    raise SolverFailedError(
                        "Neumann boundary elimination did not resolve to an interior point"
                    )
                _append_entry(k, inward_idx, coeff)
                b_vec[k] -= coeff * boundary_offset
                return

            b_vec[k] -= coeff * bc_value_array[nj, ni]

        # f_xx stencil + f_x contribution
        _add_neighbor(i - 1, j, a_c * inv_hx2 - d_c * inv_2hx)
        _add_neighbor(i + 1, j, a_c * inv_hx2 + d_c * inv_2hx)

        # f_yy stencil + f_y contribution
        _add_neighbor(i, j - 1, c_c * inv_hy2 - e_c * inv_2hy)
        _add_neighbor(i, j + 1, c_c * inv_hy2 + e_c * inv_2hy)

        # f_xy stencil (cross derivative)
        if abs(bxy) > 1e-15:
            _add_neighbor(i + 1, j + 1, bxy * inv_4hxhy, mixed_neighbor=True)
            _add_neighbor(i - 1, j + 1, -bxy * inv_4hxhy, mixed_neighbor=True)
            _add_neighbor(i + 1, j - 1, -bxy * inv_4hxhy, mixed_neighbor=True)
            _add_neighbor(i - 1, j - 1, bxy * inv_4hxhy, mixed_neighbor=True)

    A = sparse.coo_matrix(
        (data[:entry_count], (rows[:entry_count], cols[:entry_count])),
        shape=(n_interior, n_interior),
    ).tocsr()

    if not np.all(np.isfinite(A.data)):
        raise SolverFailedError("Assembled sparse matrix contains non-finite values")
    if not np.all(np.isfinite(b_vec)):
        raise SolverFailedError("Assembled right-hand side contains non-finite values")

    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            u_flat = np.asarray(spsolve(A, b_vec), dtype=float)
    except Exception as exc:
        logger.error("PDE linear solver failed: %s", exc, exc_info=True)
        raise SolverFailedError(f"Linear solver failed: {exc}") from exc

    if u_flat.shape != (n_interior,) or not np.all(np.isfinite(u_flat)):
        raise SolverFailedError("Linear solver returned a non-finite or invalid solution")
    diagnostics = _compute_diagnostics(A, b_vec, u_flat, diagnostic_warnings)

    # Build solution: NaN outside domain, BC on boundary, solved values inside
    u = np.full((ny, nx), np.nan)
    u[interior_mask] = u_flat[index_grid[interior_mask]]
    dirichlet_boundary = boundary_mask & ~neumann_mask
    u[dirichlet_boundary] = bc_value_array[dirichlet_boundary]

    for bj, bi in np.argwhere(neumann_mask):
        u[bj, bi] = _estimate_neumann_boundary_value(
            int(bi),
            int(bj),
            u,
            interior_mask,
            hx,
            hy,
            float(neumann_value_array[bj, bi]),
        )

    logger.info("PDE 2D solved: %dx%d grid, %d interior points", nx, ny, n_interior)
    return PDESolution(
        grid=(x, y),
        u=u,
        success=True,
        message="OK",
        n_eval=0,
        mask=mask_array,
        diagnostics=diagnostics,
    )


def _estimate_neumann_boundary_value(
    bi: int,
    bj: int,
    u: np.ndarray,
    interior_mask: np.ndarray,
    hx: float,
    hy: float,
    neumann_val: float,
) -> float:
    """Estimate the solution value at a Neumann boundary point.

    Uses the nearest interior neighbor's value plus ``h * g_N``, where ``g_N``
    is the prescribed outward-normal derivative ``du/dn``. The positive grid
    distance applies on all four edges because the step is from interior to
    boundary in the outward direction.

    Args:
        bi: x-index of the boundary point.
        bj: y-index of the boundary point.
        u: Solution array (ny, nx).
        interior_mask: Boolean array marking interior points.
        hx: Grid spacing in x direction.
        hy: Grid spacing in y direction.
        neumann_val: Prescribed normal derivative value.

    Returns:
        Estimated solution value at the Neumann boundary point.
    """
    ny, nx = u.shape
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = bi + di, bj + dj
        if 0 <= ni < nx and 0 <= nj < ny and interior_mask[nj, ni]:
            val = u[nj, ni]
            if not np.isnan(val):
                # Step from interior to boundary
                if di != 0:
                    return val + hx * abs(di) * neumann_val
                else:
                    return val + hy * abs(dj) * neumann_val
    # Fallback: use 0 if no interior neighbor found
    return 0.0
