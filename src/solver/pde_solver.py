"""PDE solver for multivariate scalar fields using finite differences.

Supports 2D elliptic PDEs of the general form:

    a(x,y)*f_xx + b(x,y)*f_xy + c(x,y)*f_yy + d(x,y)*f_x + e(x,y)*f_y + g(x,y)*f = rhs(x,y)

The solver probes the user-supplied residual function at each grid point to
extract local coefficients, then assembles a sparse linear system using central
differences (5-point stencil for the Laplacian, central differences for first
derivatives and mixed derivative).

Supports:
- Arbitrary domain shapes via boolean masks
- Dirichlet and Neumann boundary conditions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from utils import SolverFailedError, get_logger, normalize_params

logger = get_logger(__name__)

# Boundary condition type constants
BC_DIRICHLET = "dirichlet"
BC_NEUMANN = "neumann"


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
    """

    grid: tuple[np.ndarray, ...]
    u: np.ndarray
    success: bool
    message: str
    n_eval: int = 0
    mask: np.ndarray | None = None


def _classify_mask(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    """Classify grid points as interior, boundary, or exterior.

    A point (i, j) where mask[j, i] is True is *boundary* if any of its 4
    direct neighbors is False or lies outside the grid. Otherwise it is
    *interior*.

    Args:
        mask: Boolean array of shape (ny, nx).

    Returns:
        (interior_mask, boundary_mask, index_map)
        - interior_mask: bool (ny, nx)
        - boundary_mask: bool (ny, nx)
        - index_map: {(i, j): flat_index} for interior points only
    """
    ny, nx = mask.shape
    interior = np.zeros_like(mask)
    boundary = np.zeros_like(mask)

    for j in range(ny):
        for i in range(nx):
            if not mask[j, i]:
                continue
            # Check if all 4 neighbors are inside the mask
            is_interior = True
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny or not mask[nj, ni]:
                    is_interior = False
                    break
            if is_interior:
                interior[j, i] = True
            else:
                boundary[j, i] = True

    # Build index map for interior points (row-major order)
    index_map: dict[tuple[int, int], int] = {}
    idx = 0
    for j in range(ny):
        for i in range(nx):
            if interior[j, i]:
                index_map[(i, j)] = idx
                idx += 1

    return interior, boundary, index_map


def _probe_coefficients(
    residual_func: Callable[..., float],
    xi: float,
    yj: float,
    params: dict,
) -> tuple[float, float, float, float, float, float, float]:
    """Probe the residual function to extract linear PDE coefficients.

    Given residual R(x,y,f,fx,fy,fxx,fxy,fyy) which should equal zero,
    we assume R is affine in (f, fx, fy, fxx, fxy, fyy):
        R = a*fxx + b*fxy + c*fyy + d*fx + e*fy + g*f + rhs_const
    We extract a,b,c,d,e,g by probing with unit perturbations.

    Returns:
        (a_fxx, b_fxy, c_fyy, d_fx, e_fy, g_f, rhs_const)
    """
    # R(all zeros) gives the constant part (negated RHS)
    r0 = residual_func(xi, yj, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, **params)

    # Probe each derivative direction
    g_f = residual_func(xi, yj, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, **params) - r0
    d_fx = residual_func(xi, yj, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, **params) - r0
    e_fy = residual_func(xi, yj, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, **params) - r0
    a_fxx = residual_func(xi, yj, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, **params) - r0
    b_fxy = residual_func(xi, yj, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, **params) - r0
    c_fyy = residual_func(xi, yj, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, **params) - r0

    return (a_fxx, b_fxy, c_fyy, d_fx, e_fy, g_f, r0)


def _default_rectangular_mask(nx: int, ny: int) -> np.ndarray:
    """Create a mask where every point is inside the domain."""
    return np.ones((ny, nx), dtype=bool)


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
        bc_neumann_value: Float array (ny, nx) with normal derivative values
            for Neumann boundary points. Default: zeros.

    Returns:
        PDESolution with grid, solution array, and mask.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    params = normalize_params(parameters)

    if nx < 3 or ny < 3:
        raise SolverFailedError("Grid must have at least 3 points per dimension")

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    hx = (x_max - x_min) / (nx - 1) if nx > 1 else 1.0
    hy = (y_max - y_min) / (ny - 1) if ny > 1 else 1.0

    # Build mask
    if mask is None:
        mask = _default_rectangular_mask(nx, ny)

    # Default BC arrays
    if bc_values is None:
        bc_values = np.zeros((ny, nx))
    if bc_neumann_value is None:
        bc_neumann_value = np.zeros((ny, nx))

    # Classify points
    interior_mask, boundary_mask, index_map = _classify_mask(mask)
    n_interior = len(index_map)

    if n_interior <= 0:
        u = np.full((ny, nx), np.nan)
        u[mask] = 0.0
        if bc_values is not None:
            u[boundary_mask] = bc_values[boundary_mask]
        return PDESolution(
            grid=(x, y),
            u=u,
            success=True,
            message="No interior points",
            n_eval=0,
            mask=mask,
        )

    # Finite difference weights
    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)
    inv_2hx = 1.0 / (2.0 * hx)
    inv_2hy = 1.0 / (2.0 * hy)
    inv_4hxhy = 1.0 / (4.0 * hx * hy)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b_vec = np.zeros(n_interior)

    def _is_neumann(ni: int, nj: int) -> bool:
        if bc_type is None:
            return False
        return bc_type[nj, ni] == BC_NEUMANN

    for (i, j), k in index_map.items():
        try:
            a_c, bxy, c_c, d_c, e_c, g_c, r0 = _probe_coefficients(
                residual_func,
                x[i],
                y[j],
                params,
            )
        except Exception as exc:
            logger.error("PDE coefficient probe failed at (%g, %g): %s", x[i], y[j], exc)
            raise SolverFailedError(f"Coefficient probe failed: {exc}") from exc

        # RHS: -r0 (since R = operator(f) + r0 = 0  =>  operator(f) = -r0)
        b_vec[k] = -r0

        # Central point coefficient
        center_coeff = -2.0 * a_c * inv_hx2 - 2.0 * c_c * inv_hy2 + g_c
        rows.append(k)
        cols.append(k)
        data.append(center_coeff)

        def _add_neighbor(ni: int, nj: int, coeff: float) -> None:
            """Add matrix entry for neighbor or move to RHS for boundary."""
            if (ni, nj) in index_map:
                # Interior neighbor → matrix entry
                rows.append(k)
                cols.append(index_map[(ni, nj)])
                data.append(coeff)
            elif 0 <= ni < nx and 0 <= nj < ny and boundary_mask[nj, ni]:
                # Boundary neighbor
                if _is_neumann(ni, nj):
                    # Neumann BC: ghost-point substitution
                    # The outward normal direction from interior (i,j) to boundary
                    # (ni,nj). We use: u_boundary = u_interior + h * g_N
                    # where h is the step size and g_N is the prescribed derivative.
                    di, dj = ni - i, nj - j
                    if di != 0:
                        h_step = hx * di  # signed step
                    else:
                        h_step = hy * dj
                    ghost_val = h_step * bc_neumann_value[nj, ni]
                    # u[nj,ni] = u[j,i] + ghost_val
                    # Substitute into stencil: coeff * u[nj,ni] = coeff * (u[j,i] + ghost_val)
                    # Add coeff to center and move coeff*ghost_val to RHS
                    rows.append(k)
                    cols.append(k)
                    data.append(coeff)  # adds to center coefficient
                    b_vec[k] -= coeff * ghost_val
                else:
                    # Dirichlet BC: move known value to RHS
                    b_vec[k] -= coeff * bc_values[nj, ni]
            # else: exterior point — contributes nothing (zero flux assumption)

        # f_xx stencil + f_x contribution
        _add_neighbor(i - 1, j, a_c * inv_hx2 - d_c * inv_2hx)
        _add_neighbor(i + 1, j, a_c * inv_hx2 + d_c * inv_2hx)

        # f_yy stencil + f_y contribution
        _add_neighbor(i, j - 1, c_c * inv_hy2 - e_c * inv_2hy)
        _add_neighbor(i, j + 1, c_c * inv_hy2 + e_c * inv_2hy)

        # f_xy stencil (cross derivative)
        if abs(bxy) > 1e-15:
            _add_neighbor(i + 1, j + 1, bxy * inv_4hxhy)
            _add_neighbor(i - 1, j + 1, -bxy * inv_4hxhy)
            _add_neighbor(i + 1, j - 1, -bxy * inv_4hxhy)
            _add_neighbor(i - 1, j - 1, bxy * inv_4hxhy)

    A = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(n_interior, n_interior),
    ).tocsr()

    try:
        u_flat = spsolve(A, b_vec)
    except Exception as exc:
        logger.error("PDE linear solver failed: %s", exc, exc_info=True)
        raise SolverFailedError(f"Linear solver failed: {exc}") from exc

    # Build solution: NaN outside domain, BC on boundary, solved values inside
    u = np.full((ny, nx), np.nan)
    for (i, j), k in index_map.items():
        u[j, i] = u_flat[k]
    # Set boundary values
    for j in range(ny):
        for i in range(nx):
            if boundary_mask[j, i]:
                if _is_neumann(i, j):
                    # For Neumann boundary points, approximate value from
                    # nearest interior neighbor + h * g_N
                    u[j, i] = _estimate_neumann_boundary_value(
                        i,
                        j,
                        u,
                        index_map,
                        interior_mask,
                        hx,
                        hy,
                        bc_neumann_value[j, i],
                    )
                else:
                    u[j, i] = bc_values[j, i]

    logger.info("PDE 2D solved: %dx%d grid, %d interior points", nx, ny, n_interior)
    return PDESolution(
        grid=(x, y),
        u=u,
        success=True,
        message="OK",
        n_eval=0,
        mask=mask,
    )


def _estimate_neumann_boundary_value(
    bi: int,
    bj: int,
    u: np.ndarray,
    index_map: dict[tuple[int, int], int],
    interior_mask: np.ndarray,
    hx: float,
    hy: float,
    neumann_val: float,
) -> float:
    """Estimate the solution value at a Neumann boundary point.

    Uses the nearest interior neighbor's value plus h * g_N where g_N is the
    prescribed normal derivative.
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
