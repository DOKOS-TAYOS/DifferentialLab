"""PDE solver for multivariate scalar fields using finite differences.

Supports 2D elliptic PDEs of the general form:

    a(x,y)*f_xx + b(x,y)*f_xy + c(x,y)*f_yy + d(x,y)*f_x + e(x,y)*f_y + g(x,y)*f = rhs(x,y)

The solver probes the user-supplied residual function at each grid point to
extract local coefficients, then assembles a sparse linear system using central
differences (5-point stencil for the Laplacian, central differences for first
derivatives and mixed derivative).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from utils import SolverFailedError, get_logger, normalize_params

logger = get_logger(__name__)


@dataclass
class PDESolution:
    """Container for PDE solution data.

    Attributes:
        grid: Tuple of 1D arrays, one per variable (e.g. (x_vals, y_vals)).
        u: Solution array. For 2D: shape (ny, nx).
        success: Whether the solver converged.
        message: Solver status message.
        n_eval: Number of iterations (if applicable).
    """

    grid: tuple[np.ndarray, ...]
    u: np.ndarray
    success: bool
    message: str
    n_eval: int = 0


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
) -> PDESolution:
    """Solve a general 2D linear elliptic PDE using finite differences.

    The residual_func is called as
    ``residual_func(x, y, f, f_x, f_y, f_xx, f_xy, f_yy, **params)``
    and should return the value that must be zero at the solution.

    For example, for the equation ``-f_xx - f_yy = sin(pi*x)*sin(pi*y)``,
    the residual is ``-f_xx - f_yy - sin(pi*x)*sin(pi*y)``.

    The solver probes the residual function at each grid point to extract
    local coefficients for the general operator
    ``a*f_xx + b*f_xy + c*f_yy + d*f_x + e*f_y + g*f = rhs``,
    then assembles a sparse linear system using central differences.

    Args:
        residual_func: Callable returning residual at (x,y) with derivatives.
        x_min: Domain x start.
        x_max: Domain x end.
        y_min: Domain y start.
        y_max: Domain y end.
        nx: Number of x grid points.
        ny: Number of y grid points.
        bc_values: Optional 2D array of boundary values (ny, nx). If None, zero BC.
        parameters: Optional dict of parameters for residual_func.

    Returns:
        PDESolution with grid and solution array.
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

    u = np.zeros((ny, nx))
    if bc_values is not None:
        u[:] = bc_values

    n_interior = (nx - 2) * (ny - 2)
    if n_interior <= 0:
        return PDESolution(
            grid=(x, y),
            u=u,
            success=True,
            message="No interior points",
            n_eval=0,
        )

    # Finite difference weights
    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)
    inv_2hx = 1.0 / (2.0 * hx)
    inv_2hy = 1.0 / (2.0 * hy)
    inv_4hxhy = 1.0 / (4.0 * hx * hy)

    def k_idx(i: int, j: int) -> int:
        return (j - 1) * (nx - 2) + (i - 1)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b = np.zeros(n_interior)

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            k = k_idx(i, j)

            try:
                a, bxy, c, d, e, g, r0 = _probe_coefficients(residual_func, x[i], y[j], params)
            except Exception as exc:
                logger.error("PDE coefficient probe failed at (%g, %g): %s", x[i], y[j], exc)
                raise SolverFailedError(f"Coefficient probe failed: {exc}") from exc

            # RHS: -r0 (since R = operator(f) + r0 = 0  =>  operator(f) = -r0)
            b[k] = -r0

            # Central point coefficient:
            #   a*(-2/hx²) + c*(-2/hy²) + g  for the center
            center_coeff = -2.0 * a * inv_hx2 - 2.0 * c * inv_hy2 + g
            rows.append(k)
            cols.append(k)
            data.append(center_coeff)

            # Helper: add matrix entry or move to RHS if neighbor is on boundary
            def _add(ni: int, nj: int, coeff: float) -> None:
                if 1 <= ni <= nx - 2 and 1 <= nj <= ny - 2:
                    rows.append(k)
                    cols.append(k_idx(ni, nj))
                    data.append(coeff)
                else:
                    # Boundary value: move to RHS
                    b[k] -= coeff * u[nj, ni]

            # f_xx stencil: (f[i-1] - 2f[i] + f[i+1]) / hx²
            # Left neighbor (i-1,j)
            _add(i - 1, j, a * inv_hx2 - d * inv_2hx)  # fxx + fx contribution
            # Right neighbor (i+1,j)
            _add(i + 1, j, a * inv_hx2 + d * inv_2hx)

            # f_yy stencil: (f[j-1] - 2f[j] + f[j+1]) / hy²
            # Bottom neighbor (i,j-1)
            _add(i, j - 1, c * inv_hy2 - e * inv_2hy)  # fyy + fy contribution
            # Top neighbor (i,j+1)
            _add(i, j + 1, c * inv_hy2 + e * inv_2hy)

            # f_xy stencil: (f[i+1,j+1] - f[i-1,j+1] - f[i+1,j-1] + f[i-1,j-1]) / (4*hx*hy)
            if abs(bxy) > 1e-15:
                _add(i + 1, j + 1, bxy * inv_4hxhy)
                _add(i - 1, j + 1, -bxy * inv_4hxhy)
                _add(i + 1, j - 1, -bxy * inv_4hxhy)
                _add(i - 1, j - 1, bxy * inv_4hxhy)

    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_interior, n_interior)).tocsr()

    try:
        u_flat = spsolve(A, b)
    except Exception as exc:
        logger.error("PDE linear solver failed: %s", exc, exc_info=True)
        raise SolverFailedError(f"Linear solver failed: {exc}") from exc

    # Fill interior
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            u[j, i] = u_flat[k_idx(i, j)]

    logger.info("PDE 2D solved: %dx%d grid", nx, ny)
    return PDESolution(
        grid=(x, y),
        u=u,
        success=True,
        message="OK",
        n_eval=0,
    )
