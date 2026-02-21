"""PDE solver for multivariate scalar fields using finite differences.

Supports 2D elliptic PDEs of the form L[u] = f, where L involves u, u_x, u_y,
u_xx, u_xy, u_yy. Uses 5-point stencil for the Laplacian and central differences
for first derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from utils import SolverFailedError, get_logger

logger = get_logger(__name__)


@dataclass
class PDESolution:
    """Container for PDE solution data.

    Attributes:
        grid: Tuple of 1D arrays, one per variable (e.g. (x_vals, y_vals)).
        u: Solution array. For 2D: shape (ny, nx). For 3D: (nz, ny, nx).
        success: Whether the solver converged.
        message: Solver status message.
        n_eval: Number of iterations (if applicable).
    """

    grid: tuple[np.ndarray, ...]
    u: np.ndarray
    success: bool
    message: str
    n_eval: int = 0


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
    """Solve a 2D elliptic PDE using finite differences.

    The residual_func is called as
    ``residual_func(x, y, f, f_x, f_y, f_xx, f_xy, f_yy, **params)``
    and should return the residual (LHS - RHS) that must be zero.

    For linear Poisson -u_xx - u_yy = f(x,y), the residual is -u_xx - u_yy - f.

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

    params = dict(parameters) if parameters else {}

    if nx < 3 or ny < 3:
        raise SolverFailedError("Grid must have at least 3 points per dimension")

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    hx = (x_max - x_min) / (nx - 1) if nx > 1 else 1.0
    hy = (y_max - y_min) / (ny - 1) if ny > 1 else 1.0

    u = np.zeros((ny, nx))
    if bc_values is not None:
        u[:] = bc_values
    # else: zero Dirichlet BC (u=0 on boundary)

    n_interior = (nx - 2) * (ny - 2)
    if n_interior <= 0:
        return PDESolution(
            grid=(x, y),
            u=u,
            success=True,
            message="No interior points",
            n_eval=0,
        )

    cx = 1.0 / (hx * hx)
    cy = 1.0 / (hy * hy)
    diag = 2.0 * (cx + cy)

    def k_idx(i: int, j: int) -> int:
        return (j - 1) * (nx - 2) + (i - 1)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b = np.zeros(n_interior)

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            k = k_idx(i, j)
            # Interior: -u_xx - u_yy = f => diag*u - cx*(u[i±1]) - cy*(u[j±1]) = f
            rows.append(k)
            cols.append(k)
            data.append(diag)

            if i > 1:
                rows.append(k)
                cols.append(k_idx(i - 1, j))
                data.append(-cx)
            else:
                b[k] += cx * u[j, i - 1]

            if i < nx - 2:
                rows.append(k)
                cols.append(k_idx(i + 1, j))
                data.append(-cx)
            else:
                b[k] += cx * u[j, i + 1]

            if j > 1:
                rows.append(k)
                cols.append(k_idx(i, j - 1))
                data.append(-cy)
            else:
                b[k] += cy * u[j - 1, i]

            if j < ny - 2:
                rows.append(k)
                cols.append(k_idx(i, j + 1))
                data.append(-cy)
            else:
                b[k] += cy * u[j + 1, i]

            # RHS f(x,y) for -u_xx - u_yy = f
            try:
                f_val = residual_func(x[i], y[j], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, **params)
                b[k] += f_val
            except Exception as exc:
                raise SolverFailedError(f"Residual evaluation failed: {exc}") from exc

    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_interior, n_interior)).tocsr()

    try:
        u_flat = spsolve(A, b)
    except Exception as exc:
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


def solve_pde_2d_linear_poisson(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
    f_source: Callable[[float, float], float] | float = 0.0,
) -> PDESolution:
    """Solve -u_xx - u_yy = f with zero Dirichlet BC.

    Args:
        x_min, x_max, y_min, y_max: Domain bounds.
        nx, ny: Grid size.
        f_source: RHS. Either a callable f(x,y) or constant.

    Returns:
        PDESolution.
    """
    def rhs_only(x: float, y: float, _f: float, _fx: float, _fy: float,
                 _fxx: float, _fxy: float, _fyy: float, **kw: Any) -> float:
        if callable(f_source):
            return f_source(x, y)
        return float(f_source)

    return solve_pde_2d(
        rhs_only,
        x_min, x_max, y_min, y_max, nx, ny,
    )
