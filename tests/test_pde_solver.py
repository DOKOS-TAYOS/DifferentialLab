"""Tests for solver.pde_solver."""

from __future__ import annotations

import numpy as np

from solver.pde_solver import BC_DIRICHLET, BC_NEUMANN, PDESolution, solve_pde_2d


def _laplace_residual(
    x: float,
    y: float,
    f: float,
    fx: float,
    fy: float,
    fxx: float,
    fxy: float,
    fyy: float,
    **kw: object,
) -> float:
    """Residual for Laplace equation -f_xx - f_yy = 0."""
    return -fxx - fyy


def test_laplace_zero_bc() -> None:
    """Laplace -f_xx - f_yy = 0 with f=0 on all boundaries -> solution is 0."""
    result = solve_pde_2d(
        _laplace_residual,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        nx=11,
        ny=11,
        bc_values=None,  # Zero BC by default
    )
    assert isinstance(result, PDESolution)
    assert result.success is True
    assert result.grid[0].shape == (11,)
    assert result.grid[1].shape == (11,)
    assert result.u.shape == (11, 11)
    np.testing.assert_allclose(result.u, 0.0, atol=1e-10)


def test_laplace_with_bc_values() -> None:
    """Laplace with Dirichlet BC: f=1 on top edge, f=0 elsewhere."""
    nx, ny = 15, 15
    bc_values = np.zeros((ny, nx))
    bc_values[-1, :] = 1.0  # Top edge f=1

    result = solve_pde_2d(
        _laplace_residual,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        nx=nx,
        ny=ny,
        bc_values=bc_values,
    )
    assert result.success is True
    assert result.u.shape == (ny, nx)
    np.testing.assert_allclose(result.u[-1, :], 1.0)  # Top edge
    np.testing.assert_allclose(result.u[0, :], 0.0)  # Bottom edge
    np.testing.assert_allclose(result.u[0:-1, 0], 0.0)  # Left edge (excl top-left corner)
    np.testing.assert_allclose(result.u[0:-1, -1], 0.0)  # Right edge (excl top-right corner)
    # Interior should be between 0 and 1 (maximum principle for Laplace)
    interior = result.u[1:-1, 1:-1]
    assert np.all(interior >= -1e-6) and np.all(interior <= 1.0 + 1e-6)


def test_poisson_simple_rhs() -> None:
    """Poisson -f_xx - f_yy = 1 with zero BC."""

    def residual(
        x: float,
        y: float,
        f: float,
        fx: float,
        fy: float,
        fxx: float,
        fxy: float,
        fyy: float,
        **kw: object,
    ) -> float:
        return -fxx - fyy - 1.0

    result = solve_pde_2d(
        residual,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        nx=11,
        ny=11,
    )
    assert result.success is True
    # Poisson -f_xx - f_yy = 1 with zero BC: solution is positive (dome-shaped)
    interior = result.u[1:-1, 1:-1]
    assert np.all(interior >= -1e-6)
    assert np.max(interior) > 0.01


# ── Mask-based domain tests ──────────────────────────────────────────────


def test_laplace_circular_mask_zero_bc() -> None:
    """Laplace on circular domain with zero Dirichlet BC -> solution is 0."""
    nx, ny = 21, 21
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    mask = X**2 + Y**2 <= 1.0

    result = solve_pde_2d(
        _laplace_residual,
        x_min=-1.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
        nx=nx,
        ny=ny,
        bc_values=np.zeros((ny, nx)),
        mask=mask,
    )
    assert result.success is True
    assert result.mask is not None
    # Interior (non-NaN, non-boundary) should be ~0
    valid = ~np.isnan(result.u)
    np.testing.assert_allclose(result.u[valid], 0.0, atol=1e-10)


def test_poisson_circular_mask() -> None:
    """Poisson -f_xx - f_yy = 1 on circular domain, zero Dirichlet BC."""
    nx, ny = 21, 21
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    mask = X**2 + Y**2 <= 1.0

    def residual(x, y, f, fx, fy, fxx, fxy, fyy, **kw):
        return -fxx - fyy - 1.0

    result = solve_pde_2d(
        residual,
        x_min=-1.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
        nx=nx,
        ny=ny,
        bc_values=np.zeros((ny, nx)),
        mask=mask,
    )
    assert result.success is True
    # Exterior should be NaN
    assert np.any(np.isnan(result.u))
    # Interior should be positive (dome shape)
    valid = result.u[~np.isnan(result.u)]
    assert np.all(valid >= -1e-6)
    assert np.max(valid) > 0.01


def test_backward_compat_no_mask() -> None:
    """Existing API (no mask) should still work identically."""
    result = solve_pde_2d(
        _laplace_residual,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        nx=11,
        ny=11,
        bc_values=None,
    )
    assert result.success is True
    assert result.u.shape == (11, 11)
    # With mask=None, exterior is NaN for consistency but all points are in domain
    # so there should be no NaN
    assert not np.any(np.isnan(result.u))


# ── Neumann BC tests ─────────────────────────────────────────────────────


def test_neumann_zero_flux_bottom() -> None:
    """Laplace with zero Neumann on bottom, Dirichlet=1 on top, zero on sides.

    Zero flux at bottom means df/dy=0 there, so the solution gradient at
    the bottom row should be approximately zero.
    """
    nx, ny = 15, 15
    bc_values = np.zeros((ny, nx))
    bc_values[-1, :] = 1.0  # Top = Dirichlet 1

    bc_type = np.full((ny, nx), BC_DIRICHLET, dtype=object)
    bc_type[0, :] = BC_NEUMANN  # Bottom = Neumann

    neumann_val = np.zeros((ny, nx))  # zero flux at bottom

    result = solve_pde_2d(
        _laplace_residual,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        nx=nx,
        ny=ny,
        bc_values=bc_values,
        bc_type=bc_type,
        bc_neumann_value=neumann_val,
    )
    assert result.success is True
    # With zero Neumann at bottom, the gradient df/dy ≈ 0 at bottom
    # Check that row 0 ≈ row 1 (finite difference approximation)
    hy = 1.0 / (ny - 1)
    grad_bottom = (result.u[1, 1:-1] - result.u[0, 1:-1]) / hy
    np.testing.assert_allclose(grad_bottom, 0.0, atol=0.15)


def test_mixed_dirichlet_neumann() -> None:
    """Laplace with mixed BCs: Dirichlet on top/bottom, Neumann=0 on sides."""
    nx, ny = 15, 15
    bc_values = np.zeros((ny, nx))
    bc_values[0, :] = 0.0  # Bottom = 0
    bc_values[-1, :] = 1.0  # Top = 1

    bc_type = np.full((ny, nx), BC_DIRICHLET, dtype=object)
    bc_type[:, 0] = BC_NEUMANN  # Left = Neumann
    bc_type[:, -1] = BC_NEUMANN  # Right = Neumann

    neumann_val = np.zeros((ny, nx))  # Zero flux on sides

    result = solve_pde_2d(
        _laplace_residual,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        nx=nx,
        ny=ny,
        bc_values=bc_values,
        bc_type=bc_type,
        bc_neumann_value=neumann_val,
    )
    assert result.success is True
    # With zero flux on sides and linear BCs top/bottom, solution should be
    # approximately linear in y: u ≈ y (since bottom=0, top=1)
    y_vals = np.linspace(0.0, 1.0, ny)
    for j in range(1, ny - 1):
        mid_col = nx // 2
        np.testing.assert_allclose(result.u[j, mid_col], y_vals[j], atol=0.1)
