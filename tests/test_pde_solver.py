"""Tests for solver.pde_solver."""

from __future__ import annotations

import numpy as np

from solver.pde_solver import PDESolution, solve_pde_2d


def _laplace_residual(x: float, y: float, f: float, fx: float, fy: float,
                      fxx: float, fxy: float, fyy: float, **kw: object) -> float:
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
    np.testing.assert_allclose(result.u[0, :], 0.0)   # Bottom edge
    np.testing.assert_allclose(result.u[0:-1, 0], 0.0)   # Left edge (excl top-left corner)
    np.testing.assert_allclose(result.u[0:-1, -1], 0.0)  # Right edge (excl top-right corner)
    # Interior should be between 0 and 1 (maximum principle for Laplace)
    interior = result.u[1:-1, 1:-1]
    assert np.all(interior >= -1e-6) and np.all(interior <= 1.0 + 1e-6)


def test_poisson_simple_rhs() -> None:
    """Poisson -f_xx - f_yy = 1 with zero BC."""
    def residual(x: float, y: float, f: float, fx: float, fy: float,
                 fxx: float, fxy: float, fyy: float, **kw: object) -> float:
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
