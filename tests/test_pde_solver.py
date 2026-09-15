"""Tests for solver.pde_solver."""

from __future__ import annotations

import numpy as np
import pytest

from solver.pde_solver import (
    BC_DIRICHLET,
    BC_NEUMANN,
    PDEDiagnostics,
    PDESolution,
    _classify_mask,
    solve_pde_2d,
)
from utils import SolverFailedError


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


def test_classify_mask_returns_boundary_and_row_major_index_grid() -> None:
    """Interior points should get a compact row-major index grid."""
    mask = np.zeros((5, 6), dtype=bool)
    mask[1:4, 1:5] = True

    interior, boundary, index_grid = _classify_mask(mask)

    expected_interior = np.zeros_like(mask)
    expected_interior[2, 2] = True
    expected_interior[2, 3] = True
    expected_boundary = mask & ~expected_interior

    np.testing.assert_array_equal(interior, expected_interior)
    np.testing.assert_array_equal(boundary, expected_boundary)
    np.testing.assert_array_equal(index_grid[~expected_interior], -1)
    assert index_grid[2, 2] == 0
    assert index_grid[2, 3] == 1


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
    assert isinstance(result.diagnostics, PDEDiagnostics)
    assert result.diagnostics.matrix_shape == (81, 81)
    assert result.diagnostics.nnz > 0
    assert result.diagnostics.discrete_residual_l2 == pytest.approx(0.0)
    assert result.diagnostics.condition_estimate is not None


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


@pytest.mark.parametrize(
    ("edge", "normal_derivative"),
    [
        ("left", -2.0),
        ("right", 2.0),
        ("bottom", 3.0),
        ("top", -3.0),
    ],
)
def test_nonzero_outward_neumann_data_on_each_rectangular_edge(
    edge: str,
    normal_derivative: float,
) -> None:
    """Nonzero Neumann data uses du/dn on every rectangular edge."""
    nx, ny = 9, 8
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    x_mesh, y_mesh = np.meshgrid(x, y)
    exact = 1.5 + 2.0 * x_mesh - 3.0 * y_mesh
    bc_type = np.full((ny, nx), BC_DIRICHLET, dtype=object)
    neumann_values = np.zeros((ny, nx))

    if edge == "left":
        edge_index = (slice(1, -1), 0)
    elif edge == "right":
        edge_index = (slice(1, -1), -1)
    elif edge == "bottom":
        edge_index = (0, slice(1, -1))
    else:
        edge_index = (-1, slice(1, -1))
    bc_type[edge_index] = BC_NEUMANN
    neumann_values[edge_index] = normal_derivative

    result = solve_pde_2d(
        _laplace_residual,
        0.0,
        1.0,
        0.0,
        1.0,
        nx,
        ny,
        bc_values=exact,
        bc_type=bc_type,
        bc_neumann_value=neumann_values,
    )

    np.testing.assert_allclose(result.u[1:-1, 1:-1], exact[1:-1, 1:-1], atol=1.0e-11)
    np.testing.assert_allclose(result.u[edge_index], exact[edge_index], atol=1.0e-11)


def test_mixed_derivative_next_to_nonzero_neumann_boundary() -> None:
    """The f_xy stencil eliminates a Neumann diagonal along its normal."""
    nx = ny = 9
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    x_mesh, y_mesh = np.meshgrid(x, y)
    exact = x_mesh * y_mesh
    bc_type = np.full((ny, nx), BC_DIRICHLET, dtype=object)
    bc_type[1:-1, 0] = BC_NEUMANN
    neumann_values = np.zeros((ny, nx))
    neumann_values[1:-1, 0] = -y[1:-1]

    def coefficients(
        x_value: float,
        y_value: float,
        params: dict[str, float],
    ) -> tuple[float, float, float, float, float, float, float]:
        del x_value, y_value, params
        return (-1.0, -0.5, -1.0, 0.0, 0.0, 0.0, 0.5)

    result = solve_pde_2d(
        _laplace_residual,
        0.0,
        1.0,
        0.0,
        1.0,
        nx,
        ny,
        bc_values=exact,
        bc_type=bc_type,
        bc_neumann_value=neumann_values,
        coefficient_provider=coefficients,
    )

    np.testing.assert_allclose(result.u[1:-1, 1:-1], exact[1:-1, 1:-1], atol=1.0e-11)
    np.testing.assert_allclose(result.u[1:-1, 0], exact[1:-1, 0], atol=1.0e-11)


def test_mixed_derivative_rejects_ambiguous_masked_neumann_diagonal() -> None:
    """A masked diagonal without a unique grid normal must fail clearly."""
    nx = ny = 7
    mask = np.ones((ny, nx), dtype=bool)
    mask[3, 2] = False
    bc_type = np.full((ny, nx), BC_DIRICHLET, dtype=object)
    bc_type[3, 3] = BC_NEUMANN

    def coefficients(
        x: float,
        y: float,
        params: dict[str, float],
    ) -> tuple[float, float, float, float, float, float, float]:
        del x, y, params
        return (-1.0, -0.5, -1.0, 0.0, 0.0, 0.0, 0.0)

    with pytest.raises(
        SolverFailedError,
        match=r"mixed-derivative stencil.*unique rectangular grid normal",
    ):
        solve_pde_2d(
            _laplace_residual,
            0.0,
            1.0,
            0.0,
            1.0,
            nx,
            ny,
            mask=mask,
            bc_type=bc_type,
            coefficient_provider=coefficients,
        )


def test_solve_pde_2d_uses_coefficient_provider_without_probing_residual() -> None:
    def residual_should_not_be_called(*args: object, **kwargs: object) -> float:
        raise AssertionError("generic coefficient probing should not run")

    def coefficients(x: float, y: float, params: dict[str, float]) -> tuple[float, ...]:
        return (-1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0)

    result = solve_pde_2d(
        residual_should_not_be_called,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        nx=9,
        ny=9,
        coefficient_provider=coefficients,
    )

    assert result.success is True
    np.testing.assert_allclose(result.u, 0.0, atol=1e-10)


def test_nonlinear_residual_is_rejected_at_the_probe_coordinate() -> None:
    """A residual nonlinear in the solution state is outside the solver contract."""

    def nonlinear_residual(
        x: float,
        y: float,
        f: float,
        fx: float,
        fy: float,
        fxx: float,
        fxy: float,
        fyy: float,
        **kwargs: object,
    ) -> float:
        return -fxx - fyy + f**2

    with pytest.raises(SolverFailedError, match=r"not affine.*\(0\.25, 0\.25\)"):
        solve_pde_2d(nonlinear_residual, 0.0, 1.0, 0.0, 1.0, 5, 5)


@pytest.mark.parametrize(
    ("principal_coefficients", "classification"),
    [
        ((1.0, 0.0, -1.0), "hyperbolic"),
        ((1.0, 0.0, 0.0), "parabolic"),
        ((1.0, 2.0, 1.0), "degenerate"),
        ((0.0, 0.0, 0.0), "zero principal part"),
    ],
)
def test_non_elliptic_principal_operators_are_rejected(
    principal_coefficients: tuple[float, float, float],
    classification: str,
) -> None:
    """Hyperbolic, parabolic, and degenerate operators must fail explicitly."""

    def provider(
        x: float,
        y: float,
        params: dict[str, float],
    ) -> tuple[float, float, float, float, float, float, float]:
        del x, y, params
        a_c, bxy, c_c = principal_coefficients
        return (a_c, bxy, c_c, 0.0, 0.0, 0.0, 0.0)

    with pytest.raises(SolverFailedError, match=r"not strictly elliptic|degenerate"):
        solve_pde_2d(
            _laplace_residual,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            5,
            coefficient_provider=provider,
        )


@pytest.mark.parametrize("orientation", [1.0, -1.0])
def test_globally_consistent_ellipticity_orientations_are_accepted(
    orientation: float,
) -> None:
    """Uniformly positive- and negative-definite principal parts remain valid."""

    def provider(
        x: float,
        y: float,
        params: dict[str, float],
    ) -> tuple[float, float, float, float, float, float, float]:
        del x, y, params
        return (orientation, 0.0, orientation, 0.0, 0.0, 0.0, 0.0)

    result = solve_pde_2d(
        _laplace_residual,
        0.0,
        1.0,
        0.0,
        1.0,
        7,
        7,
        coefficient_provider=provider,
    )
    np.testing.assert_allclose(result.u, 0.0, atol=1.0e-12)


def test_spatially_changing_ellipticity_orientation_is_rejected() -> None:
    """Pointwise definite coefficients must keep one orientation globally."""

    def sign_changing_provider(
        x: float,
        y: float,
        params: dict[str, float],
    ) -> tuple[float, float, float, float, float, float, float]:
        del y, params
        orientation = -1.0 if x < 0.5 else 1.0
        return (orientation, 0.0, orientation, 0.0, 0.0, 0.0, 0.0)

    with pytest.raises(
        SolverFailedError,
        match=r"ellipticity orientation changes.*negative definite.*positive definite",
    ):
        solve_pde_2d(
            _laplace_residual,
            0.0,
            1.0,
            0.0,
            1.0,
            7,
            7,
            coefficient_provider=sign_changing_provider,
        )


def test_singular_pure_neumann_system_is_rejected() -> None:
    """The constant nullspace of a pure-Neumann Laplacian is not a success."""
    shape = (7, 7)
    bc_type = np.full(shape, BC_NEUMANN, dtype=object)

    with pytest.raises(SolverFailedError, match=r"Linear solver failed|singular"):
        solve_pde_2d(
            _laplace_residual,
            0.0,
            1.0,
            0.0,
            1.0,
            shape[1],
            shape[0],
            bc_type=bc_type,
        )


def test_non_finite_domain_and_coefficients_are_rejected() -> None:
    """Non-finite public inputs and direct coefficients fail before solving."""
    with pytest.raises(SolverFailedError, match="x_min must be finite"):
        solve_pde_2d(_laplace_residual, np.nan, 1.0, 0.0, 1.0, 5, 5)

    def non_finite_provider(
        x: float,
        y: float,
        params: dict[str, float],
    ) -> tuple[float, float, float, float, float, float, float]:
        del x, y, params
        return (-1.0, 0.0, -1.0, 0.0, 0.0, 0.0, np.inf)

    with pytest.raises(SolverFailedError, match="coefficients.*must be finite"):
        solve_pde_2d(
            _laplace_residual,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            5,
            coefficient_provider=non_finite_provider,
        )


def test_boundary_shapes_labels_and_relevant_values_are_validated() -> None:
    """Boundary data has an exact shape, supported labels, and finite used values."""
    with pytest.raises(SolverFailedError, match=r"bc_values must have shape \(5, 5\)"):
        solve_pde_2d(
            _laplace_residual,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            5,
            bc_values=np.zeros((4, 5)),
        )

    invalid_types = np.full((5, 5), BC_DIRICHLET, dtype=object)
    invalid_types[0, 0] = "periodic"
    with pytest.raises(SolverFailedError, match="boundary labels"):
        solve_pde_2d(
            _laplace_residual,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            5,
            bc_type=invalid_types,
        )

    non_finite_boundary = np.zeros((5, 5))
    non_finite_boundary[0, 0] = np.inf
    with pytest.raises(SolverFailedError, match="Dirichlet boundary"):
        solve_pde_2d(
            _laplace_residual,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            5,
            bc_values=non_finite_boundary,
        )


def test_large_system_does_not_compute_a_dense_condition_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Condition diagnostics must not densify systems above the fixed bound."""

    def fail_if_called(*args: object, **kwargs: object) -> float:
        raise AssertionError("dense condition estimate should not run")

    monkeypatch.setattr(np.linalg, "cond", fail_if_called)
    result = solve_pde_2d(_laplace_residual, 0.0, 1.0, 0.0, 1.0, 19, 19)

    assert result.diagnostics is not None
    assert result.diagnostics.matrix_shape == (289, 289)
    assert result.diagnostics.condition_estimate is None


def test_manufactured_poisson_solution_has_second_order_grid_convergence() -> None:
    """Central differences should converge near order two on a smooth solution."""

    def residual(
        x: float,
        y: float,
        f: float,
        fx: float,
        fy: float,
        fxx: float,
        fxy: float,
        fyy: float,
        **kwargs: object,
    ) -> float:
        del f, fx, fy, fxy, kwargs
        exact = np.sin(np.pi * x) * np.sin(np.pi * y)
        return -fxx - fyy - 2.0 * np.pi**2 * exact

    errors: list[float] = []
    for grid_size in (11, 21, 41):
        result = solve_pde_2d(
            residual,
            0.0,
            1.0,
            0.0,
            1.0,
            grid_size,
            grid_size,
        )
        x_grid, y_grid = result.grid
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        exact = np.sin(np.pi * x_mesh) * np.sin(np.pi * y_mesh)
        errors.append(float(np.sqrt(np.mean((result.u - exact) ** 2))))

    observed_orders = [
        np.log(errors[index] / errors[index + 1]) / np.log(2.0) for index in range(len(errors) - 1)
    ]
    assert min(observed_orders) > 1.8
    assert max(observed_orders) < 2.2
