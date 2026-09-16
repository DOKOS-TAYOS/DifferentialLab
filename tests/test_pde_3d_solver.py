"""Numerical and validation tests for the scalar rectangular PDE 3D solver."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import solver.pde_3d_solver as pde_3d_solver
from solver.pde_3d_solver import (
    _assemble_pde_3d,
    _boundary_substitution_3d,
    _prepare_boundary_3d,
    solve_pde_3d,
)
from solver.pde_types import (
    PDEBoundaryCondition3D,
    PDEBoundaryConditions3D,
    PDECoefficients3D,
    PDEDiagnostics,
    PDESolution3D,
)
from utils import SolverFailedError


def test_prepared_boundary_uses_compact_kind_and_inward_direction_arrays() -> None:
    """Internal 3D boundary storage preserves substitutions without object/index grids."""
    grid = np.linspace(0.0, 1.0, 5)
    prepared = _prepare_boundary_3d(
        PDEBoundaryConditions3D(x_min=PDEBoundaryCondition3D.neumann(2.0)),
        x=grid,
        y=grid,
        z=grid,
        hx=0.25,
        hy=0.25,
        hz=0.25,
    )

    assert prepared.kind.dtype == np.uint8
    assert prepared.inward_direction.dtype == np.uint8
    assert not hasattr(prepared, "inward_i")
    assert not hasattr(prepared, "step")
    inward_i, inward_j, inward_k, factor, offset = _boundary_substitution_3d(prepared, 0, 2, 2)
    assert (inward_i, inward_j, inward_k) == (1, 2, 2)
    assert factor == pytest.approx(1.0)
    assert offset == pytest.approx(0.5)


def test_assembler_uses_direct_path_for_dirichlet_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dirichlet neighbors contribute their value without boundary substitution."""
    grid = np.linspace(0.0, 1.0, 3)
    boundary = _prepare_boundary_3d(
        PDEBoundaryConditions3D(
            x_min=PDEBoundaryCondition3D.dirichlet(2.0),
            x_max=PDEBoundaryCondition3D.dirichlet(2.0),
            y_min=PDEBoundaryCondition3D.dirichlet(2.0),
            y_max=PDEBoundaryCondition3D.dirichlet(2.0),
            z_min=PDEBoundaryCondition3D.dirichlet(2.0),
            z_max=PDEBoundaryCondition3D.dirichlet(2.0),
        ),
        x=grid,
        y=grid,
        z=grid,
        hx=0.5,
        hy=0.5,
        hz=0.5,
    )
    coefficients: PDECoefficients3D = (
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    def coefficient_at(
        _x: float,
        _y: float,
        _z: float,
    ) -> tuple[PDECoefficients3D, int]:
        """Return a constant positive-definite Laplacian."""
        return coefficients, 1

    def fail_if_substitution_is_called(
        _boundary: object,
        _i: int,
        _j: int,
        _k: int,
    ) -> tuple[int, int, int, float, float]:
        """Reject the non-Dirichlet elimination path in this regression test."""
        raise AssertionError("Dirichlet neighbors must not use boundary substitution")

    monkeypatch.setattr(
        pde_3d_solver,
        "_boundary_substitution_3d",
        fail_if_substitution_is_called,
    )
    assembled = _assemble_pde_3d(
        grid,
        grid,
        grid,
        hx=0.5,
        hy=0.5,
        hz=0.5,
        boundary=boundary,
        coefficient_at=coefficient_at,
    )

    assert assembled.matrix.shape == (1, 1)
    assert assembled.rhs == pytest.approx([-48.0])


def _negative_laplacian_residual(
    x: float,
    y: float,
    z: float,
    f: float,
    fx: float,
    fy: float,
    fz: float,
    fxx: float,
    fxy: float,
    fxz: float,
    fyy: float,
    fyz: float,
    fzz: float,
    **kwargs: object,
) -> float:
    """Return the homogeneous negative-Laplacian residual."""
    del x, y, z, f, fx, fy, fz, fxy, fxz, fyz, kwargs
    return -fxx - fyy - fzz


def test_zero_dirichlet_solution_has_public_shape_and_diagnostics() -> None:
    """The basic rectangular solve returns z/y/x ordering and sparse evidence."""
    result = solve_pde_3d(
        _negative_laplacian_residual,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        7,
        6,
        5,
    )

    assert isinstance(result, PDESolution3D)
    assert result.u.shape == (5, 6, 7)
    assert tuple(grid.shape for grid in result.grid) == ((7,), (6,), (5,))
    np.testing.assert_allclose(result.u, 0.0, atol=1.0e-12)
    assert isinstance(result.diagnostics, PDEDiagnostics)
    assert result.diagnostics.matrix_shape == (60, 60)
    assert result.diagnostics.nnz > 0
    assert result.diagnostics.relative_residual_l2 == pytest.approx(0.0)


def test_residual_path_rejects_nonlinearity() -> None:
    """A residual that is nonlinear in the solution state must not be linearized."""

    def nonlinear(
        x: float,
        y: float,
        z: float,
        f: float,
        fx: float,
        fy: float,
        fz: float,
        fxx: float,
        fxy: float,
        fxz: float,
        fyy: float,
        fyz: float,
        fzz: float,
    ) -> float:
        del x, y, z, fx, fy, fz, fxy, fxz, fyz
        return -fxx - fyy - fzz + f**2

    with pytest.raises(SolverFailedError, match="not affine"):
        solve_pde_3d(nonlinear, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 5, 5, 5)


@pytest.mark.parametrize(
    "provider",
    [
        lambda x, y, z, params: (1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        lambda x, y, z, params: (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_non_elliptic_or_degenerate_principal_operator_is_rejected(
    provider: Callable[..., tuple[float, ...]],
) -> None:
    """The 3x3 principal matrix must be strictly definite."""
    with pytest.raises(SolverFailedError, match="not strictly elliptic"):
        solve_pde_3d(
            None,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            5,
            5,
            coefficient_provider=provider,
        )


def test_non_finite_direct_coefficients_are_rejected() -> None:
    """The direct provider path still validates every coefficient."""

    def provider(
        x: float,
        y: float,
        z: float,
        params: dict[str, float],
    ) -> tuple[float, ...]:
        del x, y, z, params
        return (-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.inf)

    with pytest.raises(SolverFailedError, match="coefficients.*must be finite"):
        solve_pde_3d(
            None,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            5,
            5,
            coefficient_provider=provider,
        )


def test_finite_direct_coefficient_provider_solves_without_residual_probing() -> None:
    """The direct path accepts the documented eleven scalar coefficients."""

    def provider(
        x: float,
        y: float,
        z: float,
        params: dict[str, float],
    ) -> tuple[float, ...]:
        del x, y, z, params
        return (-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    result = solve_pde_3d(
        None,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        5,
        5,
        5,
        coefficient_provider=provider,
    )
    np.testing.assert_allclose(result.u, 0.0)


def test_periodic_laplacian_nullspace_is_reported_as_singular() -> None:
    """The constant nullspace must be a solver failure, not a finite-looking result."""
    boundaries = PDEBoundaryConditions3D(periodic_x=True, periodic_y=True, periodic_z=True)
    with pytest.raises(SolverFailedError, match=r"Linear solver failed|singular"):
        solve_pde_3d(
            _negative_laplacian_residual,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            7,
            7,
            7,
            boundary_conditions=boundaries,
        )


def test_robin_face_recovers_linear_manufactured_solution() -> None:
    """One-sided Robin elimination is exact for a linear field."""

    def exact(x: float | np.ndarray, y: float | np.ndarray, z: float | np.ndarray) -> object:
        return 1.0 + 2.0 * x + 3.0 * y + 4.0 * z

    dirichlet = PDEBoundaryCondition3D.dirichlet(exact)
    boundaries = PDEBoundaryConditions3D(
        x_min=dirichlet,
        x_max=PDEBoundaryCondition3D.robin(
            alpha=2.0,
            beta=0.5,
            gamma=lambda x, y, z: 2.0 * float(exact(x, y, z)) + 1.0,
        ),
        y_min=dirichlet,
        y_max=dirichlet,
        z_min=dirichlet,
        z_max=dirichlet,
    )
    result = solve_pde_3d(
        _negative_laplacian_residual,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        8,
        7,
        6,
        boundary_conditions=boundaries,
    )
    x, y, z = result.grid
    x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z, indexing="xy")
    expected = np.transpose(np.asarray(exact(x_mesh, y_mesh, z_mesh)), (2, 0, 1))
    np.testing.assert_allclose(result.u, expected, atol=2.0e-11)


def test_neumann_face_recovers_linear_manufactured_solution() -> None:
    """The outward-normal sign on a maximum-coordinate face is explicit."""

    def exact(x: float, y: float, z: float) -> float:
        return 1.0 + 2.0 * x + 3.0 * y + 4.0 * z

    dirichlet = PDEBoundaryCondition3D.dirichlet(exact)
    boundaries = PDEBoundaryConditions3D(
        x_min=dirichlet,
        x_max=PDEBoundaryCondition3D.neumann(2.0),
        y_min=dirichlet,
        y_max=dirichlet,
        z_min=dirichlet,
        z_max=dirichlet,
    )
    result = solve_pde_3d(
        _negative_laplacian_residual,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        8,
        7,
        6,
        boundary_conditions=boundaries,
    )
    z_mesh, y_mesh, x_mesh = np.meshgrid(*result.grid[::-1], indexing="ij")
    expected = 1.0 + 2.0 * x_mesh + 3.0 * y_mesh + 4.0 * z_mesh
    np.testing.assert_allclose(result.u, expected, atol=2.0e-11)


def test_periodic_axis_uses_nonduplicated_endpoint_and_wraps() -> None:
    """A periodic x-axis solves a nonsingular manufactured Helmholtz problem."""
    boundaries = PDEBoundaryConditions3D(periodic_x=True)

    def residual(
        x: float,
        y: float,
        z: float,
        f: float,
        fx: float,
        fy: float,
        fz: float,
        fxx: float,
        fxy: float,
        fxz: float,
        fyy: float,
        fyz: float,
        fzz: float,
    ) -> float:
        del fx, fy, fz, fxy, fxz, fyz
        exact = np.sin(2.0 * np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
        return -fxx - fyy - fzz + f - (6.0 * np.pi**2 + 1.0) * exact

    result = solve_pde_3d(
        residual,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        20,
        11,
        11,
        boundary_conditions=boundaries,
    )
    x, y, z = result.grid
    z_mesh, y_mesh, x_mesh = np.meshgrid(z, y, x, indexing="ij")
    exact = np.sin(2.0 * np.pi * x_mesh) * np.sin(np.pi * y_mesh) * np.sin(np.pi * z_mesh)
    assert x[-1] == pytest.approx(19.0 / 20.0)
    assert np.max(np.abs(result.u - exact)) < 1.7e-2


def test_fully_periodic_mixed_derivatives_wrap_all_diagonal_planes() -> None:
    """xy, xz, and yz mixed stencils wrap across periodic seams."""
    boundaries = PDEBoundaryConditions3D(periodic_x=True, periodic_y=True, periodic_z=True)
    cross_xy, cross_xz, cross_yz = 0.20, -0.15, 0.10

    def residual(
        x: float,
        y: float,
        z: float,
        f: float,
        fx: float,
        fy: float,
        fz: float,
        fxx: float,
        fxy: float,
        fxz: float,
        fyy: float,
        fyz: float,
        fzz: float,
    ) -> float:
        del fx, fy, fz
        phase_x, phase_y, phase_z = 2.0 * np.pi * x, 2.0 * np.pi * y, 2.0 * np.pi * z
        exact = np.sin(phase_x) * np.cos(phase_y) * np.sin(phase_z)
        xy = (2.0 * np.pi) ** 2 * -np.cos(phase_x) * np.sin(phase_y) * np.sin(phase_z)
        xz = (2.0 * np.pi) ** 2 * np.cos(phase_x) * np.cos(phase_y) * np.cos(phase_z)
        yz = (2.0 * np.pi) ** 2 * -np.sin(phase_x) * np.sin(phase_y) * np.cos(phase_z)
        forcing = (3.0 * (2.0 * np.pi) ** 2 + 1.0) * exact
        forcing += cross_xy * xy + cross_xz * xz + cross_yz * yz
        return -fxx - fyy - fzz + cross_xy * fxy + cross_xz * fxz + cross_yz * fyz + f - forcing

    result = solve_pde_3d(
        residual,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        14,
        14,
        14,
        boundary_conditions=boundaries,
    )
    x, y, z = result.grid
    z_mesh, y_mesh, x_mesh = np.meshgrid(z, y, x, indexing="ij")
    exact = (
        np.sin(2.0 * np.pi * x_mesh) * np.cos(2.0 * np.pi * y_mesh) * np.sin(2.0 * np.pi * z_mesh)
    )
    assert np.max(np.abs(result.u - exact)) < 7.0e-2


def test_smooth_manufactured_solution_has_second_order_convergence() -> None:
    """Nested modest grids demonstrate near-second-order central differences."""

    def residual(
        x: float,
        y: float,
        z: float,
        f: float,
        fx: float,
        fy: float,
        fz: float,
        fxx: float,
        fxy: float,
        fxz: float,
        fyy: float,
        fyz: float,
        fzz: float,
    ) -> float:
        del f, fx, fy, fz, fxy, fxz, fyz
        exact = np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
        return -fxx - fyy - fzz - 3.0 * np.pi**2 * exact

    errors: list[float] = []
    spacings: list[float] = []
    for grid_size in (7, 11, 15):
        result = solve_pde_3d(
            residual,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            grid_size,
            grid_size,
            grid_size,
        )
        x, y, z = result.grid
        z_mesh, y_mesh, x_mesh = np.meshgrid(z, y, x, indexing="ij")
        exact = np.sin(np.pi * x_mesh) * np.sin(np.pi * y_mesh) * np.sin(np.pi * z_mesh)
        errors.append(float(np.sqrt(np.mean((result.u - exact) ** 2))))
        spacings.append(1.0 / (grid_size - 1))
    observed_orders = [
        np.log(errors[index] / errors[index + 1]) / np.log(spacings[index] / spacings[index + 1])
        for index in range(len(errors) - 1)
    ]
    assert min(observed_orders) > 1.8
    assert max(observed_orders) < 2.2
