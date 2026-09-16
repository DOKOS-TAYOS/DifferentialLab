"""Tests for coupled linear vector PDE systems in two dimensions."""

from __future__ import annotations

import numpy as np
import pytest

from solver import (
    PDEBoundaryCondition,
    VectorPDEBoundaryConditions,
    VectorPDECoefficients,
    VectorPDEDiagnostics,
    VectorPDESolution,
    solve_vector_pde_2d,
)
from utils import SolverFailedError

_AXX = np.array([[2.0, 0.30], [0.30, 1.5]])
_AXY = np.array([[0.20, 0.10], [0.10, -0.15]])
_AYY = np.array([[1.7, 0.25], [0.25, 1.8]])
_AX = np.array([[0.12, -0.05], [0.04, -0.08]])
_AY = np.array([[-0.06, 0.03], [0.07, 0.09]])
_A0 = np.array([[0.4, 0.7], [-0.5, 0.2]])


def _exact_components(x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
    """Return the genuinely different manufactured component fields."""
    return np.asarray(
        [
            np.sin(np.pi * x) * np.sin(np.pi * y),
            np.sin(2.0 * np.pi * x) * np.sin(np.pi * y),
        ]
    )


def _manufactured_state(x: float, y: float) -> tuple[np.ndarray, ...]:
    """Return f, fx, fy, fxx, fxy, fyy for the manufactured solution."""
    f = _exact_components(x, y)
    fx = np.array(
        [
            np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
            2.0 * np.pi * np.cos(2.0 * np.pi * x) * np.sin(np.pi * y),
        ]
    )
    fy = np.array(
        [
            np.pi * np.sin(np.pi * x) * np.cos(np.pi * y),
            np.pi * np.sin(2.0 * np.pi * x) * np.cos(np.pi * y),
        ]
    )
    fxx = np.array([-(np.pi**2) * f[0], -4.0 * np.pi**2 * f[1]])
    fxy = np.array(
        [
            np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y),
            2.0 * np.pi**2 * np.cos(2.0 * np.pi * x) * np.cos(np.pi * y),
        ]
    )
    fyy = -(np.pi**2) * f
    return f, fx, fy, fxx, fxy, fyy


def _manufactured_provider(
    x: float,
    y: float,
    _params: dict[str, float],
) -> VectorPDECoefficients:
    """Return a coupled elliptic operator and exact manufactured forcing."""
    f, fx, fy, fxx, fxy, fyy = _manufactured_state(x, y)
    constant = -(_AXX @ fxx + _AXY @ fxy + _AYY @ fyy + _AX @ fx + _AY @ fy + _A0 @ f)
    return VectorPDECoefficients(_AXX, _AXY, _AYY, _AX, _AY, _A0, constant)


def _manufactured_boundaries() -> VectorPDEBoundaryConditions:
    """Return exact component-specific Dirichlet data."""
    component_conditions = tuple(
        PDEBoundaryCondition.dirichlet(
            lambda x, y, component=component: float(_exact_components(x, y)[component])
        )
        for component in range(2)
    )
    return VectorPDEBoundaryConditions(
        left=component_conditions,
        right=component_conditions,
        bottom=component_conditions,
        top=component_conditions,
    )


def _solve_manufactured(points: int) -> VectorPDESolution:
    """Solve the coupled manufactured problem at one resolution."""
    return solve_vector_pde_2d(
        None,
        0.0,
        1.0,
        0.0,
        1.0,
        points,
        points,
        components=2,
        coefficient_provider=_manufactured_provider,
        boundary_conditions=_manufactured_boundaries(),
    )


def _component_errors(solution: VectorPDESolution) -> np.ndarray:
    """Compute component L2 grid errors for the manufactured fields."""
    x, y = solution.grid
    xx, yy = np.meshgrid(x, y)
    exact = _exact_components(xx, yy)
    return np.sqrt(np.mean((solution.u - exact) ** 2, axis=(1, 2)))


def test_coupled_manufactured_solution_converges_at_second_order() -> None:
    """Off-diagonal derivative and zero-order blocks affect both equations."""
    grids = (13, 25, 49)
    errors = np.asarray([_component_errors(_solve_manufactured(points)) for points in grids])
    orders = np.log(errors[:-1] / errors[1:]) / np.log(2.0)

    assert np.all(np.diff(errors, axis=0) < 0.0)
    assert np.all(orders > 1.65), (errors, orders)
    assert np.any(_A0 - np.diag(np.diag(_A0)))
    assert np.any(_AXY != 0.0)


def test_vector_residual_probe_recovers_full_cross_component_affinity() -> None:
    """A residual path using every matrix block matches the manufactured solution."""

    def residual(
        x: float,
        y: float,
        f: np.ndarray,
        fx: np.ndarray,
        fy: np.ndarray,
        fxx: np.ndarray,
        fxy: np.ndarray,
        fyy: np.ndarray,
        **_params: float,
    ) -> np.ndarray:
        coefficients = _manufactured_provider(x, y, {})
        return (
            coefficients.fxx @ fxx
            + coefficients.fxy @ fxy
            + coefficients.fyy @ fyy
            + coefficients.fx @ fx
            + coefficients.fy @ fy
            + coefficients.f @ f
            + coefficients.constant
        )

    result = solve_vector_pde_2d(
        residual,
        0.0,
        1.0,
        0.0,
        1.0,
        17,
        17,
        components=2,
        boundary_conditions=_manufactured_boundaries(),
    )

    assert result.u.shape == (2, 17, 17)
    assert np.max(_component_errors(result)) < 0.015
    assert isinstance(result.diagnostics, VectorPDEDiagnostics)
    assert len(result.diagnostics.component_residual_l2) == 2
    assert result.diagnostics.relative_residual_l2 < 1.0e-10


def test_nonlinear_cross_component_residual_is_rejected() -> None:
    def residual(
        _x: float,
        _y: float,
        f: np.ndarray,
        _fx: np.ndarray,
        _fy: np.ndarray,
        fxx: np.ndarray,
        _fxy: np.ndarray,
        fyy: np.ndarray,
        **_params: float,
    ) -> np.ndarray:
        return np.array([fxx[0] + fyy[0] + f[0] * f[1], fxx[1] + fyy[1]])

    with pytest.raises(SolverFailedError, match="not affine.*complete coupled state"):
        solve_vector_pde_2d(
            residual,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            5,
            components=2,
        )


def test_vector_residual_must_return_exact_component_length() -> None:
    def residual(
        _x: float,
        _y: float,
        _f: np.ndarray,
        _fx: np.ndarray,
        _fy: np.ndarray,
        _fxx: np.ndarray,
        _fxy: np.ndarray,
        _fyy: np.ndarray,
        **_params: float,
    ) -> np.ndarray:
        return np.zeros(3)

    with pytest.raises(SolverFailedError, match=r"must have shape \(2,\), got \(3,\)"):
        solve_vector_pde_2d(
            residual,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            5,
            components=2,
        )


def test_vector_boundary_container_type_is_validated_explicitly() -> None:
    with pytest.raises(SolverFailedError, match="VectorPDEBoundaryConditions object"):
        solve_vector_pde_2d(
            None,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            5,
            components=2,
            coefficient_provider=_identity_laplacian_provider,
            boundary_conditions=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fxx", np.eye(3), r"fxx.*shape \(2, 2\)"),
        ("fxy", np.ones((2, 1)), r"fxy.*shape \(2, 2\)"),
        ("constant", np.zeros((2, 1)), r"constant.*shape \(2,\)"),
    ],
)
def test_coefficient_matrix_and_vector_dimensions_are_exact(
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    def provider(
        _x: float,
        _y: float,
        _params: dict[str, float],
    ) -> VectorPDECoefficients:
        values: dict[str, np.ndarray] = {
            "fxx": np.eye(2),
            "fxy": np.zeros((2, 2)),
            "fyy": np.eye(2),
            "fx": np.zeros((2, 2)),
            "fy": np.zeros((2, 2)),
            "f": np.zeros((2, 2)),
            "constant": np.zeros(2),
        }
        values[field] = value
        return VectorPDECoefficients(**values)

    with pytest.raises(SolverFailedError, match=message):
        solve_vector_pde_2d(
            None,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            5,
            components=2,
            coefficient_provider=provider,
        )


def test_indefinite_principal_symbol_is_rejected() -> None:
    def provider(
        _x: float,
        _y: float,
        _params: dict[str, float],
    ) -> VectorPDECoefficients:
        indefinite = np.diag([1.0, -1.0])
        zero = np.zeros((2, 2))
        return VectorPDECoefficients(indefinite, zero, indefinite, zero, zero, zero, np.zeros(2))

    with pytest.raises(SolverFailedError, match="principal symbol is not uniformly definite"):
        solve_vector_pde_2d(
            None,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            5,
            components=2,
            coefficient_provider=provider,
        )


def test_principal_symbol_reports_the_first_failing_direction() -> None:
    def provider(
        _x: float,
        _y: float,
        _params: dict[str, float],
    ) -> VectorPDECoefficients:
        positive = np.eye(2)
        negative = -np.eye(2)
        zero = np.zeros((2, 2))
        return VectorPDECoefficients(positive, zero, negative, zero, zero, zero, np.zeros(2))

    with pytest.raises(
        SolverFailedError,
        match=r"principal symbol is not uniformly definite in sampled direction 8",
    ):
        solve_vector_pde_2d(
            None,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            5,
            components=2,
            coefficient_provider=provider,
        )


def test_principal_symbol_orientation_must_be_consistent_in_domain_component() -> None:
    def provider(
        x: float,
        _y: float,
        _params: dict[str, float],
    ) -> VectorPDECoefficients:
        principal = np.eye(2) if x < 0.5 else -np.eye(2)
        zero = np.zeros((2, 2))
        return VectorPDECoefficients(principal, zero, principal, zero, zero, zero, np.zeros(2))

    with pytest.raises(SolverFailedError, match="orientation changes within connected"):
        solve_vector_pde_2d(
            None,
            0.0,
            1.0,
            0.0,
            1.0,
            7,
            7,
            components=2,
            coefficient_provider=provider,
        )


def _identity_laplacian_provider(
    _x: float,
    _y: float,
    _params: dict[str, float],
) -> VectorPDECoefficients:
    zero = np.zeros((2, 2))
    return VectorPDECoefficients(np.eye(2), zero, np.eye(2), zero, zero, zero, np.zeros(2))


def test_boundary_condition_broadcasting_is_explicit_and_component_aware() -> None:
    shared = PDEBoundaryCondition.dirichlet(2.0)
    shared_result = solve_vector_pde_2d(
        None,
        0.0,
        1.0,
        0.0,
        1.0,
        7,
        7,
        components=2,
        coefficient_provider=_identity_laplacian_provider,
        boundary_conditions=VectorPDEBoundaryConditions(
            left=shared,
            right=shared,
            bottom=shared,
            top=shared,
        ),
    )
    np.testing.assert_allclose(shared_result.u, 2.0)

    specific = (
        PDEBoundaryCondition.dirichlet(1.0),
        PDEBoundaryCondition.dirichlet(3.0),
    )
    specific_result = solve_vector_pde_2d(
        None,
        0.0,
        1.0,
        0.0,
        1.0,
        7,
        7,
        components=2,
        coefficient_provider=_identity_laplacian_provider,
        boundary_conditions=VectorPDEBoundaryConditions(
            left=specific,
            right=specific,
            bottom=specific,
            top=specific,
        ),
    )
    np.testing.assert_allclose(specific_result.u[0], 1.0)
    np.testing.assert_allclose(specific_result.u[1], 3.0)

    ambiguous_length_one = VectorPDEBoundaryConditions(left=(shared,))
    with pytest.raises(SolverFailedError, match="length-one sequences are not broadcast"):
        solve_vector_pde_2d(
            None,
            0.0,
            1.0,
            0.0,
            1.0,
            5,
            5,
            components=2,
            coefficient_provider=_identity_laplacian_provider,
            boundary_conditions=ambiguous_length_one,
        )


def test_shared_neumann_uses_outward_normal_with_component_specific_anchors() -> None:
    shared_left = PDEBoundaryCondition.neumann(-1.0)
    shared_right = PDEBoundaryCondition.neumann(1.0)
    component_values = (
        PDEBoundaryCondition.dirichlet(lambda x, _y: x),
        PDEBoundaryCondition.dirichlet(lambda x, _y: x + 2.0),
    )
    result = solve_vector_pde_2d(
        None,
        0.0,
        1.0,
        0.0,
        1.0,
        9,
        7,
        components=2,
        coefficient_provider=_identity_laplacian_provider,
        boundary_conditions=VectorPDEBoundaryConditions(
            left=shared_left,
            right=shared_right,
            bottom=component_values,
            top=component_values,
        ),
    )
    expected_0 = np.broadcast_to(result.grid[0], (len(result.grid[1]), len(result.grid[0])))
    np.testing.assert_allclose(result.u[0], expected_0, atol=1.0e-12)
    np.testing.assert_allclose(result.u[1], expected_0 + 2.0, atol=1.0e-12)


def test_periodic_vector_grid_uses_nonduplicated_endpoints() -> None:
    identity = np.eye(2)
    zero = np.zeros((2, 2))

    def provider(
        _x: float,
        _y: float,
        _params: dict[str, float],
    ) -> VectorPDECoefficients:
        return VectorPDECoefficients(identity, zero, identity, zero, zero, -identity, np.zeros(2))

    result = solve_vector_pde_2d(
        None,
        0.0,
        2.0 * np.pi,
        0.0,
        2.0 * np.pi,
        8,
        9,
        components=2,
        coefficient_provider=provider,
        boundary_conditions=VectorPDEBoundaryConditions(periodic_x=True, periodic_y=True),
    )

    assert result.u.shape == (2, 9, 8)
    assert result.grid[0][-1] < 2.0 * np.pi
    assert result.grid[1][-1] < 2.0 * np.pi
    np.testing.assert_allclose(result.u, 0.0)
