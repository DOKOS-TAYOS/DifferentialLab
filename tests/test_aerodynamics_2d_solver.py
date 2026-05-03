"""Tests for 2D aerodynamics solver."""

from __future__ import annotations

import numpy as np
import pytest

from complex_problems.aerodynamics_2d import model
from complex_problems.aerodynamics_2d.solver import solve_aerodynamics_2d


def test_nonlinear_ns_runs_and_returns_finite_fields() -> None:
    result = solve_aerodynamics_2d(
        approximation="nonlinear_ns",
        nx=48,
        ny=32,
        lx=4.0,
        ly=2.0,
        t_max=0.2,
        dt=0.002,
        sample_every=5,
        rho=1.0,
        nu=0.02,
        u_inf=1.0,
        penalization=0.01,
        obstacle_shape="cylinder",
        obstacle_size_x=0.35,
        obstacle_size_y=0.35,
    )

    assert result.u.shape[1:] == (32, 48)
    assert np.any(result.obstacle_mask)
    assert np.all(np.isfinite(result.speed))
    assert result.magnitudes["max_divergence_l2"] < 5e-1


def test_stokes_approximation_runs() -> None:
    result = solve_aerodynamics_2d(
        approximation="stokes",
        nx=40,
        ny=28,
        lx=3.0,
        ly=2.0,
        t_max=0.1,
        dt=0.002,
        sample_every=4,
        rho=1.0,
        nu=0.03,
        u_inf=0.8,
        penalization=0.01,
        obstacle_shape="ellipse",
        obstacle_size_x=0.4,
        obstacle_size_y=0.2,
    )
    assert len(result.t) > 2
    assert np.all(np.isfinite(result.drag_coeff))
    assert np.all(np.isfinite(result.lift_coeff))


def test_aerodynamics_rejects_invalid_approximation() -> None:
    with pytest.raises(ValueError):
        solve_aerodynamics_2d(
            approximation="invalid",
            nx=32,
            ny=24,
            lx=2.0,
            ly=1.0,
        )


def test_periodic_derivatives_match_roll_formulas_without_calling_np_roll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    f = np.arange(20, dtype=float).reshape(4, 5)
    u = f + 1.0
    v = 2.0 * f - 3.0
    dx = 0.2
    dy = 0.5

    expected_ddx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)
    expected_ddy = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dy)
    expected_lap = (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / (dx * dx) + (
        np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)
    ) / (dy * dy)
    expected_div = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dx) + (
        np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)
    ) / (2.0 * dy)

    def fail_roll(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("periodic derivative helpers should avoid np.roll temporaries")

    monkeypatch.setattr(np, "roll", fail_roll)

    np.testing.assert_allclose(model.ddx_periodic(f, dx), expected_ddx)
    np.testing.assert_allclose(model.ddy_periodic(f, dy), expected_ddy)
    np.testing.assert_allclose(model.laplacian_periodic(f, dx, dy), expected_lap)
    np.testing.assert_allclose(model.divergence_periodic(u, v, dx, dy), expected_div)
