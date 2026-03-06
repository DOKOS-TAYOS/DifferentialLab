"""Tests for 2D aerodynamics solver."""

from __future__ import annotations

import numpy as np
import pytest

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
