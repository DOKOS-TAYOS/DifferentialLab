"""Tests for the 2D membrane complex-problem solver."""

from __future__ import annotations

import numpy as np

from complex_problems.membrane_2d.model import build_initial_displacement
from complex_problems.membrane_2d.solver import solve_membrane_2d


def test_membrane_verlet_linear_keeps_energy_drift_small() -> None:
    u0 = build_initial_displacement(
        nx=24,
        ny=24,
        shape="mode",
        amplitude=1.0,
        sigma=0.2,
        mode_x=1,
        mode_y=1,
        boundary="fixed",
    )
    result = solve_membrane_2d(
        u0=u0,
        t_min=0.0,
        t_max=2.0,
        dt=0.01,
        mass=1.0,
        k_linear=1.0,
        boundary="fixed",
        integrator="verlet",
    )

    assert result.displacement.shape == (len(result.t), 24, 24)
    assert result.velocity.shape == (len(result.t), 24, 24)
    assert result.spectrum_power.shape == (24, 24)
    assert abs(result.magnitudes["energy_drift_rel"]) < 5e-2


def test_membrane_non_linear_terms_run_and_return_finite_values() -> None:
    u0 = build_initial_displacement(
        nx=20,
        ny=18,
        shape="gaussian",
        amplitude=0.8,
        sigma=0.14,
        center_x=0.45,
        center_y=0.55,
        boundary="periodic",
    )
    result = solve_membrane_2d(
        u0=u0,
        t_min=0.0,
        t_max=1.0,
        dt=0.01,
        mass=1.0,
        k_linear=1.0,
        boundary="periodic",
        integrator="verlet",
        alpha=0.05,
        beta=0.02,
        high_order_coeff=0.005,
        high_order_power=5,
    )

    assert np.all(np.isfinite(result.total_energy))
    assert np.all(np.isfinite(result.displacement))
    assert np.all(np.isfinite(result.velocity))


def test_membrane_rk45_runs_small_case() -> None:
    u0 = build_initial_displacement(
        nx=10,
        ny=10,
        shape="random",
        amplitude=0.2,
        sigma=0.1,
        random_seed=42,
        boundary="fixed",
    )
    result = solve_membrane_2d(
        u0=u0,
        t_min=0.0,
        t_max=0.2,
        dt=0.02,
        mass=1.0,
        k_linear=0.8,
        boundary="fixed",
        integrator="rk45",
    )

    assert len(result.t) > 2
    assert np.all(np.isfinite(result.total_energy))
