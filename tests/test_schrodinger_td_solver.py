"""Tests for time-dependent Schrodinger solver (1D/2D)."""

from __future__ import annotations

import numpy as np
import pytest

from complex_problems.schrodinger_td.solver import solve_schrodinger_td


def test_schrodinger_1d_preserves_norm_and_returns_expected_shapes() -> None:
    result = solve_schrodinger_td(
        dimension=1,
        x_min=-12.0,
        x_max=12.0,
        nx=256,
        t_min=0.0,
        t_max=1.0,
        dt=0.002,
        hbar=1.0,
        mass=1.0,
        boundary="periodic",
        potential_type="free",
        packet_type="gaussian",
        sigma=0.8,
        x0=-3.0,
        k0x=2.0,
    )

    assert result.dimension == 1
    assert result.psi.shape == (len(result.t), len(result.x))
    assert result.magnitude.shape == result.psi.shape
    assert result.phase.shape == result.psi.shape
    assert np.all(np.isfinite(result.magnitude))
    assert abs(result.magnitudes["norm_drift_rel"]) < 1e-2


def test_schrodinger_2d_runs_and_observables_are_finite() -> None:
    result = solve_schrodinger_td(
        dimension=2,
        x_min=-8.0,
        x_max=8.0,
        nx=64,
        y_min=-8.0,
        y_max=8.0,
        ny=64,
        t_min=0.0,
        t_max=0.2,
        dt=0.004,
        hbar=1.0,
        mass=1.0,
        boundary="periodic",
        potential_type="harmonic",
        omega=0.25,
        packet_type="gaussian",
        sigma=0.9,
        x0=-1.5,
        y0=0.5,
        k0x=1.0,
        k0y=0.0,
    )

    assert result.dimension == 2
    assert result.y is not None
    assert result.ky is not None
    assert result.psi.shape == (len(result.t), len(result.y), len(result.x))
    assert result.spectrum_power.shape == (len(result.y), len(result.x))
    assert np.all(np.isfinite(result.magnitude))
    assert np.all(np.isfinite(result.invariants["energy"]))
    assert abs(result.magnitudes["norm_drift_rel"]) < 2e-2


def test_schrodinger_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError):
        solve_schrodinger_td(
            dimension=3,
            x_min=-5.0,
            x_max=5.0,
            nx=64,
        )
