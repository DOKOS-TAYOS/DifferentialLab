"""Tests for steady and transient pipe-flow solvers."""

from __future__ import annotations

import numpy as np
import pytest

from complex_problems.pipe_flow.solver import solve_pipe_flow


def test_steady_pipe_flow_matches_pressure_drop_direction() -> None:
    result = solve_pipe_flow(
        model_type="steady",
        length=10.0,
        nx=160,
        profile="constant",
        d0=0.05,
        d_in=0.05,
        d_out=0.05,
        rho=1000.0,
        mu=1.0e-3,
        roughness=1.0e-5,
        friction_model="auto",
        p_in=2.0e5,
        p_out=1.9e5,
    )

    p = result.pressure[0]
    assert np.all(np.diff(p) <= 1e-8)
    assert result.flow_rate_mean[0] > 0.0
    assert abs((p[0] - p[-1]) - (2.0e5 - 1.9e5)) / (2.0e5 - 1.9e5) < 1e-2


def test_transient_pipe_flow_runs_with_finite_fields() -> None:
    result = solve_pipe_flow(
        model_type="transient",
        length=20.0,
        nx=180,
        profile="sinusoidal",
        d0=0.06,
        profile_amplitude=0.05,
        profile_waves=1.0,
        rho=1000.0,
        mu=1.0e-3,
        roughness=1.0e-5,
        friction_model="auto",
        p_out=1.9e5,
        p_base=2.0e5,
        p_amp=1.5e3,
        p_freq_hz=1.0,
        wave_speed=150.0,
        damping=0.2,
        t_max=0.15,
        dt=2e-4,
        sample_every=10,
    )

    assert result.pressure.shape[0] == len(result.t)
    assert result.velocity.shape[0] == len(result.t)
    assert np.all(np.isfinite(result.pressure))
    assert np.all(np.isfinite(result.velocity))
    assert result.magnitudes["cfl"] < 0.95


def test_transient_pipe_flow_rejects_unstable_cfl() -> None:
    with pytest.raises(ValueError):
        solve_pipe_flow(
            model_type="transient",
            length=2.0,
            nx=40,
            profile="constant",
            d0=0.05,
            p_out=1.0e5,
            p_base=1.1e5,
            p_amp=100.0,
            p_freq_hz=1.0,
            wave_speed=500.0,
            damping=0.1,
            t_max=0.05,
            dt=0.01,
        )
