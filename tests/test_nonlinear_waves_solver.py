"""Tests for nonlinear wave solvers (NLSE and KdV)."""

from __future__ import annotations

import numpy as np
import pytest

from complex_problems.nonlinear_waves.solver import solve_nonlinear_waves


def test_nlse_split_step_preserves_norm_reasonably() -> None:
    result = solve_nonlinear_waves(
        model_type="nlse",
        x_min=-20.0,
        x_max=20.0,
        nx=256,
        t_min=0.0,
        t_max=1.5,
        dt=0.002,
        profile="sech",
        amplitude=1.0,
        sigma=1.0,
        center=0.0,
        beta2=1.0,
        gamma=1.0,
        initial_phase_k=0.0,
    )

    assert result.model_type == "nlse"
    assert result.phase is not None
    assert result.field.shape == result.magnitude.shape
    assert np.all(np.isfinite(result.magnitude))
    assert abs(result.magnitudes["norm_drift_rel"]) < 1e-2


def test_kdv_pseudospectral_runs_and_mass_drift_stays_small() -> None:
    result = solve_nonlinear_waves(
        model_type="kdv",
        x_min=-25.0,
        x_max=25.0,
        nx=256,
        t_min=0.0,
        t_max=0.5,
        dt=0.0005,
        profile="sech",
        amplitude=0.4,
        sigma=1.2,
        center=0.0,
        c=0.0,
        alpha=4.0,
        beta_disp=1.0,
    )

    assert result.model_type == "kdv"
    assert result.phase is None
    assert result.field.shape == result.magnitude.shape
    assert np.all(np.isfinite(result.field))
    assert abs(result.magnitudes["mass_drift_rel"]) < 5e-2


def test_kdv_ui_defaults_stay_finite_for_long_horizon() -> None:
    result = solve_nonlinear_waves(
        model_type="kdv",
        x_min=-20.0,
        x_max=20.0,
        nx=512,
        t_min=0.0,
        t_max=8.0,
        dt=0.002,
        profile="sech",
        amplitude=1.0,
        sigma=1.0,
        center=0.0,
        c=0.0,
        alpha=6.0,
        beta_disp=1.0,
    )

    assert np.all(np.isfinite(result.field))
    assert np.all(np.isfinite(result.invariants["mass"]))
    assert abs(result.magnitudes["mass_drift_rel"]) < 1e-6


def test_nlse_result_materializes_lazy_caches_on_demand() -> None:
    result = solve_nonlinear_waves(
        model_type="nlse",
        x_min=-10.0,
        x_max=10.0,
        nx=128,
        t_min=0.0,
        t_max=0.1,
        dt=0.01,
        profile="sech",
        amplitude=1.0,
        sigma=1.0,
        center=0.0,
        beta2=1.0,
        gamma=1.0,
        initial_phase_k=0.0,
    )

    assert result._magnitude_cache is None
    assert result._phase_cache is None

    magnitude = result.magnitude
    phase = result.phase

    assert result._magnitude_cache is magnitude
    assert result._phase_cache is phase
    assert result.magnitude is magnitude
    assert result.phase is phase
    np.testing.assert_allclose(magnitude, np.abs(result.field) ** 2)
    np.testing.assert_allclose(phase, np.angle(result.field))


def test_nlse_store_every_keeps_aligned_history_and_final_frame() -> None:
    kwargs = {
        "model_type": "nlse",
        "x_min": -10.0,
        "x_max": 10.0,
        "nx": 64,
        "t_min": 0.0,
        "t_max": 0.1,
        "dt": 0.01,
        "profile": "sech",
        "amplitude": 1.0,
        "sigma": 1.0,
        "center": 0.0,
        "beta2": 1.0,
        "gamma": 1.0,
        "initial_phase_k": 0.0,
    }

    full = solve_nonlinear_waves(**kwargs)
    sampled = solve_nonlinear_waves(**kwargs, store_every=4)

    np.testing.assert_allclose(sampled.t, [0.0, 0.04, 0.08, 0.1])
    assert sampled.field.shape == (len(sampled.t), 64)
    for values in sampled.invariants.values():
        assert values.shape == sampled.t.shape
    np.testing.assert_allclose(sampled.field[-1], full.field[-1])
    np.testing.assert_allclose(sampled.spectrum_power, full.spectrum_power)
    assert sampled.magnitudes["max_intensity"] == pytest.approx(full.magnitudes["max_intensity"])
    assert sampled.metadata["solver_dt"] == pytest.approx(0.01)
    assert sampled.metadata["solver_steps"] == 10
    assert sampled.metadata["store_every"] == 4
    assert sampled.metadata["stored_steps"] == len(sampled.t)


def test_nonlinear_waves_store_every_one_matches_default() -> None:
    kwargs = {
        "model_type": "kdv",
        "x_min": -10.0,
        "x_max": 10.0,
        "nx": 64,
        "t_min": 0.0,
        "t_max": 0.05,
        "dt": 0.01,
        "profile": "sech",
        "amplitude": 0.4,
        "sigma": 1.0,
        "center": 0.0,
        "c": 0.0,
        "alpha": 4.0,
        "beta_disp": 1.0,
    }

    default = solve_nonlinear_waves(**kwargs)
    explicit = solve_nonlinear_waves(**kwargs, store_every=1)

    np.testing.assert_allclose(explicit.t, default.t)
    np.testing.assert_allclose(explicit.field, default.field)
    for key, values in explicit.invariants.items():
        np.testing.assert_allclose(values, default.invariants[key])
    np.testing.assert_allclose(explicit.spectrum_power, default.spectrum_power)


def test_kdv_result_keeps_phase_disabled() -> None:
    result = solve_nonlinear_waves(
        model_type="kdv",
        x_min=-10.0,
        x_max=10.0,
        nx=128,
        t_min=0.0,
        t_max=0.1,
        dt=0.01,
        profile="sech",
        amplitude=0.4,
        sigma=1.0,
        center=0.0,
        c=0.0,
        alpha=4.0,
        beta_disp=1.0,
    )

    assert result._magnitude_cache is None
    assert result.phase is None

    magnitude = result.magnitude
    assert result._magnitude_cache is magnitude
    assert result.magnitude is magnitude
    np.testing.assert_allclose(magnitude, result.field)


def test_nonlinear_waves_rejects_unknown_model_type() -> None:
    with pytest.raises(ValueError):
        solve_nonlinear_waves(
            model_type="unknown",
            x_min=-5.0,
            x_max=5.0,
            nx=64,
        )
