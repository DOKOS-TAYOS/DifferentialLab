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


def test_nonlinear_waves_rejects_unknown_model_type() -> None:
    with pytest.raises(ValueError):
        solve_nonlinear_waves(
            model_type="unknown",
            x_min=-5.0,
            x_max=5.0,
            nx=64,
        )
