"""Tests for nonlinear wave solvers (NLSE and KdV)."""

from __future__ import annotations

import numpy as np
import pytest

from complex_problems.nonlinear_waves.model import (
    build_kdv_soliton_profile,
    build_kdv_soliton_train,
    build_periodic_grid,
    compute_kdv_invariants,
)
from complex_problems.nonlinear_waves.solver import _simulate_kdv, solve_nonlinear_waves


def test_kdv_soliton_characteristics_and_peak() -> None:
    x = np.linspace(-10.0, 10.0, 1001)
    profile, characteristics = build_kdv_soliton_profile(
        x, amplitude=1.0, center=1.25, c=0.0, alpha=6.0, beta_disp=1.0
    )
    assert characteristics.inverse_width == pytest.approx(np.sqrt(0.5))
    assert characteristics.speed == pytest.approx(2.0)
    assert profile[np.argmin(abs(x - 1.25))] == pytest.approx(1.0, abs=1e-4)


def test_kdv_soliton_train_is_sum_of_profiles() -> None:
    x = np.linspace(-20.0, 20.0, 512, endpoint=False)
    train, characteristics = build_kdv_soliton_train(
        x, amplitudes=[1.2, 0.5], centers=[-8.0, -2.0], c=0.0, alpha=6.0, beta_disp=1.0
    )
    expected = sum(
        build_kdv_soliton_profile(
            x,
            amplitude=item.amplitude,
            center=item.center,
            c=0.0,
            alpha=6.0,
            beta_disp=1.0,
        )[0]
        for item in characteristics
    )
    np.testing.assert_allclose(train, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beta_disp": 0.0, "alpha": 6.0, "amplitude": 1.0},
        {"beta_disp": 1.0, "alpha": 0.0, "amplitude": 1.0},
        {"beta_disp": 1.0, "alpha": 6.0, "amplitude": -1.0},
    ],
)
def test_kdv_soliton_rejects_non_real_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="alpha \\* amplitude / beta_disp > 0"):
        build_kdv_soliton_profile(np.array([0.0]), center=0.0, c=0.0, **kwargs)


def test_kdv_soliton_train_rejects_mismatched_lists() -> None:
    with pytest.raises(ValueError, match="equal length"):
        build_kdv_soliton_train(
            np.array([0.0]),
            amplitudes=[1.0, 0.5],
            centers=[0.0],
            c=0.0,
            alpha=6.0,
            beta_disp=1.0,
        )


def test_kdv_soliton_propagates_with_expected_shape_and_direction() -> None:
    result = solve_nonlinear_waves(
        model_type="kdv",
        x_min=-20.0,
        x_max=20.0,
        nx=512,
        t_max=1.0,
        dt=0.002,
        profile="kdv_soliton",
        amplitude=1.0,
        center=-5.0,
        c=0.0,
        alpha=6.0,
        beta_disp=1.0,
    )
    expected = build_kdv_soliton_profile(
        result.x, amplitude=1.0, center=-3.0, c=0.0, alpha=6.0, beta_disp=1.0
    )[0]
    relative_l2 = np.linalg.norm(result.field[-1] - expected) / np.linalg.norm(expected)
    assert relative_l2 < 2e-2
    assert result.x[np.argmax(result.field[-1])] > result.x[np.argmax(result.field[0])]
    assert result.metadata["soliton_count"] == 1


def test_kdv_train_solver_exposes_reproducibility_metadata() -> None:
    result = solve_nonlinear_waves(
        model_type="kdv",
        x_min=-20.0,
        x_max=20.0,
        nx=128,
        t_max=0.1,
        dt=0.01,
        profile="kdv_soliton_train",
        soliton_amplitudes=[1.2, 0.5],
        soliton_centers=[-8.0, -2.0],
        c=0.0,
        alpha=6.0,
        beta_disp=1.0,
    )
    assert np.all(np.isfinite(result.field))
    assert result.metadata["soliton_count"] == 2
    assert result.metadata["soliton_speeds"] == pytest.approx([2.4, 1.0])
    assert np.max(abs(result.field[0])) > 0.1


def test_kdv_hamiltonian_uses_configured_coefficients() -> None:
    nx = 128
    x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
    dx = float(x[1] - x[0])
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    u = 0.4 * np.sin(x) + 0.1 * np.cos(2.0 * x)
    c, alpha, beta = 1.3, 2.5, 0.7
    ux = np.fft.ifft(1j * k * np.fft.fft(u)).real
    expected = float(np.sum(beta * ux**2 / 2.0 - alpha * u**3 / 6.0 - c * u**2 / 2.0) * dx)
    actual = compute_kdv_invariants(u, dx=dx, k=k, c=c, alpha=alpha, beta_disp=beta)[2]
    assert actual == pytest.approx(expected)


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
        profile="kdv_soliton_train",
        soliton_amplitudes=[1.2, 0.5],
        soliton_centers=[-8.0, -2.0],
        c=0.0,
        alpha=6.0,
        beta_disp=1.0,
    )

    assert np.all(np.isfinite(result.field))
    assert np.all(np.isfinite(result.invariants["mass"]))
    assert abs(result.magnitudes["mass_drift_rel"]) < 1e-6


def test_kdv_three_soliton_train_stays_finite_and_bounded() -> None:
    result = solve_nonlinear_waves(
        model_type="kdv",
        x_min=-20.0,
        x_max=20.0,
        nx=384,
        t_max=7.5,
        dt=0.002,
        profile="kdv_soliton_train",
        soliton_amplitudes=[3.0, 1.0, 1.0],
        soliton_centers=[0.0, 6.0, 12.0],
        c=0.0,
        alpha=6.0,
        beta_disp=1.0,
        store_every=4,
    )

    assert np.all(np.isfinite(result.field))
    assert all(np.all(np.isfinite(values)) for values in result.invariants.values())
    assert np.all(np.isfinite(result.spectrum_power))
    assert all(np.isfinite(value) for value in result.magnitudes.values())
    assert np.max(np.abs(result.field)) < 10.0


def test_kdv_rejects_non_finite_internal_state() -> None:
    x, dx, k = build_periodic_grid(-4.0, 4.0, 32)
    t = np.array([0.0, 0.01])
    stored_steps = np.array([0, 1])

    with pytest.raises(RuntimeError, match="KdV integration became numerically unstable"):
        _simulate_kdv(
            x=x,
            t=t,
            stored_steps=stored_steps,
            dx=dx,
            k=k,
            u0=np.full(len(x), 1e200),
            c=0.0,
            alpha=6.0,
            beta_disp=1.0,
        )


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


@pytest.mark.parametrize("model_type", ["nlse", "kdv"])
def test_spectrum_power_history_is_lazy_cached_for_both_models(model_type: str) -> None:
    kwargs = {
        "model_type": model_type,
        "x_min": -10.0,
        "x_max": 10.0,
        "nx": 64,
        "t_min": 0.0,
        "t_max": 0.05,
        "dt": 0.01,
        "profile": "sech",
        "amplitude": 1.0 if model_type == "nlse" else 0.4,
        "sigma": 1.0,
        "center": 0.0,
        "beta2": 1.0,
        "gamma": 1.0,
        "initial_phase_k": 0.0,
        "c": 0.0,
        "alpha": 4.0,
        "beta_disp": 1.0,
    }
    result = solve_nonlinear_waves(**kwargs)

    assert result._spectrum_power_history_cache is None
    spectrum_history = result.spectrum_power_history
    assert spectrum_history.shape == (len(result.t), len(result.k))
    assert spectrum_history.dtype == np.float32
    assert result.spectrum_power_history is spectrum_history
    expected = np.stack(
        [np.abs(np.fft.fftshift(np.fft.fft(frame))) ** 2 for frame in result.field]
    ).astype(np.float32)
    np.testing.assert_allclose(spectrum_history, expected, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(spectrum_history[-1], result.spectrum_power, rtol=2e-6, atol=1e-5)


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
