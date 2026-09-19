"""Pure regression tests for fixed-shape KdV soliton-center tracking."""

from __future__ import annotations

import numpy as np

from complex_problems.nonlinear_waves.soliton_tracking import (
    periodic_soliton_profile,
    track_kdv_soliton_centers,
)
from complex_problems.nonlinear_waves.solver import solve_nonlinear_waves


def _history(
    x: np.ndarray,
    centers: np.ndarray,
    amplitudes: tuple[float, ...],
    widths: tuple[float, ...],
) -> np.ndarray:
    return np.array(
        [
            sum(
                (
                    periodic_soliton_profile(
                        x,
                        center=float(center),
                        amplitude=amplitude,
                        inverse_width=width,
                        x_min=-20.0,
                        x_max=20.0,
                    )
                    for center, amplitude, width in zip(row, amplitudes, widths, strict=True)
                ),
                start=np.zeros_like(x),
            )
            for row in centers
        ]
    )


def test_tracks_one_soliton_with_smooth_phase_displacement() -> None:
    x = np.linspace(-20.0, 20.0, 512, endpoint=False)
    t = np.linspace(0.0, 4.0, 61)
    expected = -8.0 + 2.0 * t + 0.35 * np.sin(np.pi * t / 4.0)
    tracked = track_kdv_soliton_centers(
        x,
        t,
        _history(x, expected[:, None], (1.0,), (0.8,)),
        amplitudes=(1.0,),
        inverse_widths=(0.8,),
        initial_centers=(-8.0,),
        speeds=(2.0,),
        x_min=-20.0,
        x_max=20.0,
    )
    np.testing.assert_allclose(tracked.centers[:, 0], expected, atol=0.025)


def test_tracks_periodic_crossing_in_unwrapped_coordinates() -> None:
    x = np.linspace(-20.0, 20.0, 512, endpoint=False)
    t = np.linspace(0.0, 2.0, 41)
    expected = 18.0 + 2.0 * t
    tracked = track_kdv_soliton_centers(
        x,
        t,
        _history(x, expected[:, None], (1.0,), (1.0,)),
        amplitudes=(1.0,),
        inverse_widths=(1.0,),
        initial_centers=(18.0,),
        speeds=(2.0,),
        x_min=-20.0,
        x_max=20.0,
    )
    np.testing.assert_allclose(tracked.centers[:, 0], expected, atol=0.025)
    assert np.max(np.abs(np.diff(tracked.centers[:, 0]))) < 0.2


def test_tracks_two_nonconstant_soliton_trajectories() -> None:
    x = np.linspace(-20.0, 20.0, 512, endpoint=False)
    t = np.linspace(0.0, 3.0, 61)
    expected = np.column_stack(
        (
            -10.0 + 3.0 * t + 0.25 * np.sin(np.pi * t / 3.0),
            4.0 + 0.8 * t - 0.2 * np.sin(np.pi * t / 3.0),
        )
    )
    tracked = track_kdv_soliton_centers(
        x,
        t,
        _history(x, expected, (1.4, 0.6), (0.9, 0.6)),
        amplitudes=(1.4, 0.6),
        inverse_widths=(0.9, 0.6),
        initial_centers=(-10.0, 4.0),
        speeds=(3.0, 0.8),
        x_min=-20.0,
        x_max=20.0,
    )
    np.testing.assert_allclose(tracked.centers, expected, atol=0.04)
    midpoint = len(t) // 2
    assert abs(tracked.centers[midpoint, 0] - (-10.0 + 3.0 * t[midpoint])) > 0.1


def test_invalid_anchor_uses_finite_prediction_fallback() -> None:
    x = np.linspace(-20.0, 20.0, 64, endpoint=False)
    t = np.array([0.0, 1.0])
    numerical = np.zeros((2, len(x)))
    numerical[1] = np.nan
    tracked = track_kdv_soliton_centers(
        x,
        t,
        numerical,
        amplitudes=(1.0,),
        inverse_widths=(1.0,),
        initial_centers=(0.0,),
        speeds=(2.0,),
        x_min=-20.0,
        x_max=20.0,
    )
    assert tracked.used_fallback.tolist() == [False, True]
    np.testing.assert_allclose(tracked.centers[:, 0], [0.0, 2.0])


def test_representative_kdv_train_has_finite_continuous_phase_displacement() -> None:
    result = solve_nonlinear_waves(
        model_type="kdv",
        x_min=-20.0,
        x_max=20.0,
        nx=512,
        t_min=0.0,
        t_max=8.0,
        dt=0.002,
        profile="kdv_soliton_train",
        soliton_amplitudes=[1.2, 0.5, 3.0, 1.0],
        soliton_centers=[-8.0, -2.0, -1.0, -3.0],
        c=0.0,
        alpha=6.0,
        beta_disp=1.0,
    )
    metadata = result.metadata
    tracked = track_kdv_soliton_centers(
        result.x,
        result.t,
        result.field.real,
        amplitudes=tuple(metadata["soliton_amplitudes"]),
        inverse_widths=tuple(metadata["soliton_inverse_widths"]),
        initial_centers=tuple(metadata["soliton_centers"]),
        speeds=tuple(metadata["soliton_speeds"]),
        x_min=float(metadata["x_min"]),
        x_max=float(metadata["x_max"]),
    )
    free_final = np.asarray(metadata["soliton_centers"]) + np.asarray(
        metadata["soliton_speeds"]
    ) * (result.t[-1] - result.t[0])
    assert len(tracked.anchor_indices) == 240
    assert np.all(np.isfinite(tracked.centers))
    assert np.max(np.abs(np.diff(tracked.centers, axis=0))) < 1.0
    assert np.max(np.abs(tracked.centers[-1] - free_final)) > 0.1
