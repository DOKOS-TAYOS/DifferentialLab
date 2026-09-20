"""Regression tests for observation-driven KdV soliton-center tracking."""

from __future__ import annotations

import numpy as np

from complex_problems.nonlinear_waves.soliton_tracking import (
    TrackedSolitonCenters,
    periodic_soliton_profile,
    track_kdv_soliton_centers,
)
from complex_problems.nonlinear_waves.solver import solve_nonlinear_waves

X_MIN = -20.0
X_MAX = 20.0


def _history(
    x: np.ndarray,
    centers: np.ndarray,
    amplitudes: tuple[float, ...],
    widths: tuple[float, ...],
) -> np.ndarray:
    """Construct deterministic periodic fixed-shape soliton frames."""
    return np.array(
        [
            sum(
                (
                    periodic_soliton_profile(
                        x,
                        center=float(center),
                        amplitude=amplitude,
                        inverse_width=width,
                        x_min=X_MIN,
                        x_max=X_MAX,
                    )
                    for center, amplitude, width in zip(row, amplitudes, widths, strict=True)
                ),
                start=np.zeros_like(x),
            )
            for row in centers
        ]
    )


def _evolving_history(
    x: np.ndarray,
    centers: np.ndarray,
    amplitudes: np.ndarray,
    widths: tuple[float, ...],
) -> np.ndarray:
    """Construct periodic frames with independently evolving peak heights."""
    return np.array(
        [
            sum(
                (
                    periodic_soliton_profile(
                        x,
                        center=float(center),
                        amplitude=float(amplitude),
                        inverse_width=width,
                        x_min=X_MIN,
                        x_max=X_MAX,
                    )
                    for center, amplitude, width in zip(
                        center_row, amplitude_row, widths, strict=True
                    )
                ),
                start=np.zeros_like(x),
            )
            for center_row, amplitude_row in zip(centers, amplitudes, strict=True)
        ]
    )


def _track(
    x: np.ndarray,
    t: np.ndarray,
    centers: np.ndarray,
    amplitudes: tuple[float, ...],
    widths: tuple[float, ...],
    initial_centers: tuple[float, ...],
    speeds: tuple[float, ...],
) -> TrackedSolitonCenters:
    """Track a synthetic history with the common periodic domain."""
    return track_kdv_soliton_centers(
        x,
        t,
        _history(x, centers, amplitudes, widths),
        amplitudes=amplitudes,
        inverse_widths=widths,
        initial_centers=initial_centers,
        speeds=speeds,
        x_min=X_MIN,
        x_max=X_MAX,
    )


def _fixed_shape_rms(
    x: np.ndarray,
    centers: np.ndarray,
    numerical: np.ndarray,
    amplitudes: tuple[float, ...],
    widths: tuple[float, ...],
) -> float:
    """Return the aggregate fixed-shape reconstruction RMS on supplied frames."""
    return float(np.sqrt(np.mean(np.square(_history(x, centers, amplitudes, widths) - numerical))))


def test_tracks_one_soliton_with_nonlinear_phase_shift() -> None:
    x = np.linspace(X_MIN, X_MAX, 512, endpoint=False)
    t = np.linspace(0.0, 4.0, 81)
    phase = 0.8 * (3.0 * (t / 4.0) ** 2 - 2.0 * (t / 4.0) ** 3)
    expected = -8.0 + 2.0 * t + phase
    tracked = _track(x, t, expected[:, None], (1.0,), (0.8,), (-8.0,), (2.0,))
    np.testing.assert_allclose(tracked.centers[:, 0], expected, atol=0.025)
    assert np.all(tracked.observed[:, 0])


def test_tracks_periodic_crossing_without_an_unwrapped_jump() -> None:
    x = np.linspace(X_MIN, X_MAX, 512, endpoint=False)
    t = np.linspace(0.0, 2.0, 41)
    expected = 18.0 + 2.0 * t
    tracked = _track(x, t, expected[:, None], (1.0,), (1.0,), (18.0,), (2.0,))
    np.testing.assert_allclose(tracked.centers[:, 0], expected, atol=0.025)
    assert np.all(tracked.observed[:, 0])
    assert np.max(np.abs(np.diff(tracked.centers[:, 0]))) < 0.2


def test_tracks_negative_polarity_soliton_trough() -> None:
    x = np.linspace(X_MIN, X_MAX, 512, endpoint=False)
    t = np.linspace(0.0, 3.0, 61)
    expected = -9.0 + 1.5 * t + 0.3 * np.sin(np.pi * t / 3.0)
    tracked = _track(x, t, expected[:, None], (-1.1,), (0.9,), (-9.0,), (1.5,))
    np.testing.assert_allclose(tracked.centers[:, 0], expected, atol=0.03)
    assert np.all(tracked.observed[:, 0])


def test_separated_similar_amplitudes_keep_their_identities() -> None:
    x = np.linspace(X_MIN, X_MAX, 512, endpoint=False)
    t = np.linspace(0.0, 3.0, 61)
    expected = np.column_stack((-12.0 + 1.4 * t, 5.0 + 0.45 * t))
    tracked = _track(
        x,
        t,
        expected,
        (1.2, 1.0),
        (0.9, 0.85),
        (-12.0, 5.0),
        (1.4, 0.45),
    )
    np.testing.assert_allclose(tracked.centers, expected, atol=0.035)
    assert np.all(tracked.observed)


def test_merge_is_ambiguous_then_post_interaction_peaks_recover_phase() -> None:
    x = np.linspace(X_MIN, X_MAX, 512, endpoint=False)
    t = np.linspace(0.0, 7.0, 141)
    free = np.column_stack((-6.0 + 3.0 * t, 4.0 + 0.5 * t))
    phase_progress = np.clip((t - 3.5) / 1.0, 0.0, 1.0)
    phase_progress = 3.0 * phase_progress**2 - 2.0 * phase_progress**3
    expected = free + np.column_stack((phase_progress, -phase_progress))
    tracked = _track(
        x,
        t,
        expected,
        (1.2, 1.0),
        (1.0, 0.9),
        (-6.0, 4.0),
        (3.0, 0.5),
    )
    collision = np.flatnonzero((t > 3.8) & (t < 4.05))
    assert not np.any(tracked.observed[collision])
    np.testing.assert_allclose(tracked.centers[-1], expected[-1], atol=0.06)
    assert np.all(tracked.observed[-10:])
    assert tracked.centers[-1, 0] - free[-1, 0] > 0.8
    assert tracked.centers[-1, 1] - free[-1, 1] < -0.8


def test_distractor_pulse_cannot_steal_a_tracked_identity() -> None:
    x = np.linspace(X_MIN, X_MAX, 512, endpoint=False)
    t = np.linspace(0.0, 2.0, 61)
    free = -8.0 + 2.0 * t
    numerical = _history(x, free[:, None], (1.0,), (1.0,))
    numerical += periodic_soliton_profile(
        x,
        center=12.0,
        amplitude=1.45,
        inverse_width=1.0,
        x_min=X_MIN,
        x_max=X_MAX,
    )
    tracked = track_kdv_soliton_centers(
        x,
        t,
        numerical,
        amplitudes=(1.0,),
        inverse_widths=(1.0,),
        initial_centers=(-8.0,),
        speeds=(2.0,),
        x_min=X_MIN,
        x_max=X_MAX,
    )
    np.testing.assert_allclose(tracked.centers[:, 0], free, atol=0.03)


def test_reliable_phase_displacement_can_exceed_one_quarter_domain() -> None:
    x = np.linspace(X_MIN, X_MAX, 512, endpoint=False)
    t = np.linspace(0.0, 8.0, 161)
    phase = 12.0 * (3.0 * (t / 8.0) ** 2 - 2.0 * (t / 8.0) ** 3)
    expected = -16.0 + 1.0 * t + phase
    tracked = _track(x, t, expected[:, None], (1.0,), (0.9,), (-16.0,), (1.0,))
    np.testing.assert_allclose(tracked.centers[:, 0], expected, atol=0.04)
    assert tracked.centers[-1, 0] - (-16.0 + 8.0) > (X_MAX - X_MIN) / 4.0


def test_reacquires_after_material_amplitude_evolution() -> None:
    x = np.linspace(X_MIN, X_MAX, 512, endpoint=False)
    t = np.linspace(0.0, 6.0, 121)
    expected = -10.0 + t + np.where(t >= 3.0, 2.0, 0.0)
    numerical = _evolving_history(
        x,
        expected[:, None],
        np.where((t < 2.0)[:, None], 1.0, np.where((t < 3.0)[:, None], 0.0, 2.0)),
        (1.0,),
    )
    tracked = track_kdv_soliton_centers(
        x,
        t,
        numerical,
        amplitudes=(1.0,),
        inverse_widths=(1.0,),
        initial_centers=(-10.0,),
        speeds=(1.0,),
        x_min=X_MIN,
        x_max=X_MAX,
    )
    assert not np.any(tracked.observed[(t >= 2.0) & (t < 3.0), 0])
    assert np.any(tracked.observed[t >= 3.0, 0])
    np.testing.assert_allclose(tracked.centers[-1, 0], expected[-1], atol=0.04)
    assert np.max(np.abs(np.diff(tracked.centers[:, 0]))) < 0.2


def test_amplitude_rank_protects_reacquired_identities() -> None:
    x = np.linspace(X_MIN, X_MAX, 512, endpoint=False)
    t = np.linspace(0.0, 6.0, 121)
    free = np.column_stack((-12.0 + 1.2 * t, -2.0 + 0.8 * t, 9.0 + 0.4 * t))
    expected = free + np.where((t >= 3.0)[:, None], (1.5, -1.0, 0.7), 0.0)
    numerical = _evolving_history(
        x,
        expected,
        np.where(
            (t < 2.0)[:, None],
            (3.0, 2.0, 1.0),
            np.where((t < 3.0)[:, None], 0.0, (5.0, 3.5, 1.7)),
        ),
        (1.0, 1.0, 1.0),
    )
    tracked = track_kdv_soliton_centers(
        x,
        t,
        numerical,
        amplitudes=(3.0, 2.0, 1.0),
        inverse_widths=(1.0, 1.0, 1.0),
        initial_centers=(-12.0, -2.0, 9.0),
        speeds=(1.2, 0.8, 0.4),
        x_min=X_MIN,
        x_max=X_MAX,
    )
    assert np.all(np.any(tracked.observed[t >= 3.0], axis=0))
    np.testing.assert_allclose(tracked.centers[-1], expected[-1], atol=0.05)


def test_distorted_amplitude_rank_does_not_block_clear_continuity() -> None:
    x = np.linspace(X_MIN, X_MAX, 512, endpoint=False)
    t = np.linspace(0.0, 5.0, 101)
    expected = np.column_stack((-12.0 + t, 0.0 + 0.7 * t, 10.0 + 0.4 * t))
    numerical = _evolving_history(
        x,
        expected,
        np.where(t[:, None] < 2.5, (3.0, 2.0, 1.0), (2.2, 5.0, 1.3)),
        (1.0, 1.0, 1.0),
    )
    tracked = track_kdv_soliton_centers(
        x,
        t,
        numerical,
        amplitudes=(3.0, 2.0, 1.0),
        inverse_widths=(1.0, 1.0, 1.0),
        initial_centers=(-12.0, 0.0, 10.0),
        speeds=(1.0, 0.7, 0.4),
        x_min=X_MIN,
        x_max=X_MAX,
    )
    np.testing.assert_allclose(tracked.centers[-1], expected[-1], atol=0.05)
    assert np.all(tracked.observed[-10:])


def test_near_equal_candidates_remain_unassigned() -> None:
    x = np.linspace(X_MIN, X_MAX, 512, endpoint=False)
    t = np.linspace(0.0, 2.0, 21)
    numerical = np.zeros((len(t), len(x)))
    numerical[0] = periodic_soliton_profile(
        x,
        center=-5.0,
        amplitude=1.0,
        inverse_width=1.0,
        x_min=X_MIN,
        x_max=X_MAX,
    )
    for frame in range(1, len(t)):
        numerical[frame] = sum(
            (
                periodic_soliton_profile(
                    x,
                    center=center,
                    amplitude=1.0,
                    inverse_width=1.0,
                    x_min=X_MIN,
                    x_max=X_MAX,
                )
                for center in (-7.0, -3.0)
            ),
            start=np.zeros_like(x),
        )
    tracked = track_kdv_soliton_centers(
        x,
        t,
        numerical,
        amplitudes=(1.0,),
        inverse_widths=(1.0,),
        initial_centers=(-5.0,),
        speeds=(0.0,),
        x_min=X_MIN,
        x_max=X_MAX,
    )
    assert not np.any(tracked.observed[1:, 0])


def test_negative_soliton_ranks_reacquire_after_height_evolution() -> None:
    x = np.linspace(X_MIN, X_MAX, 512, endpoint=False)
    t = np.linspace(0.0, 5.0, 101)
    free = np.column_stack((-12.0 + 1.1 * t, 6.0 + 0.5 * t))
    expected = free + np.where((t >= 3.0)[:, None], (1.0, -0.8), 0.0)
    numerical = _evolving_history(
        x,
        expected,
        np.where(
            (t < 2.0)[:, None],
            (-3.0, -1.0),
            np.where((t < 3.0)[:, None], 0.0, (-5.0, -2.0)),
        ),
        (1.0, 1.0),
    )
    tracked = track_kdv_soliton_centers(
        x,
        t,
        numerical,
        amplitudes=(-3.0, -1.0),
        inverse_widths=(1.0, 1.0),
        initial_centers=(-12.0, 6.0),
        speeds=(1.1, 0.5),
        x_min=X_MIN,
        x_max=X_MAX,
    )
    assert np.all(np.any(tracked.observed[t >= 3.0], axis=0))
    np.testing.assert_allclose(tracked.centers[-1], expected[-1], atol=0.05)


def test_representative_kdv_train_reports_observational_diagnostics() -> None:
    result = solve_nonlinear_waves(
        model_type="kdv",
        x_min=X_MIN,
        x_max=X_MAX,
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
    free = (
        np.asarray(metadata["soliton_centers"])[None, :]
        + (result.t[:, None] - result.t[0]) * np.asarray(metadata["soliton_speeds"])[None, :]
    )
    anchor_tracks = tracked.centers[tracked.anchor_indices]
    assert len(tracked.anchor_indices) == 240
    assert np.all(np.isfinite(tracked.centers))
    assert np.any(~tracked.observed[1:])
    dx = (X_MAX - X_MIN) / len(result.x)
    for index, width in enumerate(metadata["soliton_inverse_widths"]):
        mask = tracked.observed[:, index]
        periodic_error = (
            anchor_tracks[mask, index] - tracked.observed_centers[mask, index] + 20.0
        ) % 40.0 - 20.0
        assert np.max(np.abs(periodic_error)) < max(1.5 * dx, 0.05 / float(width))
    assert np.max(np.abs(tracked.centers - free)) > 10.0
    amplitudes = tuple(float(value) for value in metadata["soliton_amplitudes"])
    widths = tuple(float(value) for value in metadata["soliton_inverse_widths"])
    free_rms = _fixed_shape_rms(
        result.x,
        free[tracked.anchor_indices],
        result.field.real[tracked.anchor_indices],
        amplitudes,
        widths,
    )
    tracked_rms = _fixed_shape_rms(
        result.x,
        anchor_tracks,
        result.field.real[tracked.anchor_indices],
        amplitudes,
        widths,
    )
    assert tracked_rms < free_rms
    phase = tracked.centers - free
    assert not np.any(np.abs(np.diff(phase, axis=0)) > 20.0)
