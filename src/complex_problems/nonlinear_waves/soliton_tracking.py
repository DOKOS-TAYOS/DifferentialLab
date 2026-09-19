"""Bounded, diagnostic tracking of fixed-shape KdV soliton profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class TrackedSolitonCenters:
    """Unwrapped center tracks, fit diagnostics, and prediction fallbacks."""

    centers: np.ndarray
    anchor_indices: np.ndarray
    used_fallback: np.ndarray
    prediction_data_rms: np.ndarray
    fitted_data_rms: np.ndarray


_PREDICTION_REGULARIZATION = 0.05
"""Weak, fixed tracking prior that resolves nearly equivalent periodic fits."""

_DATA_RMS_TOLERANCE = 1e-10
"""Relative numerical allowance when comparing fitted and predicted data RMS."""


def periodic_soliton_profile(
    x: np.ndarray,
    *,
    center: float,
    amplitude: float,
    inverse_width: float,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    """Evaluate a fixed-shape soliton using periodic minimum-image distance."""
    length = x_max - x_min
    delta = ((x - center + length / 2.0) % length) - length / 2.0
    return amplitude / np.cosh(inverse_width * delta) ** 2


def track_kdv_soliton_centers(
    x: np.ndarray,
    t: np.ndarray,
    numerical: np.ndarray,
    *,
    amplitudes: tuple[float, ...],
    inverse_widths: tuple[float, ...],
    initial_centers: tuple[float, ...],
    speeds: tuple[float, ...],
    x_min: float,
    x_max: float,
    max_anchor_frames: int = 240,
    max_spatial_samples: int = 256,
    max_nfev: int = 25,
) -> TrackedSolitonCenters:
    """Fit continuous unwrapped fixed-shape soliton centers to KdV history.

    The first frame is retained exactly from the supplied initial-condition metadata.
    Later anchors are jointly fitted from a speed-based prediction, then interpolated
    across every stored time. Each fit is confined both to a profile-scale local step
    and to a cumulative phase-displacement envelope around free propagation. A failed
    or non-improving fit uses the finite bounded prediction instead.
    """
    if numerical.shape != (len(t), len(x)):
        raise ValueError("numerical must have shape (len(t), len(x)).")
    n_solitons = len(amplitudes)
    if not n_solitons or not all(
        len(values) == n_solitons for values in (inverse_widths, initial_centers, speeds)
    ):
        raise ValueError("Soliton parameter sequences must be non-empty and have equal length.")
    if (
        x_max <= x_min
        or len(x) == 0
        or len(t) == 0
        or max_anchor_frames < 2
        or max_spatial_samples < 1
    ):
        raise ValueError("Invalid tracking grid or bounded-fit settings.")

    anchor_count = min(len(t), max_anchor_frames)
    anchor_indices = np.unique(np.linspace(0, len(t) - 1, anchor_count, dtype=int))
    sample_count = min(len(x), max_spatial_samples)
    sample_indices = np.unique(np.linspace(0, len(x) - 1, sample_count, dtype=int))
    sample_x = x[sample_indices]
    length = x_max - x_min
    amplitudes_array = np.asarray(amplitudes, dtype=float)
    widths_array = np.asarray(inverse_widths, dtype=float)
    speeds_array = np.asarray(speeds, dtype=float)
    initial_centers_array = np.asarray(initial_centers, dtype=float)
    if not (
        np.all(np.isfinite(amplitudes_array))
        and np.all(np.isfinite(widths_array))
        and np.all(widths_array > 0.0)
        and np.all(np.isfinite(speeds_array))
        and np.all(np.isfinite(initial_centers_array))
    ):
        raise ValueError("Soliton parameters must be finite and inverse widths positive.")

    dx = length / len(x)
    local_radius = np.minimum(
        length / 16.0,
        np.maximum(2.0 / widths_array, 4.0 * dx),
    )
    width_scale = np.maximum(1.0 / widths_array, dx)
    anchors = np.empty((len(anchor_indices), n_solitons), dtype=float)
    anchors[0] = initial_centers_array
    fallback = np.zeros(len(anchor_indices), dtype=bool)
    prediction_data_rms = np.full(len(anchor_indices), np.nan, dtype=float)
    fitted_data_rms = np.full(len(anchor_indices), np.nan, dtype=float)

    def data_residual(centers: np.ndarray, target: np.ndarray) -> np.ndarray:
        model = np.zeros_like(sample_x, dtype=float)
        for amplitude, width, center in zip(amplitudes_array, widths_array, centers, strict=True):
            model += periodic_soliton_profile(
                sample_x,
                center=float(center),
                amplitude=float(amplitude),
                inverse_width=float(width),
                x_min=x_min,
                x_max=x_max,
            )
        return model - target

    def combined_residual(
        centers: np.ndarray,
        target: np.ndarray,
        prediction: np.ndarray,
    ) -> np.ndarray:
        """Combine sampled field residual with a weak sequential-prediction prior."""
        prediction_prior = (
            np.sqrt(len(sample_x))
            * _PREDICTION_REGULARIZATION
            * (centers - prediction)
            / width_scale
        )
        return np.concatenate((data_residual(centers, target), prediction_prior))

    def rms(residual_values: np.ndarray) -> float:
        """Return a finite data-only RMS or NaN for an invalid residual."""
        if not np.all(np.isfinite(residual_values)):
            return float("nan")
        return float(np.sqrt(np.mean(np.square(residual_values))))

    initial_target = np.asarray(numerical[anchor_indices[0], sample_indices], dtype=float)
    initial_rms = rms(data_residual(anchors[0], initial_target))
    prediction_data_rms[0] = initial_rms
    fitted_data_rms[0] = initial_rms

    for anchor_position in range(1, len(anchor_indices)):
        previous_index = anchor_indices[anchor_position - 1]
        frame_index = anchor_indices[anchor_position]
        delta_t = float(t[frame_index] - t[previous_index])
        prediction = anchors[anchor_position - 1] + speeds_array * delta_t
        elapsed = float(t[frame_index] - t[anchor_indices[0]])
        free_center = initial_centers_array + speeds_array * elapsed
        envelope_lower = free_center - length / 4.0
        envelope_upper = free_center + length / 4.0
        lower = np.maximum(prediction - local_radius, envelope_lower)
        upper = np.minimum(prediction + local_radius, envelope_upper)
        target = np.asarray(numerical[frame_index, sample_indices], dtype=float)
        prediction_data_rms[anchor_position] = rms(data_residual(prediction, target))

        if not np.all(lower <= upper):
            anchors[anchor_position] = np.clip(prediction, envelope_lower, envelope_upper)
            fallback[anchor_position] = True
            continue

        start = np.clip(prediction, lower, upper)
        try:
            fit = least_squares(
                combined_residual,
                start,
                args=(target, prediction),
                bounds=(lower, upper),
                loss="soft_l1",
                max_nfev=max_nfev,
            )
            fitted = np.asarray(fit.x, dtype=float)
            candidate = prediction + ((fitted - prediction + length / 2.0) % length) - length / 2.0
            candidate_rms = rms(data_residual(candidate, target))
            fitted_data_rms[anchor_position] = candidate_rms
            fits_identity_bounds = bool(
                np.all(candidate >= lower)
                and np.all(candidate <= upper)
                and np.all(candidate >= envelope_lower)
                and np.all(candidate <= envelope_upper)
            )
            rms_tolerance = _DATA_RMS_TOLERANCE * max(1.0, prediction_data_rms[anchor_position])
            if not (
                fit.success
                and np.all(np.isfinite(candidate))
                and fits_identity_bounds
                and np.isfinite(prediction_data_rms[anchor_position])
                and np.isfinite(candidate_rms)
                and candidate_rms <= prediction_data_rms[anchor_position] + rms_tolerance
            ):
                raise ValueError("Center fit did not meet bounded data-fit acceptance criteria.")
            anchors[anchor_position] = candidate
        except (ValueError, FloatingPointError):
            anchors[anchor_position] = start
            fallback[anchor_position] = True

    tracks = np.empty((len(t), n_solitons), dtype=float)
    for soliton_index in range(n_solitons):
        tracks[:, soliton_index] = np.interp(
            t,
            t[anchor_indices],
            anchors[:, soliton_index],
        )
    return TrackedSolitonCenters(
        centers=tracks,
        anchor_indices=anchor_indices,
        used_fallback=fallback,
        prediction_data_rms=prediction_data_rms,
        fitted_data_rms=fitted_data_rms,
    )
