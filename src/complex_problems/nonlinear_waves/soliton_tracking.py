"""Bounded, diagnostic tracking of fixed-shape KdV soliton profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class TrackedSolitonCenters:
    """Unwrapped center tracks and the anchors that required prediction fallback."""

    centers: np.ndarray
    anchor_indices: np.ndarray
    used_fallback: np.ndarray


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
    across every stored time. A failed fit uses that finite prediction instead.
    """
    if numerical.shape != (len(t), len(x)):
        raise ValueError("numerical must have shape (len(t), len(x)).")
    n_solitons = len(amplitudes)
    if not n_solitons or not all(
        len(values) == n_solitons for values in (inverse_widths, initial_centers, speeds)
    ):
        raise ValueError("Soliton parameter sequences must be non-empty and have equal length.")
    if x_max <= x_min or len(t) == 0 or max_anchor_frames < 2 or max_spatial_samples < 1:
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
    anchors = np.empty((len(anchor_indices), n_solitons), dtype=float)
    anchors[0] = np.asarray(initial_centers, dtype=float)
    fallback = np.zeros(len(anchor_indices), dtype=bool)

    def residual(centers: np.ndarray, target: np.ndarray) -> np.ndarray:
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

    for anchor_position in range(1, len(anchor_indices)):
        previous_index = anchor_indices[anchor_position - 1]
        frame_index = anchor_indices[anchor_position]
        delta_t = float(t[frame_index] - t[previous_index])
        prediction = anchors[anchor_position - 1] + speeds_array * delta_t
        lower = prediction - length / 4.0
        upper = prediction + length / 4.0
        target = np.asarray(numerical[frame_index, sample_indices], dtype=float)
        try:
            fit = least_squares(
                residual,
                prediction,
                args=(target,),
                bounds=(lower, upper),
                loss="soft_l1",
                max_nfev=max_nfev,
            )
            fitted = np.asarray(fit.x, dtype=float)
            if not fit.success or not np.all(np.isfinite(fitted)):
                raise ValueError("Center fit did not converge to finite values.")
            anchors[anchor_position] = fitted
        except (ValueError, FloatingPointError):
            anchors[anchor_position] = prediction
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
    )
