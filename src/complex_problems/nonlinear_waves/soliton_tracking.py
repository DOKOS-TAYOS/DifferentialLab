"""Observation-driven tracking of fixed-shape KdV soliton profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks


@dataclass(frozen=True)
class TrackedSolitonCenters:
    """Unwrapped tracks plus the reliable peak observations behind them."""

    centers: np.ndarray
    anchor_indices: np.ndarray
    observed: np.ndarray
    observed_centers: np.ndarray
    candidate_count: np.ndarray


@dataclass(frozen=True)
class _PeakCandidate:
    """One periodic, sub-grid numerical extremum at an anchor."""

    center: float
    height: float
    prominence: float


# These are local tracker choices, scaled by the supplied soliton metadata rather
# than absolute coordinates or amplitudes. They reject ripples and merged peaks
# without turning the theoretical path into a positional constraint.
_MIN_PROMINENCE_FRACTION = 0.12
_AMPLITUDE_RELATIVE_TOLERANCE = 0.30
_AMPLITUDE_GLOBAL_TOLERANCE = 0.08
_DUMMY_ASSIGNMENT_COST = 2.0
_MAX_ACCEPTED_ASSIGNMENT_COST = 1.6


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


def _periodic_distance(first: float, second: float, length: float) -> float:
    """Return the signed minimum-image displacement from ``second`` to ``first``."""
    return float(((first - second + length / 2.0) % length) - length / 2.0)


def _nearest_unwrapped(wrapped_center: float, prediction: float, length: float) -> float:
    """Choose the periodic copy of a center nearest a continuity prediction."""
    return prediction + _periodic_distance(wrapped_center, prediction, length)


def _detect_periodic_peaks(
    frame: np.ndarray,
    *,
    x: np.ndarray,
    x_min: float,
    length: float,
    polarity: float,
    minimum_prominence: float,
) -> list[_PeakCandidate]:
    """Detect and parabolically refine extrema on one periodic spatial frame."""
    if not np.all(np.isfinite(frame)):
        return []

    oriented = polarity * frame
    repeated = np.concatenate((oriented, oriented, oriented))
    peak_indices, properties = find_peaks(repeated, prominence=minimum_prominence)
    point_count = len(frame)
    spacing = length / point_count
    candidates: list[_PeakCandidate] = []
    for peak_index, prominence in zip(peak_indices, properties["prominences"], strict=True):
        if not point_count <= peak_index < 2 * point_count:
            continue
        left = float(repeated[peak_index - 1])
        middle = float(repeated[peak_index])
        right = float(repeated[peak_index + 1])
        denominator = left - 2.0 * middle + right
        offset = 0.0 if denominator == 0.0 else 0.5 * (left - right) / denominator
        offset = float(np.clip(offset, -0.5, 0.5))
        grid_index = peak_index - point_count
        wrapped_center = x_min + ((float(x[grid_index]) - x_min + offset * spacing) % length)
        candidates.append(
            _PeakCandidate(
                center=wrapped_center,
                height=float(frame[grid_index]),
                prominence=float(prominence),
            )
        )
    return candidates


def _smoothstep(values: np.ndarray) -> np.ndarray:
    """Return the cubic interpolation weights used for ambiguous intervals."""
    return 3.0 * np.square(values) - 2.0 * np.power(values, 3)


def _bridge_phase_track(
    t: np.ndarray,
    anchor_indices: np.ndarray,
    phase_observations: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Bridge each unobserved interval between reliable phase observations."""
    observed_positions = np.flatnonzero(observed)
    phase = np.empty(len(t), dtype=float)
    first_position = int(observed_positions[0])
    first_frame = int(anchor_indices[first_position])
    phase[: first_frame + 1] = phase_observations[first_position]

    for left_position, right_position in zip(
        observed_positions[:-1], observed_positions[1:], strict=True
    ):
        left_frame = int(anchor_indices[left_position])
        right_frame = int(anchor_indices[right_position])
        interval_t = t[left_frame : right_frame + 1]
        span = float(t[right_frame] - t[left_frame])
        if span == 0.0:
            phase[left_frame : right_frame + 1] = phase_observations[right_position]
            continue
        normalized = (interval_t - t[left_frame]) / span
        phase[left_frame : right_frame + 1] = phase_observations[left_position] + _smoothstep(
            normalized
        ) * (phase_observations[right_position] - phase_observations[left_position])

    last_position = int(observed_positions[-1])
    last_frame = int(anchor_indices[last_position])
    phase[last_frame:] = phase_observations[last_position]
    return phase


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
) -> TrackedSolitonCenters:
    """Track identifiable periodic peaks and bridge ambiguous KdV interactions.

    The supplied numerical history is sampled only at bounded temporal anchors.
    Observed centers are associated by peak amplitude and continuity, then their
    phase shifts bridge intervals where overlapping pulses have no unique identity.
    """
    if numerical.shape != (len(t), len(x)):
        raise ValueError("numerical must have shape (len(t), len(x)).")
    n_solitons = len(amplitudes)
    if not n_solitons or not all(
        len(values) == n_solitons for values in (inverse_widths, initial_centers, speeds)
    ):
        raise ValueError("Soliton parameter sequences must be non-empty and have equal length.")
    if x_max <= x_min or len(x) < 3 or len(t) == 0 or max_anchor_frames < 2:
        raise ValueError("Invalid tracking grid or temporal-anchor settings.")

    amplitude_values = np.asarray(amplitudes, dtype=float)
    width_values = np.asarray(inverse_widths, dtype=float)
    initial_values = np.asarray(initial_centers, dtype=float)
    speed_values = np.asarray(speeds, dtype=float)
    if not (
        np.all(np.isfinite(amplitude_values))
        and np.all(amplitude_values != 0.0)
        and np.all(np.isfinite(width_values))
        and np.all(width_values > 0.0)
        and np.all(np.isfinite(initial_values))
        and np.all(np.isfinite(speed_values))
    ):
        raise ValueError("Soliton parameters must be finite, non-zero, and have positive widths.")
    polarity = float(np.sign(amplitude_values[0]))
    if not np.all(np.sign(amplitude_values) == polarity):
        raise ValueError("KdV solitons in one result must share a common polarity.")

    anchor_count = min(len(t), max_anchor_frames)
    anchor_indices = np.unique(np.linspace(0, len(t) - 1, anchor_count, dtype=int))
    anchor_t = t[anchor_indices]
    length = x_max - x_min
    max_amplitude = float(np.max(np.abs(amplitude_values)))
    amplitude_scale = np.maximum(
        _AMPLITUDE_RELATIVE_TOLERANCE * np.abs(amplitude_values),
        _AMPLITUDE_GLOBAL_TOLERANCE * max_amplitude,
    )
    minimum_prominence = _MIN_PROMINENCE_FRACTION * float(np.min(np.abs(amplitude_values)))

    observed = np.zeros((len(anchor_indices), n_solitons), dtype=bool)
    observed[0] = True  # Metadata fixes the initial phase to zero, even if it is overlapping.
    observed_centers = np.full((len(anchor_indices), n_solitons), np.nan, dtype=float)
    observed_centers[0] = initial_values
    candidate_count = np.zeros(len(anchor_indices), dtype=int)
    last_centers = initial_values.copy()
    last_times = np.full(n_solitons, float(anchor_t[0]), dtype=float)

    for anchor_position in range(1, len(anchor_indices)):
        frame = np.asarray(numerical[anchor_indices[anchor_position]], dtype=float)
        candidates = _detect_periodic_peaks(
            frame,
            x=x,
            x_min=x_min,
            length=length,
            polarity=polarity,
            minimum_prominence=minimum_prominence,
        )
        candidate_count[anchor_position] = len(candidates)
        if not candidates:
            continue

        predictions = last_centers + speed_values * (anchor_t[anchor_position] - last_times)
        compatible = np.array(
            [
                [
                    abs(candidate.height - amplitude_values[index]) <= amplitude_scale[index]
                    for candidate in candidates
                ]
                for index in range(n_solitons)
            ],
            dtype=bool,
        )
        costs = np.full((n_solitons, len(candidates) + n_solitons), _DUMMY_ASSIGNMENT_COST)
        candidate_centers = np.empty((n_solitons, len(candidates)), dtype=float)
        for identity in range(n_solitons):
            for candidate_index, candidate in enumerate(candidates):
                unwrapped = _nearest_unwrapped(candidate.center, predictions[identity], length)
                candidate_centers[identity, candidate_index] = unwrapped
                if not compatible[identity, candidate_index]:
                    costs[identity, candidate_index] = np.inf
                    continue
                amplitude_cost = (
                    abs(candidate.height - amplitude_values[identity]) / amplitude_scale[identity]
                )
                distance_cost = abs(unwrapped - predictions[identity]) / (length / 2.0)
                quality_cost = minimum_prominence / max(candidate.prominence, minimum_prominence)
                costs[identity, candidate_index] = (
                    amplitude_cost + distance_cost + 0.1 * quality_cost
                )

        identities, assignments = linear_sum_assignment(costs)
        assigned_candidates = dict(zip(identities.tolist(), assignments.tolist(), strict=True))
        for identity in range(n_solitons):
            candidate_index = assigned_candidates[identity]
            if (
                candidate_index >= len(candidates)
                or costs[identity, candidate_index] > _MAX_ACCEPTED_ASSIGNMENT_COST
            ):
                continue
            # Two resolved numerical extrema closer than their profile scales are
            # still an ambiguous decomposition, even if assignment found a pairing.
            overlaps_assigned_peak = any(
                other != identity
                and other_candidate < len(candidates)
                and abs(
                    _periodic_distance(
                        candidates[candidate_index].center,
                        candidates[other_candidate].center,
                        length,
                    )
                )
                < max(1.0 / width_values[identity], 1.0 / width_values[other])
                for other, other_candidate in assigned_candidates.items()
            )
            if overlaps_assigned_peak:
                continue
            center = candidate_centers[identity, candidate_index]
            observed[anchor_position, identity] = True
            observed_centers[anchor_position, identity] = center
            last_centers[identity] = center
            last_times[identity] = anchor_t[anchor_position]

    free_centers = initial_values[None, :] + (t[:, None] - t[0]) * speed_values[None, :]
    tracks = np.empty_like(free_centers)
    for identity in range(n_solitons):
        phase_observations = observed_centers[:, identity] - (
            initial_values[identity] + (anchor_t - t[0]) * speed_values[identity]
        )
        phase_observations[0] = 0.0
        phase = _bridge_phase_track(
            t,
            anchor_indices,
            phase_observations,
            observed[:, identity],
        )
        tracks[:, identity] = free_centers[:, identity] + phase

    return TrackedSolitonCenters(
        centers=tracks,
        anchor_indices=anchor_indices,
        observed=observed,
        observed_centers=observed_centers,
        candidate_count=candidate_count,
    )
