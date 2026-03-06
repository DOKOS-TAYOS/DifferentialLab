"""Analytical far-field solver for antenna radiation patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from complex_problems.antenna_radiation.model import (
    antenna_pattern,
    build_angular_grid,
    compute_directivity,
    compute_reaction_metrics,
    degrees,
    e_from_power_density,
    estimate_aperture_lambda,
    far_field_distance_min,
    h_from_e_field,
    normalize_pattern,
    to_db10,
    wavelength_from_frequency,
)
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class AntennaRadiationResult:
    """Result bundle for antenna radiation problem."""

    antenna_type: str
    theta: np.ndarray
    phi: np.ndarray
    pattern_norm: np.ndarray
    directivity: np.ndarray
    directivity_db: np.ndarray
    gain: np.ndarray
    gain_db: np.ndarray
    power_density: np.ndarray
    e_rms: np.ndarray
    h_rms: np.ndarray
    theta_cut_db: np.ndarray
    phi_cut_db: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    magnitudes: dict[str, float] = field(default_factory=dict)


def solve_antenna_radiation(
    *,
    antenna_type: str = "dipole",
    frequency_hz: float = 1.0e9,
    transmit_power_w: float = 10.0,
    efficiency: float = 0.9,
    observation_distance_m: float = 50.0,
    n_theta: int = 181,
    n_phi: int = 360,
    length_lambda: float = 0.5,
    loop_radius_lambda: float = 0.1,
    patch_length_lambda: float = 0.5,
    patch_width_lambda: float = 0.4,
    array_elements: int = 8,
    array_spacing_lambda: float = 0.5,
    array_phase_deg: float = 0.0,
    array_steer_theta_deg: float = 90.0,
) -> AntennaRadiationResult:
    """Solve a far-field antenna radiation case."""
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive.")
    if transmit_power_w <= 0.0:
        raise ValueError("transmit_power_w must be positive.")
    if not (0.0 < efficiency <= 1.0):
        raise ValueError("efficiency must be in (0, 1].")
    if observation_distance_m <= 0.0:
        raise ValueError("observation_distance_m must be positive.")

    theta, phi = build_angular_grid(n_theta=n_theta, n_phi=n_phi)
    p_raw = antenna_pattern(
        antenna_type=antenna_type,
        theta=theta,
        phi=phi,
        length_lambda=length_lambda,
        loop_radius_lambda=loop_radius_lambda,
        patch_length_lambda=patch_length_lambda,
        patch_width_lambda=patch_width_lambda,
        array_elements=array_elements,
        array_spacing_lambda=array_spacing_lambda,
        array_phase_deg=array_phase_deg,
        array_steer_theta_deg=array_steer_theta_deg,
    )
    p_norm = normalize_pattern(p_raw)
    d = compute_directivity(p_norm, theta, phi)
    g = efficiency * d

    p_rad = transmit_power_w * efficiency
    power_density = transmit_power_w * g / (4.0 * np.pi * observation_distance_m**2)
    e_rms = e_from_power_density(power_density)
    h_rms = h_from_e_field(e_rms)

    d_db = to_db10(d)
    g_db = to_db10(g)

    # Principal cuts.
    phi0_idx = 0
    theta90_idx = int(np.argmin(np.abs(theta - 0.5 * np.pi)))
    theta_cut_db = g_db[:, phi0_idx]
    phi_cut_db = g_db[theta90_idx, :]

    wavelength = wavelength_from_frequency(frequency_hz)
    ap_lambda = estimate_aperture_lambda(
        antenna_type=antenna_type,
        length_lambda=length_lambda,
        loop_radius_lambda=loop_radius_lambda,
        patch_length_lambda=patch_length_lambda,
        patch_width_lambda=patch_width_lambda,
        array_elements=array_elements,
        array_spacing_lambda=array_spacing_lambda,
    )
    r_ff = far_field_distance_min(wavelength, ap_lambda)
    metrics = compute_reaction_metrics(
        directivity=d,
        gain=g,
        theta=theta,
        phi=phi,
        p_norm=p_norm,
    )

    magnitudes = {
        "directivity_max_db": metrics["directivity_max_db"],
        "gain_max_db": metrics["gain_max_db"],
        "beamwidth_deg": metrics["beamwidth_deg"],
        "theta_peak_deg": metrics["theta_peak_deg"],
        "phi_peak_deg": metrics["phi_peak_deg"],
        "max_e_rms_vpm": float(np.max(e_rms)),
        "far_field_min_m": float(r_ff),
    }
    metadata = {
        "antenna_type": antenna_type.lower().strip(),
        "frequency_hz": float(frequency_hz),
        "wavelength_m": float(wavelength),
        "transmit_power_w": float(transmit_power_w),
        "radiated_power_w": float(p_rad),
        "efficiency": float(efficiency),
        "observation_distance_m": float(observation_distance_m),
        "n_theta": int(n_theta),
        "n_phi": int(n_phi),
        "length_lambda": float(length_lambda),
        "loop_radius_lambda": float(loop_radius_lambda),
        "patch_length_lambda": float(patch_length_lambda),
        "patch_width_lambda": float(patch_width_lambda),
        "array_elements": int(array_elements),
        "array_spacing_lambda": float(array_spacing_lambda),
        "array_phase_deg": float(array_phase_deg),
        "array_steer_theta_deg": float(array_steer_theta_deg),
        "is_far_field": bool(observation_distance_m >= r_ff),
    }

    logger.info(
        "Solved antenna radiation: type=%s, f=%g Hz, Dmax=%+.2f dB, Gmax=%+.2f dB",
        metadata["antenna_type"],
        frequency_hz,
        magnitudes["directivity_max_db"],
        magnitudes["gain_max_db"],
    )
    return AntennaRadiationResult(
        antenna_type=metadata["antenna_type"],
        theta=degrees(theta),
        phi=degrees(phi),
        pattern_norm=p_norm,
        directivity=d,
        directivity_db=d_db,
        gain=g,
        gain_db=g_db,
        power_density=power_density,
        e_rms=e_rms,
        h_rms=h_rms,
        theta_cut_db=theta_cut_db,
        phi_cut_db=phi_cut_db,
        metadata=metadata,
        magnitudes=magnitudes,
    )
