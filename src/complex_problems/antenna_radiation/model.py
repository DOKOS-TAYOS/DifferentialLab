"""Model helpers for far-field antenna radiation patterns."""

from __future__ import annotations

import math

import numpy as np

C0 = 299_792_458.0
ETA0 = 376.730313668

_ANTENNA_TYPES = {"dipole", "loop", "patch", "array"}


def build_angular_grid(n_theta: int, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
    """Build theta/phi angular grids."""
    if n_theta < 11:
        raise ValueError("n_theta must be >= 11.")
    if n_phi < 16:
        raise ValueError("n_phi must be >= 16.")
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    return theta, phi


def _safe_divide(num: np.ndarray, den: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    out = np.zeros_like(num, dtype=float)
    mask = np.abs(den) > eps
    out[mask] = num[mask] / den[mask]
    return out


def pattern_dipole(theta_grid: np.ndarray, *, length_lambda: float) -> np.ndarray:
    """Approximate thin-wire dipole power pattern."""
    if length_lambda <= 0:
        raise ValueError("length_lambda must be positive.")
    kl2 = np.pi * length_lambda
    s = np.sin(theta_grid)
    n = np.cos(kl2 * np.cos(theta_grid)) - np.cos(kl2)
    f = _safe_divide(n, s)
    p = np.abs(f) ** 2
    return p


def pattern_small_loop(theta_grid: np.ndarray, *, radius_lambda: float) -> np.ndarray:
    """Small-loop-like power pattern."""
    if radius_lambda <= 0:
        raise ValueError("radius_lambda must be positive.")
    scale = (2.0 * np.pi * radius_lambda) ** 2
    return (scale * np.sin(theta_grid)) ** 2


def pattern_patch(
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    *,
    length_lambda: float,
    width_lambda: float,
) -> np.ndarray:
    """Simple aperture-model patch-like power pattern."""
    if length_lambda <= 0 or width_lambda <= 0:
        raise ValueError("Patch dimensions must be positive.")
    t1 = np.cos(0.5 * np.pi * length_lambda * np.cos(theta_grid))
    arg = 0.5 * width_lambda * np.sin(theta_grid) * np.cos(phi_grid)
    t2 = np.sinc(arg)
    p = (t1 * t2) ** 2
    return p


def pattern_uniform_linear_array(
    theta_grid: np.ndarray,
    *,
    n_elements: int,
    spacing_lambda: float,
    phase_deg: float,
    steer_theta_deg: float,
) -> np.ndarray:
    """Uniform linear array factor power (array axis z, theta from z)."""
    if n_elements < 2:
        raise ValueError("n_elements must be >= 2.")
    if spacing_lambda <= 0:
        raise ValueError("spacing_lambda must be positive.")
    beta = np.deg2rad(phase_deg)
    theta0 = np.deg2rad(steer_theta_deg)
    psi = 2.0 * np.pi * spacing_lambda * (np.cos(theta_grid) - np.cos(theta0)) + beta
    num = np.sin(0.5 * n_elements * psi)
    den = n_elements * np.sin(0.5 * psi)
    af = _safe_divide(num, den)
    # Add a dipole-like element factor to avoid isotropic unrealistic lobes.
    element = np.sin(theta_grid) ** 2
    return (np.abs(af) ** 2) * element


def antenna_pattern(
    *,
    antenna_type: str,
    theta: np.ndarray,
    phi: np.ndarray,
    length_lambda: float,
    loop_radius_lambda: float,
    patch_length_lambda: float,
    patch_width_lambda: float,
    array_elements: int,
    array_spacing_lambda: float,
    array_phase_deg: float,
    array_steer_theta_deg: float,
) -> np.ndarray:
    """Build raw non-normalized power pattern on theta-phi mesh."""
    atype = antenna_type.strip().lower()
    if atype not in _ANTENNA_TYPES:
        raise ValueError(f"Unknown antenna_type '{antenna_type}'.")

    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    if atype == "dipole":
        p = pattern_dipole(TH, length_lambda=length_lambda)
    elif atype == "loop":
        p = pattern_small_loop(TH, radius_lambda=loop_radius_lambda)
    elif atype == "patch":
        p = pattern_patch(
            TH,
            PH,
            length_lambda=patch_length_lambda,
            width_lambda=patch_width_lambda,
        )
    else:
        p = pattern_uniform_linear_array(
            TH,
            n_elements=array_elements,
            spacing_lambda=array_spacing_lambda,
            phase_deg=array_phase_deg,
            steer_theta_deg=array_steer_theta_deg,
        )

    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p[p < 0.0] = 0.0
    return p


def normalize_pattern(p: np.ndarray) -> np.ndarray:
    """Normalize pattern to max=1."""
    pmax = float(np.max(p))
    if pmax <= 0.0:
        return np.zeros_like(p)
    return p / pmax


def compute_directivity(p_norm: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Compute directivity map from normalized pattern."""
    TH, _ = np.meshgrid(theta, phi, indexing="ij")
    integrand = p_norm * np.sin(TH)
    int_phi = np.trapezoid(integrand, phi, axis=1)
    total = float(np.trapezoid(int_phi, theta, axis=0))
    if total <= 0.0:
        return np.zeros_like(p_norm)
    return 4.0 * np.pi * p_norm / total


def estimate_beamwidth_deg(theta: np.ndarray, cut_power: np.ndarray) -> float:
    """Estimate -3 dB beamwidth from a theta cut."""
    if len(theta) < 3:
        return float("nan")
    idx_peak = int(np.argmax(cut_power))
    peak = float(cut_power[idx_peak])
    if peak <= 0.0:
        return float("nan")
    threshold = 0.5 * peak

    left = idx_peak
    while left > 0 and cut_power[left] > threshold:
        left -= 1
    right = idx_peak
    while right < len(theta) - 1 and cut_power[right] > threshold:
        right += 1

    if left == idx_peak or right == idx_peak:
        return float("nan")
    return float(np.rad2deg(theta[right] - theta[left]))


def estimate_aperture_lambda(
    *,
    antenna_type: str,
    length_lambda: float,
    loop_radius_lambda: float,
    patch_length_lambda: float,
    patch_width_lambda: float,
    array_elements: int,
    array_spacing_lambda: float,
) -> float:
    """Estimate largest physical dimension in wavelengths."""
    atype = antenna_type.strip().lower()
    if atype == "dipole":
        return max(1e-6, length_lambda)
    if atype == "loop":
        return max(1e-6, 2.0 * loop_radius_lambda)
    if atype == "patch":
        return max(1e-6, patch_length_lambda, patch_width_lambda)
    return max(1e-6, (array_elements - 1) * array_spacing_lambda)


def to_db10(x: np.ndarray, *, floor_db: float = -80.0) -> np.ndarray:
    """Convert linear values to dB with floor."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(np.maximum(x, 1e-16))
    return np.maximum(out, floor_db)


def wrap_angle_deg(angle: float) -> float:
    """Wrap angle to [-180, 180)."""
    val = float(angle) % 360.0
    if val >= 180.0:
        val -= 360.0
    return val


def degrees(rad: np.ndarray) -> np.ndarray:
    """Radians to degrees."""
    return np.rad2deg(rad)


def wavelength_from_frequency(frequency_hz: float) -> float:
    """Compute free-space wavelength."""
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive.")
    return C0 / frequency_hz


def far_field_distance_min(wavelength: float, aperture_lambda: float) -> float:
    """Fraunhofer minimum far-field distance."""
    d = aperture_lambda * wavelength
    return 2.0 * d * d / wavelength


def e_from_power_density(power_density: np.ndarray) -> np.ndarray:
    """Convert power density to RMS electric field magnitude."""
    return np.sqrt(np.maximum(power_density, 0.0) * ETA0)


def h_from_e_field(e_rms: np.ndarray) -> np.ndarray:
    """Convert RMS electric field to RMS magnetic field."""
    return e_rms / ETA0


def compute_reaction_metrics(
    *,
    directivity: np.ndarray,
    gain: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    p_norm: np.ndarray,
) -> dict[str, float]:
    """Compute scalar quality metrics from solved patterns."""
    dmax = float(np.max(directivity))
    gmax = float(np.max(gain))
    phi0_idx = 0
    cut = p_norm[:, phi0_idx]
    bw = estimate_beamwidth_deg(theta, cut)

    theta_peak, phi_peak = np.unravel_index(np.argmax(gain), gain.shape)
    return {
        "directivity_max_db": float(10.0 * math.log10(max(dmax, 1e-16))),
        "gain_max_db": float(10.0 * math.log10(max(gmax, 1e-16))),
        "beamwidth_deg": float(bw),
        "theta_peak_deg": float(np.rad2deg(theta[theta_peak])),
        "phi_peak_deg": float(np.rad2deg(phi[phi_peak])),
    }
