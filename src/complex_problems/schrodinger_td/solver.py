"""Split-operator spectral solver for time-dependent Schrodinger equation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from complex_problems.schrodinger_td.model import (
    build_absorbing_mask_1d,
    build_absorbing_mask_2d,
    initial_packet_1d,
    initial_packet_2d,
    potential_1d,
    potential_2d,
)
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class SchrodingerTDResult:
    """Result bundle for TDSE simulations."""

    dimension: int
    x: np.ndarray
    y: np.ndarray | None
    t: np.ndarray
    psi: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray
    potential: np.ndarray
    kx: np.ndarray
    ky: np.ndarray | None
    spectrum_power: np.ndarray
    invariants: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    magnitudes: dict[str, float] = field(default_factory=dict)


def _build_time_grid(t_min: float, t_max: float, dt: float) -> np.ndarray:
    if t_max <= t_min:
        raise ValueError("t_max must be greater than t_min.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    n_steps = int(np.ceil((t_max - t_min) / dt))
    if n_steps < 1:
        n_steps = 1
    return np.linspace(t_min, t_max, n_steps + 1)


def _observables_1d(
    psi: np.ndarray,
    *,
    x: np.ndarray,
    dx: float,
    k: np.ndarray,
    potential: np.ndarray,
    hbar: float,
    mass: float,
) -> tuple[float, float, float, float, float]:
    density = np.abs(psi) ** 2
    norm = float(np.sum(density) * dx)
    inv_norm = 1.0 / (norm + 1e-15)
    x_mean = float(np.sum(x * density) * dx * inv_norm)
    x_var = float(np.sum((x - x_mean) ** 2 * density) * dx * inv_norm)

    psi_x = np.fft.ifft(1j * k * np.fft.fft(psi))
    p_mean = float(hbar * np.sum(np.imag(np.conjugate(psi) * psi_x)) * dx * inv_norm)
    kinetic = float(0.5 * (hbar**2 / mass) * np.sum(np.abs(psi_x) ** 2) * dx * inv_norm)
    potential_e = float(np.sum(potential * density) * dx * inv_norm)
    energy = kinetic + potential_e
    return norm, x_mean, x_var, p_mean, energy


def _observables_2d(
    psi: np.ndarray,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    dx: float,
    dy: float,
    KX: np.ndarray,
    KY: np.ndarray,
    potential: np.ndarray,
    hbar: float,
    mass: float,
) -> tuple[float, float, float, float, float, float]:
    density = np.abs(psi) ** 2
    dA = dx * dy
    norm = float(np.sum(density) * dA)
    inv_norm = 1.0 / (norm + 1e-15)

    x_mean = float(np.sum(X * density) * dA * inv_norm)
    y_mean = float(np.sum(Y * density) * dA * inv_norm)
    x_var = float(np.sum((X - x_mean) ** 2 * density) * dA * inv_norm)
    y_var = float(np.sum((Y - y_mean) ** 2 * density) * dA * inv_norm)

    psi_k = np.fft.fft2(psi)
    psi_x = np.fft.ifft2(1j * KX * psi_k)
    psi_y = np.fft.ifft2(1j * KY * psi_k)
    kinetic = float(
        0.5 * (hbar**2 / mass) * np.sum(np.abs(psi_x) ** 2 + np.abs(psi_y) ** 2) * dA * inv_norm
    )
    potential_e = float(np.sum(potential * density) * dA * inv_norm)
    energy = kinetic + potential_e
    return norm, x_mean, y_mean, x_var, y_var, energy


def solve_schrodinger_td(
    *,
    dimension: int,
    x_min: float,
    x_max: float,
    nx: int,
    y_min: float = -10.0,
    y_max: float = 10.0,
    ny: int = 128,
    t_min: float = 0.0,
    t_max: float = 8.0,
    dt: float = 0.002,
    hbar: float = 1.0,
    mass: float = 1.0,
    boundary: str = "periodic",
    absorb_ratio: float = 0.1,
    absorb_strength: float = 1.0,
    potential_type: str = "free",
    omega: float = 1.0,
    v0: float = 5.0,
    width: float = 2.0,
    barrier_sigma: float = 0.4,
    lattice_k: float = 2.0,
    a_dw: float = 1.0,
    b_dw: float = 1.0,
    packet_type: str = "gaussian",
    sigma: float = 0.8,
    x0: float = 0.0,
    y0: float = 0.0,
    k0x: float = 0.0,
    k0y: float = 0.0,
    separation: float = 2.0,
    custom_potential_fn_1d=None,
    custom_potential_fn_2d=None,
    custom_packet_fn_1d=None,
    custom_packet_fn_2d=None,
) -> SchrodingerTDResult:
    """Solve TDSE in 1D or 2D with split-operator spectral method."""
    if dimension not in {1, 2}:
        raise ValueError("dimension must be 1 or 2.")
    if nx < 32:
        raise ValueError("nx must be >= 32.")
    if dimension == 2 and ny < 32:
        raise ValueError("ny must be >= 32 for 2D mode.")
    if hbar <= 0 or mass <= 0:
        raise ValueError("hbar and mass must be positive.")
    if boundary not in {"periodic", "absorbing"}:
        raise ValueError("boundary must be 'periodic' or 'absorbing'.")

    t = _build_time_grid(t_min, t_max, dt)
    x = np.linspace(x_min, x_max, nx, endpoint=False)
    dx = float((x_max - x_min) / nx)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)

    if dimension == 1:
        V = potential_1d(
            x,
            potential_type=potential_type,
            omega=omega,
            v0=v0,
            width=width,
            barrier_sigma=barrier_sigma,
            lattice_k=lattice_k,
            a_dw=a_dw,
            b_dw=b_dw,
            custom_fn=custom_potential_fn_1d,
        )
        psi = initial_packet_1d(
            x,
            packet_type=packet_type,
            sigma=sigma,
            x0=x0,
            k0x=k0x,
            separation=separation,
            custom_fn=custom_packet_fn_1d,
        )
        mask = (
            build_absorbing_mask_1d(nx, ratio=absorb_ratio, strength=absorb_strength)
            if boundary == "absorbing"
            else np.ones(nx, dtype=float)
        )

        n_t = len(t)
        psi_hist = np.zeros((n_t, nx), dtype=complex)
        norm = np.zeros(n_t)
        x_mean = np.zeros(n_t)
        x_var = np.zeros(n_t)
        p_mean = np.zeros(n_t)
        energy = np.zeros(n_t)

        psi_hist[0] = psi
        norm[0], x_mean[0], x_var[0], p_mean[0], energy[0] = _observables_1d(
            psi, x=x, dx=dx, k=kx, potential=V, hbar=hbar, mass=mass
        )

        kin_phase = np.exp(-1j * dt * (hbar * (kx**2) / (2.0 * mass)))
        pot_half = np.exp(-1j * V * dt / (2.0 * hbar))
        for i in range(1, n_t):
            psi = pot_half * psi
            psi = np.fft.ifft(kin_phase * np.fft.fft(psi))
            psi = pot_half * psi
            if boundary == "absorbing":
                psi = psi * mask
            psi_hist[i] = psi
            norm[i], x_mean[i], x_var[i], p_mean[i], energy[i] = _observables_1d(
                psi, x=x, dx=dx, k=kx, potential=V, hbar=hbar, mass=mass
            )

        magnitude = np.abs(psi_hist) ** 2
        phase = np.angle(psi_hist)
        spectrum_power = np.abs(np.fft.fftshift(np.fft.fft(psi_hist[-1]))) ** 2
        k_shift = np.fft.fftshift(kx)
        invariants = {
            "norm": norm,
            "x_mean": x_mean,
            "x_var": x_var,
            "p_mean": p_mean,
            "energy": energy,
        }
        magnitudes = {
            "norm_drift_rel": float((norm[-1] - norm[0]) / (abs(norm[0]) + 1e-12)),
            "max_density": float(np.max(magnitude)),
        }
        metadata = {
            "dimension": 1,
            "boundary": boundary,
            "potential_type": potential_type,
            "packet_type": packet_type,
            "hbar": float(hbar),
            "mass": float(mass),
            "dt": float(t[1] - t[0]) if len(t) > 1 else dt,
            "t_min": float(t[0]),
            "t_max": float(t[-1]),
            "nx": int(nx),
        }
        return SchrodingerTDResult(
            dimension=1,
            x=x,
            y=None,
            t=t,
            psi=psi_hist,
            magnitude=magnitude,
            phase=phase,
            potential=V,
            kx=k_shift,
            ky=None,
            spectrum_power=spectrum_power,
            invariants=invariants,
            metadata=metadata,
            magnitudes=magnitudes,
        )

    # 2D branch
    y = np.linspace(y_min, y_max, ny, endpoint=False)
    dy = float((y_max - y_min) / ny)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    X, Y = np.meshgrid(x, y)
    V2 = potential_2d(
        X,
        Y,
        potential_type=potential_type,
        omega=omega,
        v0=v0,
        width=width,
        barrier_sigma=barrier_sigma,
        lattice_k=lattice_k,
        a_dw=a_dw,
        b_dw=b_dw,
        custom_fn=custom_potential_fn_2d,
    )
    psi2 = initial_packet_2d(
        X,
        Y,
        packet_type=packet_type,
        sigma=sigma,
        x0=x0,
        y0=y0,
        k0x=k0x,
        k0y=k0y,
        separation=separation,
        custom_fn=custom_packet_fn_2d,
    )
    mask2 = (
        build_absorbing_mask_2d(nx, ny, ratio=absorb_ratio, strength=absorb_strength)
        if boundary == "absorbing"
        else np.ones((ny, nx), dtype=float)
    )
    KX, KY = np.meshgrid(kx, ky)

    n_t = len(t)
    psi_hist2 = np.zeros((n_t, ny, nx), dtype=complex)
    norm = np.zeros(n_t)
    x_mean = np.zeros(n_t)
    y_mean = np.zeros(n_t)
    x_var = np.zeros(n_t)
    y_var = np.zeros(n_t)
    energy = np.zeros(n_t)

    psi_hist2[0] = psi2
    norm[0], x_mean[0], y_mean[0], x_var[0], y_var[0], energy[0] = _observables_2d(
        psi2, X=X, Y=Y, dx=dx, dy=dy, KX=KX, KY=KY, potential=V2, hbar=hbar, mass=mass
    )

    kin_phase2 = np.exp(-1j * dt * (hbar * (KX**2 + KY**2) / (2.0 * mass)))
    pot_half2 = np.exp(-1j * V2 * dt / (2.0 * hbar))
    for i in range(1, n_t):
        psi2 = pot_half2 * psi2
        psi2 = np.fft.ifft2(kin_phase2 * np.fft.fft2(psi2))
        psi2 = pot_half2 * psi2
        if boundary == "absorbing":
            psi2 = psi2 * mask2
        psi_hist2[i] = psi2
        norm[i], x_mean[i], y_mean[i], x_var[i], y_var[i], energy[i] = _observables_2d(
            psi2, X=X, Y=Y, dx=dx, dy=dy, KX=KX, KY=KY, potential=V2, hbar=hbar, mass=mass
        )

    magnitude2 = np.abs(psi_hist2) ** 2
    phase2 = np.angle(psi_hist2)
    spectrum2 = np.abs(np.fft.fftshift(np.fft.fft2(psi_hist2[-1]))) ** 2
    invariants2 = {
        "norm": norm,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "x_var": x_var,
        "y_var": y_var,
        "energy": energy,
    }
    magnitudes2 = {
        "norm_drift_rel": float((norm[-1] - norm[0]) / (abs(norm[0]) + 1e-12)),
        "max_density": float(np.max(magnitude2)),
    }
    metadata2 = {
        "dimension": 2,
        "boundary": boundary,
        "potential_type": potential_type,
        "packet_type": packet_type,
        "hbar": float(hbar),
        "mass": float(mass),
        "dt": float(t[1] - t[0]) if len(t) > 1 else dt,
        "t_min": float(t[0]),
        "t_max": float(t[-1]),
        "nx": int(nx),
        "ny": int(ny),
    }
    return SchrodingerTDResult(
        dimension=2,
        x=x,
        y=y,
        t=t,
        psi=psi_hist2,
        magnitude=magnitude2,
        phase=phase2,
        potential=V2,
        kx=np.fft.fftshift(kx),
        ky=np.fft.fftshift(ky),
        spectrum_power=spectrum2,
        invariants=invariants2,
        metadata=metadata2,
        magnitudes=magnitudes2,
    )

