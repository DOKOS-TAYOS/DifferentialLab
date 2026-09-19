"""Model helpers for nonlinear wave solvers (NLSE and KdV)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KdvSolitonCharacteristics:
    """Physical parameters of one KdV soliton."""

    amplitude: float
    center: float
    inverse_width: float
    speed: float


def kdv_soliton_characteristics(
    *, amplitude: float, center: float, c: float, alpha: float, beta_disp: float
) -> KdvSolitonCharacteristics:
    """Return the inverse width and nominal speed of a real KdV soliton.

    A real soliton requires ``alpha * amplitude / beta_disp > 0``.  The
    inverse width is ``sqrt(alpha * amplitude / (12 * beta_disp))`` and the
    nominal speed is ``c + alpha * amplitude / 3``.
    """
    values = (amplitude, center, c, alpha, beta_disp)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("KdV soliton parameters must be finite.")
    ratio = alpha * amplitude / beta_disp if beta_disp != 0.0 else 0.0
    if alpha == 0.0 or beta_disp == 0.0 or ratio <= 0.0:
        raise ValueError("A real KdV soliton requires alpha * amplitude / beta_disp > 0.")
    return KdvSolitonCharacteristics(
        amplitude=float(amplitude),
        center=float(center),
        inverse_width=float(np.sqrt(ratio / 12.0)),
        speed=float(c + alpha * amplitude / 3.0),
    )


def build_kdv_soliton_profile(
    x: np.ndarray,
    *,
    amplitude: float,
    center: float,
    c: float,
    alpha: float,
    beta_disp: float,
) -> tuple[np.ndarray, KdvSolitonCharacteristics]:
    """Build the exact one-soliton traveling-wave profile at ``t=0``."""
    characteristics = kdv_soliton_characteristics(
        amplitude=amplitude,
        center=center,
        c=c,
        alpha=alpha,
        beta_disp=beta_disp,
    )
    profile = characteristics.amplitude * np.square(
        1.0 / np.cosh(characteristics.inverse_width * (x - characteristics.center))
    )
    return profile, characteristics


def build_kdv_soliton_train(
    x: np.ndarray,
    *,
    amplitudes: Sequence[float],
    centers: Sequence[float],
    c: float,
    alpha: float,
    beta_disp: float,
) -> tuple[np.ndarray, list[KdvSolitonCharacteristics]]:
    """Build separated one-soliton profiles used as initial data.

    This is a superposition of separated one-soliton profiles, not the exact
    KdV multi-soliton tau-function solution.
    """
    if len(amplitudes) != len(centers):
        raise ValueError("KdV soliton amplitudes and centers must have equal length.")
    if not amplitudes:
        raise ValueError("At least one KdV soliton is required.")
    characteristics = [
        kdv_soliton_characteristics(
            amplitude=float(amplitude),
            center=float(center),
            c=c,
            alpha=alpha,
            beta_disp=beta_disp,
        )
        for amplitude, center in zip(amplitudes, centers, strict=True)
    ]
    profile = np.zeros_like(x, dtype=float)
    for soliton in characteristics:
        profile += soliton.amplitude * np.square(
            1.0 / np.cosh(soliton.inverse_width * (x - soliton.center))
        )
    return profile, characteristics


def build_periodic_grid(
    x_min: float,
    x_max: float,
    nx: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Build periodic spatial grid and spectral k-grid."""
    if nx < 16:
        raise ValueError("nx must be at least 16.")
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min.")
    x = np.linspace(x_min, x_max, nx, endpoint=False)
    dx = float((x_max - x_min) / nx)
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    return x, dx, k


def build_initial_profile(
    x: np.ndarray,
    *,
    profile: str,
    amplitude: float,
    sigma: float,
    center: float,
    custom_fn: Callable[[float], float] | None = None,
) -> np.ndarray:
    """Build real-valued initial profile."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if profile == "gaussian":
        return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2))
    if profile == "sech":
        return amplitude / np.cosh((x - center) / sigma)
    if profile == "pulse":
        out = np.zeros_like(x)
        out[np.abs(x - center) <= sigma] = amplitude
        return out
    if profile == "custom":
        if custom_fn is None:
            raise ValueError("custom profile selected but no custom expression provided.")
        return np.array([custom_fn(float(xi)) for xi in x], dtype=float)
    raise ValueError(f"Unknown profile '{profile}'.")


def compute_nlse_invariants(
    psi: np.ndarray,
    *,
    dx: float,
    k: np.ndarray,
    beta2: float,
    gamma: float,
) -> tuple[float, float, float]:
    """Compute NLSE invariants: norm, momentum, hamiltonian (approx)."""
    norm = float(np.sum(np.abs(psi) ** 2) * dx)
    psi_x = np.fft.ifft(1j * k * np.fft.fft(psi))
    momentum = float(np.sum(np.imag(np.conjugate(psi) * psi_x)) * dx)
    hamiltonian_density = 0.5 * beta2 * np.abs(psi_x) ** 2 - 0.5 * gamma * np.abs(psi) ** 4
    hamiltonian = float(np.sum(hamiltonian_density) * dx)
    return norm, momentum, hamiltonian


def compute_kdv_invariants(
    u: np.ndarray,
    *,
    dx: float,
    k: np.ndarray,
    c: float = 0.0,
    alpha: float = 6.0,
    beta_disp: float = 1.0,
) -> tuple[float, float, float]:
    """Compute KdV invariants for the configured equation."""
    mass = float(np.sum(u) * dx)
    l2 = float(np.sum(u**2) * dx)
    ux = np.fft.ifft(1j * k * np.fft.fft(u)).real
    hamiltonian = float(
        np.sum(0.5 * beta_disp * ux**2 - (alpha / 6.0) * u**3 - 0.5 * c * u**2) * dx
    )
    return mass, l2, hamiltonian
