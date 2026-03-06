"""Model helpers for nonlinear wave solvers (NLSE and KdV)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


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
) -> tuple[float, float, float]:
    """Compute KdV invariants: mass, L2, and approximate Hamiltonian."""
    mass = float(np.sum(u) * dx)
    l2 = float(np.sum(u**2) * dx)
    ux = np.fft.ifft(1j * k * np.fft.fft(u)).real
    hamiltonian = float(np.sum(0.5 * ux**2 - (u**3) / 6.0) * dx)
    return mass, l2, hamiltonian
