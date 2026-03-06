"""Pseudo-spectral solvers for nonlinear wave models (NLSE and KdV)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from complex_problems.nonlinear_waves.model import (
    build_initial_profile,
    build_periodic_grid,
    compute_kdv_invariants,
    compute_nlse_invariants,
)
from utils import get_logger

logger = get_logger(__name__)

_MODEL_TYPES = {"nlse", "kdv"}


@dataclass
class NonlinearWavesResult:
    """Result bundle for nonlinear wave simulations."""

    model_type: str
    x: np.ndarray
    t: np.ndarray
    field: np.ndarray  # (n_t, nx) complex for NLSE, float for KdV
    magnitude: np.ndarray  # (n_t, nx)
    phase: np.ndarray | None
    k: np.ndarray
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


def _simulate_nlse(
    *,
    x: np.ndarray,
    t: np.ndarray,
    dx: float,
    k: np.ndarray,
    psi0: np.ndarray,
    beta2: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.0
    n_t, nx = len(t), len(x)
    psi_hist = np.zeros((n_t, nx), dtype=complex)
    psi = np.array(psi0, dtype=complex, copy=True)
    psi_hist[0] = psi

    lin_half = np.exp(-0.5j * beta2 * (k**2) * dt)
    norms = np.zeros(n_t)
    momenta = np.zeros(n_t)
    energies = np.zeros(n_t)
    norms[0], momenta[0], energies[0] = compute_nlse_invariants(
        psi, dx=dx, k=k, beta2=beta2, gamma=gamma
    )

    for step in range(1, n_t):
        psi = np.fft.ifft(lin_half * np.fft.fft(psi))
        psi = psi * np.exp(-1j * gamma * np.abs(psi) ** 2 * dt)
        psi = np.fft.ifft(lin_half * np.fft.fft(psi))
        psi_hist[step] = psi
        norms[step], momenta[step], energies[step] = compute_nlse_invariants(
            psi, dx=dx, k=k, beta2=beta2, gamma=gamma
        )

    magnitude = np.abs(psi_hist) ** 2
    invariants = {"norm": norms, "momentum": momenta, "hamiltonian": energies}
    return psi_hist, magnitude, invariants


def _kdv_etdrk4_coefficients(
    linear_op: np.ndarray,
    *,
    dt: float,
    contour_samples: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ETDRK4 coefficients for the semi-linear KdV Fourier system."""
    e = np.exp(dt * linear_op)
    e_half = np.exp(0.5 * dt * linear_op)
    roots = np.exp(1j * np.pi * ((np.arange(1, contour_samples + 1) - 0.5) / contour_samples))
    lr = dt * linear_op[:, None] + roots[None, :]
    q = dt * np.real(np.mean((np.exp(0.5 * lr) - 1.0) / lr, axis=1))
    f1 = dt * np.real(
        np.mean(
            (-4.0 - lr + np.exp(lr) * (4.0 - 3.0 * lr + lr**2)) / (lr**3),
            axis=1,
        )
    )
    f2 = dt * np.real(np.mean((2.0 + lr + np.exp(lr) * (-2.0 + lr)) / (lr**3), axis=1))
    f3 = dt * np.real(
        np.mean(
            (-4.0 - 3.0 * lr - lr**2 + np.exp(lr) * (4.0 - lr)) / (lr**3),
            axis=1,
        )
    )
    return e, e_half, q, f1, f2, f3


def _simulate_kdv(
    *,
    x: np.ndarray,
    t: np.ndarray,
    dx: float,
    k: np.ndarray,
    u0: np.ndarray,
    c: float,
    alpha: float,
    beta_disp: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.0
    n_t, nx = len(t), len(x)
    u_hist = np.zeros((n_t, nx), dtype=float)
    u = np.array(u0, dtype=float, copy=True)
    u_hist[0] = u

    k_max = float(np.max(np.abs(k)))
    dealias_mask = np.abs(k) <= (2.0 / 3.0) * k_max + 1e-12
    linear_op = 1j * (beta_disp * (k**3) - c * k)

    if n_t > 1:
        e, e_half, q, f1, f2, f3 = _kdv_etdrk4_coefficients(linear_op, dt=dt)
    v = np.fft.fft(u)
    v[~dealias_mask] = 0.0

    def _nonlinear(v_hat: np.ndarray) -> np.ndarray:
        u_state = np.fft.ifft(v_hat).real
        u_sq_hat = np.fft.fft(u_state * u_state)
        u_sq_hat[~dealias_mask] = 0.0
        return -0.5j * alpha * k * u_sq_hat

    mass = np.zeros(n_t)
    l2 = np.zeros(n_t)
    hamiltonian = np.zeros(n_t)
    mass[0], l2[0], hamiltonian[0] = compute_kdv_invariants(u, dx=dx, k=k)

    for step in range(1, n_t):
        n_v = _nonlinear(v)
        a = e_half * v + q * n_v
        n_a = _nonlinear(a)
        b = e_half * v + q * n_a
        n_b = _nonlinear(b)
        c_stage = e_half * a + q * (2.0 * n_b - n_v)
        n_c = _nonlinear(c_stage)
        v = e * v + f1 * n_v + 2.0 * f2 * (n_a + n_b) + f3 * n_c
        v[~dealias_mask] = 0.0
        u = np.fft.ifft(v).real
        u_hist[step] = u
        mass[step], l2[step], hamiltonian[step] = compute_kdv_invariants(u, dx=dx, k=k)

    magnitude = u_hist.copy()
    invariants = {"mass": mass, "l2": l2, "hamiltonian": hamiltonian}
    return u_hist, magnitude, invariants


def solve_nonlinear_waves(
    *,
    model_type: str,
    x_min: float,
    x_max: float,
    nx: int,
    t_min: float = 0.0,
    t_max: float = 10.0,
    dt: float = 0.002,
    profile: str = "sech",
    amplitude: float = 1.0,
    sigma: float = 0.3,
    center: float = 0.0,
    custom_profile_fn=None,
    initial_phase_k: float = 0.0,
    # NLSE parameters
    beta2: float = 1.0,
    gamma: float = 1.0,
    # KdV parameters
    c: float = 0.0,
    alpha: float = 6.0,
    beta_disp: float = 1.0,
) -> NonlinearWavesResult:
    """Solve nonlinear wave propagation for NLSE or KdV."""
    mtype = model_type.lower().strip()
    if mtype not in _MODEL_TYPES:
        raise ValueError(f"model_type must be one of {sorted(_MODEL_TYPES)}")
    x, dx, k = build_periodic_grid(x_min, x_max, nx)
    t = _build_time_grid(t_min, t_max, dt)

    base_profile = build_initial_profile(
        x,
        profile=profile,
        amplitude=amplitude,
        sigma=sigma,
        center=center,
        custom_fn=custom_profile_fn,
    )

    logger.info(
        "Solving nonlinear waves: model=%s nx=%d t=[%g,%g] dt=%g",
        mtype,
        nx,
        t_min,
        t_max,
        dt,
    )

    if mtype == "nlse":
        psi0 = base_profile.astype(complex) * np.exp(1j * initial_phase_k * x)
        field, magnitude, invariants = _simulate_nlse(
            x=x,
            t=t,
            dx=dx,
            k=k,
            psi0=psi0,
            beta2=beta2,
            gamma=gamma,
        )
        phase = np.angle(field)
        final_spectrum = np.abs(np.fft.fftshift(np.fft.fft(field[-1]))) ** 2
        k_shift = np.fft.fftshift(k)
        ref = abs(invariants["norm"][0]) + 1e-12
        magnitudes = {
            "norm_drift_rel": float((invariants["norm"][-1] - invariants["norm"][0]) / ref),
            "max_intensity": float(np.max(magnitude)),
        }
        metadata = {
            "model_type": "nlse",
            "beta2": float(beta2),
            "gamma": float(gamma),
            "initial_phase_k": float(initial_phase_k),
            "profile": profile,
        }
    else:
        u0 = base_profile
        field, magnitude, invariants = _simulate_kdv(
            x=x,
            t=t,
            dx=dx,
            k=k,
            u0=u0,
            c=c,
            alpha=alpha,
            beta_disp=beta_disp,
        )
        phase = None
        final_spectrum = np.abs(np.fft.fftshift(np.fft.fft(field[-1]))) ** 2
        k_shift = np.fft.fftshift(k)
        ref = abs(invariants["mass"][0]) + 1e-12
        magnitudes = {
            "mass_drift_rel": float((invariants["mass"][-1] - invariants["mass"][0]) / ref),
            "max_amplitude": float(np.max(np.abs(field))),
        }
        metadata = {
            "model_type": "kdv",
            "c": float(c),
            "alpha": float(alpha),
            "beta_disp": float(beta_disp),
            "profile": profile,
        }

    metadata.update(
        {
            "x_min": float(x_min),
            "x_max": float(x_max),
            "nx": int(nx),
            "dt": float(t[1] - t[0]) if len(t) > 1 else dt,
            "t_min": float(t[0]),
            "t_max": float(t[-1]),
        }
    )

    return NonlinearWavesResult(
        model_type=mtype,
        x=x,
        t=t,
        field=field,
        magnitude=magnitude,
        phase=phase,
        k=k_shift,
        spectrum_power=final_spectrum,
        invariants=invariants,
        metadata=metadata,
        magnitudes=magnitudes,
    )
