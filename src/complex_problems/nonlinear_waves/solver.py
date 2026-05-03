"""Pseudo-spectral solvers for nonlinear wave models (NLSE and KdV)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
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
    k: np.ndarray
    spectrum_power: np.ndarray
    invariants: dict[str, np.ndarray]
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)
    magnitudes: dict[str, float] = dataclass_field(default_factory=dict)
    _phase_supported: bool = dataclass_field(default=True, repr=False)
    _magnitude_cache: np.ndarray | None = dataclass_field(default=None, init=False, repr=False)
    _phase_cache: np.ndarray | None = dataclass_field(default=None, init=False, repr=False)

    @property
    def magnitude(self) -> np.ndarray:
        """Return intensity/amplitude history, materializing it only when needed."""
        if self._magnitude_cache is None:
            self._magnitude_cache = (
                np.abs(self.field) ** 2 if np.iscomplexobj(self.field) else self.field
            )
        return self._magnitude_cache

    @property
    def phase(self) -> np.ndarray | None:
        """Return phase history for complex fields only."""
        if not self._phase_supported:
            return None
        if self._phase_cache is None:
            self._phase_cache = np.angle(self.field)
        return self._phase_cache


def _build_time_grid(t_min: float, t_max: float, dt: float) -> np.ndarray:
    if t_max <= t_min:
        raise ValueError("t_max must be greater than t_min.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    n_steps = int(np.ceil((t_max - t_min) / dt))
    if n_steps < 1:
        n_steps = 1
    return np.linspace(t_min, t_max, n_steps + 1)


def _stored_step_indices(n_steps: int, store_every: int) -> np.ndarray:
    """Return solver-step indices to keep, always including first and final."""
    if store_every < 1:
        raise ValueError("store_every must be >= 1.")
    steps = np.arange(0, n_steps + 1, store_every, dtype=int)
    if steps[-1] != n_steps:
        steps = np.append(steps, n_steps)
    return steps


def _simulate_nlse(
    *,
    x: np.ndarray,
    t: np.ndarray,
    stored_steps: np.ndarray,
    dx: float,
    k: np.ndarray,
    psi0: np.ndarray,
    beta2: float,
    gamma: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, float]:
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.0
    n_steps, nx = len(t) - 1, len(x)
    n_stored = len(stored_steps)
    psi_hist = np.zeros((n_stored, nx), dtype=complex)
    psi = np.array(psi0, dtype=complex, copy=True)
    psi_hist[0] = psi

    lin_half = np.exp(-0.5j * beta2 * (k**2) * dt)
    norms = np.zeros(n_stored)
    momenta = np.zeros(n_stored)
    energies = np.zeros(n_stored)
    norms[0], momenta[0], energies[0] = compute_nlse_invariants(
        psi, dx=dx, k=k, beta2=beta2, gamma=gamma
    )
    max_intensity = float(np.max(np.abs(psi) ** 2))
    store_pos = 1

    for step in range(1, n_steps + 1):
        psi = np.fft.ifft(lin_half * np.fft.fft(psi))
        psi = psi * np.exp(-1j * gamma * np.abs(psi) ** 2 * dt)
        psi = np.fft.ifft(lin_half * np.fft.fft(psi))
        max_intensity = max(max_intensity, float(np.max(np.abs(psi) ** 2)))
        if store_pos < n_stored and step == int(stored_steps[store_pos]):
            psi_hist[store_pos] = psi
            norms[store_pos], momenta[store_pos], energies[store_pos] = compute_nlse_invariants(
                psi, dx=dx, k=k, beta2=beta2, gamma=gamma
            )
            store_pos += 1

    invariants = {"norm": norms, "momentum": momenta, "hamiltonian": energies}
    return psi_hist, invariants, psi, max_intensity


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
    stored_steps: np.ndarray,
    dx: float,
    k: np.ndarray,
    u0: np.ndarray,
    c: float,
    alpha: float,
    beta_disp: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, float]:
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.0
    n_steps, nx = len(t) - 1, len(x)
    n_stored = len(stored_steps)
    u_hist = np.zeros((n_stored, nx), dtype=float)
    u = np.array(u0, dtype=float, copy=True)
    u_hist[0] = u

    k_max = float(np.max(np.abs(k)))
    dealias_mask = np.abs(k) <= (2.0 / 3.0) * k_max + 1e-12
    linear_op = 1j * (beta_disp * (k**3) - c * k)

    mass = np.zeros(n_stored)
    l2 = np.zeros(n_stored)
    hamiltonian = np.zeros(n_stored)
    mass[0], l2[0], hamiltonian[0] = compute_kdv_invariants(u, dx=dx, k=k)
    max_amplitude = float(np.max(np.abs(u)))
    store_pos = 1

    e, e_half, q, f1, f2, f3 = _kdv_etdrk4_coefficients(linear_op, dt=dt)
    v = np.fft.fft(u)
    v[~dealias_mask] = 0.0

    def _nonlinear(v_hat: np.ndarray) -> np.ndarray:
        u_state = np.fft.ifft(v_hat).real
        u_sq_hat = np.fft.fft(u_state * u_state)
        u_sq_hat[~dealias_mask] = 0.0
        return -0.5j * alpha * k * u_sq_hat

    for step in range(1, n_steps + 1):
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
        max_amplitude = max(max_amplitude, float(np.max(np.abs(u))))
        if store_pos < n_stored and step == int(stored_steps[store_pos]):
            u_hist[store_pos] = u
            mass[store_pos], l2[store_pos], hamiltonian[store_pos] = compute_kdv_invariants(
                u, dx=dx, k=k
            )
            store_pos += 1

    invariants = {"mass": mass, "l2": l2, "hamiltonian": hamiltonian}
    return u_hist, invariants, u, max_amplitude


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
    custom_profile_fn: Callable[[float], float] | None = None,
    initial_phase_k: float = 0.0,
    # NLSE parameters
    beta2: float = 1.0,
    gamma: float = 1.0,
    # KdV parameters
    c: float = 0.0,
    alpha: float = 6.0,
    beta_disp: float = 1.0,
    store_every: int = 1,
) -> NonlinearWavesResult:
    """Solve nonlinear wave propagation for NLSE or KdV."""
    mtype = model_type.lower().strip()
    if mtype not in _MODEL_TYPES:
        raise ValueError(f"model_type must be one of {sorted(_MODEL_TYPES)}")
    if store_every < 1:
        raise ValueError("store_every must be >= 1.")
    x, dx, k = build_periodic_grid(x_min, x_max, nx)
    t = _build_time_grid(t_min, t_max, dt)
    n_steps = len(t) - 1
    stored_steps = _stored_step_indices(n_steps, store_every)
    t_stored = t[stored_steps]

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
        field, invariants, final_field, max_intensity = _simulate_nlse(
            x=x,
            t=t,
            stored_steps=stored_steps,
            dx=dx,
            k=k,
            psi0=psi0,
            beta2=beta2,
            gamma=gamma,
        )
        final_spectrum = np.abs(np.fft.fftshift(np.fft.fft(final_field))) ** 2
        k_shift = np.fft.fftshift(k)
        ref = abs(invariants["norm"][0]) + 1e-12
        magnitudes = {
            "norm_drift_rel": float((invariants["norm"][-1] - invariants["norm"][0]) / ref),
            "max_intensity": max_intensity,
        }
        metadata = {
            "model_type": "nlse",
            "beta2": float(beta2),
            "gamma": float(gamma),
            "initial_phase_k": float(initial_phase_k),
            "profile": profile,
        }
        phase_supported = True
    else:
        u0 = base_profile
        field, invariants, final_field, max_amplitude = _simulate_kdv(
            x=x,
            t=t,
            stored_steps=stored_steps,
            dx=dx,
            k=k,
            u0=u0,
            c=c,
            alpha=alpha,
            beta_disp=beta_disp,
        )
        final_spectrum = np.abs(np.fft.fftshift(np.fft.fft(final_field))) ** 2
        k_shift = np.fft.fftshift(k)
        ref = abs(invariants["mass"][0]) + 1e-12
        magnitudes = {
            "mass_drift_rel": float((invariants["mass"][-1] - invariants["mass"][0]) / ref),
            "max_amplitude": max_amplitude,
        }
        metadata = {
            "model_type": "kdv",
            "c": float(c),
            "alpha": float(alpha),
            "beta_disp": float(beta_disp),
            "profile": profile,
        }
        phase_supported = False

    metadata.update(
        {
            "x_min": float(x_min),
            "x_max": float(x_max),
            "nx": int(nx),
            "dt": float(t_stored[1] - t_stored[0]) if len(t_stored) > 1 else dt,
            "solver_dt": float(dt),
            "solver_steps": int(n_steps),
            "store_every": int(store_every),
            "stored_steps": int(len(stored_steps)),
            "t_min": float(t_stored[0]),
            "t_max": float(t_stored[-1]),
        }
    )

    return NonlinearWavesResult(
        model_type=mtype,
        x=x,
        t=t_stored,
        field=field,
        k=k_shift,
        spectrum_power=final_spectrum,
        invariants=invariants,
        metadata=metadata,
        magnitudes=magnitudes,
        _phase_supported=phase_supported,
    )
