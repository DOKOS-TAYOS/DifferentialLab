"""Solver for a 2D nonlinear membrane (coupled oscillator grid)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from complex_problems.membrane_2d.model import (
    acceleration_field,
    apply_fixed_boundary,
    compute_energy_terms,
    compute_fft_power_2d,
)
from utils import SolverFailedError, get_logger

logger = get_logger(__name__)


@dataclass
class Membrane2DResult:
    """Result bundle for the membrane solver."""

    t: np.ndarray
    displacement: np.ndarray  # (n_t, ny, nx)
    velocity: np.ndarray  # (n_t, ny, nx)
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray
    total_energy: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    spectrum_power: np.ndarray
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


def _integrate_verlet(
    *,
    t: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    mass: float,
    k_linear: float,
    boundary: str,
    alpha: float,
    beta: float,
    high_order_coeff: float,
    high_order_power: int,
) -> tuple[np.ndarray, np.ndarray]:
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.0
    n_t = len(t)
    ny, nx = u0.shape
    u_hist = np.zeros((n_t, ny, nx), dtype=float)
    v_hist = np.zeros((n_t, ny, nx), dtype=float)

    u = np.array(u0, dtype=float, copy=True)
    v = np.array(v0, dtype=float, copy=True)
    if boundary == "fixed":
        apply_fixed_boundary(u, v)

    a = acceleration_field(
        u,
        mass=mass,
        k_linear=k_linear,
        boundary=boundary,
        alpha=alpha,
        beta=beta,
        high_order_coeff=high_order_coeff,
        high_order_power=high_order_power,
    )
    u_hist[0] = u
    v_hist[0] = v

    for step in range(1, n_t):
        u = u + dt * v + 0.5 * dt * dt * a
        if boundary == "fixed":
            apply_fixed_boundary(u)

        a_new = acceleration_field(
            u,
            mass=mass,
            k_linear=k_linear,
            boundary=boundary,
            alpha=alpha,
            beta=beta,
            high_order_coeff=high_order_coeff,
            high_order_power=high_order_power,
        )
        v = v + 0.5 * dt * (a + a_new)
        if boundary == "fixed":
            apply_fixed_boundary(u, v)
        a = a_new

        u_hist[step] = u
        v_hist[step] = v

    return u_hist, v_hist


def _integrate_rk45(
    *,
    t: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    mass: float,
    k_linear: float,
    boundary: str,
    alpha: float,
    beta: float,
    high_order_coeff: float,
    high_order_power: int,
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = u0.shape
    size = ny * nx
    state0 = np.concatenate([u0.ravel(), v0.ravel()])

    def _rhs(_t: float, state: np.ndarray) -> np.ndarray:
        u = state[:size].reshape(ny, nx).copy()
        v = state[size:].reshape(ny, nx).copy()
        if boundary == "fixed":
            apply_fixed_boundary(u, v)

        a = acceleration_field(
            u,
            mass=mass,
            k_linear=k_linear,
            boundary=boundary,
            alpha=alpha,
            beta=beta,
            high_order_coeff=high_order_coeff,
            high_order_power=high_order_power,
        )
        dudt = v
        dvdt = a
        if boundary == "fixed":
            apply_fixed_boundary(dudt, dvdt)
        return np.concatenate([dudt.ravel(), dvdt.ravel()])

    sol = solve_ivp(
        fun=_rhs,
        t_span=(float(t[0]), float(t[-1])),
        y0=state0,
        method="RK45",
        t_eval=t,
        rtol=1e-6,
        atol=1e-9,
        dense_output=False,
    )
    if not sol.success:
        raise SolverFailedError(f"Membrane RK45 failed: {sol.message}")

    states = sol.y.T
    u_hist = states[:, :size].reshape(len(t), ny, nx)
    v_hist = states[:, size:].reshape(len(t), ny, nx)
    if boundary == "fixed":
        for idx in range(len(t)):
            apply_fixed_boundary(u_hist[idx], v_hist[idx])
    return u_hist, v_hist


def solve_membrane_2d(
    *,
    u0: np.ndarray,
    v0: np.ndarray | None = None,
    t_min: float = 0.0,
    t_max: float = 20.0,
    dt: float = 0.02,
    mass: float = 1.0,
    k_linear: float = 1.0,
    boundary: str = "fixed",
    integrator: str = "verlet",
    alpha: float = 0.0,
    beta: float = 0.0,
    high_order_coeff: float = 0.0,
    high_order_power: int = 5,
) -> Membrane2DResult:
    """Solve the membrane dynamics for the provided initial fields."""
    if boundary not in {"fixed", "periodic"}:
        raise ValueError("boundary must be 'fixed' or 'periodic'.")
    if integrator not in {"verlet", "rk45"}:
        raise ValueError("integrator must be 'verlet' or 'rk45'.")
    if mass <= 0:
        raise ValueError("mass must be positive.")
    if k_linear < 0:
        raise ValueError("k_linear must be non-negative.")
    if high_order_power < 2:
        raise ValueError("high_order_power must be >= 2.")

    u0_arr = np.asarray(u0, dtype=float)
    if u0_arr.ndim != 2:
        raise ValueError("u0 must be a 2D array.")
    ny, nx = u0_arr.shape

    if v0 is None:
        v0_arr = np.zeros_like(u0_arr)
    else:
        v0_arr = np.asarray(v0, dtype=float)
        if v0_arr.shape != u0_arr.shape:
            raise ValueError("v0 must have the same shape as u0.")

    t = _build_time_grid(t_min, t_max, dt)
    logger.info(
        "Solving 2D membrane: grid=%dx%d, t=[%g,%g], steps=%d, integrator=%s",
        nx,
        ny,
        t_min,
        t_max,
        len(t) - 1,
        integrator,
    )

    if integrator == "verlet":
        u_hist, v_hist = _integrate_verlet(
            t=t,
            u0=u0_arr,
            v0=v0_arr,
            mass=mass,
            k_linear=k_linear,
            boundary=boundary,
            alpha=alpha,
            beta=beta,
            high_order_coeff=high_order_coeff,
            high_order_power=high_order_power,
        )
    else:
        u_hist, v_hist = _integrate_rk45(
            t=t,
            u0=u0_arr,
            v0=v0_arr,
            mass=mass,
            k_linear=k_linear,
            boundary=boundary,
            alpha=alpha,
            beta=beta,
            high_order_coeff=high_order_coeff,
            high_order_power=high_order_power,
        )

    kinetic = np.zeros(len(t))
    potential = np.zeros(len(t))
    total = np.zeros(len(t))
    for idx in range(len(t)):
        ek, ep, et = compute_energy_terms(
            u_hist[idx],
            v_hist[idx],
            mass=mass,
            k_linear=k_linear,
            boundary=boundary,
            alpha=alpha,
            beta=beta,
            high_order_coeff=high_order_coeff,
            high_order_power=high_order_power,
        )
        kinetic[idx] = ek
        potential[idx] = ep
        total[idx] = et

    kx, ky, power = compute_fft_power_2d(u_hist[-1])
    e0 = abs(total[0]) + 1e-12
    magnitudes = {
        "energy_drift_rel": float((total[-1] - total[0]) / e0),
        "energy_drift_abs": float(total[-1] - total[0]),
        "max_displacement": float(np.max(np.abs(u_hist))),
        "max_velocity": float(np.max(np.abs(v_hist))),
    }
    metadata = {
        "nx": nx,
        "ny": ny,
        "dt": float(t[1] - t[0]) if len(t) > 1 else dt,
        "t_min": float(t[0]),
        "t_max": float(t[-1]),
        "mass": float(mass),
        "k_linear": float(k_linear),
        "boundary": boundary,
        "integrator": integrator,
        "alpha": float(alpha),
        "beta": float(beta),
        "high_order_coeff": float(high_order_coeff),
        "high_order_power": int(high_order_power),
    }

    return Membrane2DResult(
        t=t,
        displacement=u_hist,
        velocity=v_hist,
        kinetic_energy=kinetic,
        potential_energy=potential,
        total_energy=total,
        kx=kx,
        ky=ky,
        spectrum_power=power,
        metadata=metadata,
        magnitudes=magnitudes,
    )

