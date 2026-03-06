"""Steady and transient 1D pipe-flow solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from complex_problems.pipe_flow.model import (
    area_from_diameter,
    build_pipe_grid,
    diameter_profile,
    friction_factor,
    reynolds_number,
)
from utils import get_logger

logger = get_logger(__name__)

_MODEL_TYPES = {"steady", "transient"}


@dataclass
class PipeFlowResult:
    """Result bundle for pipe-flow simulations."""

    model_type: str
    x: np.ndarray
    t: np.ndarray
    diameter: np.ndarray
    area: np.ndarray
    pressure: np.ndarray
    velocity: np.ndarray
    reynolds: np.ndarray
    friction: np.ndarray
    flow_rate_mean: np.ndarray
    flow_rate_std: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    magnitudes: dict[str, float] = field(default_factory=dict)


def _integrate_pressure_from_dpdx(x: np.ndarray, p0: float, dpdx: np.ndarray) -> np.ndarray:
    p = np.zeros_like(x, dtype=float)
    p[0] = p0
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        p[i] = p[i - 1] - 0.5 * (dpdx[i - 1] + dpdx[i]) * dx
    return p


def _solve_steady(
    *,
    x: np.ndarray,
    diameter: np.ndarray,
    area: np.ndarray,
    rho: float,
    mu: float,
    roughness: float,
    p_in: float,
    p_out: float,
    friction_model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    target_dp = float(p_in - p_out)
    if abs(target_dp) < 1e-14:
        u = np.zeros_like(x)
        re = reynolds_number(rho, mu, u, diameter)
        f = friction_factor(re, roughness=roughness, diameter=diameter, model=friction_model)
        p = np.full_like(x, fill_value=p_in, dtype=float)
        return p, u, re, f, 0.0

    sign = 1.0 if target_dp > 0 else -1.0
    target_abs = abs(target_dp)

    def residual(q_abs: float) -> float:
        u_abs = q_abs / area
        re = reynolds_number(rho, mu, u_abs, diameter)
        f = friction_factor(re, roughness=roughness, diameter=diameter, model=friction_model)
        dpdx = f * rho * u_abs * u_abs / (2.0 * diameter)
        dp = float(np.trapezoid(dpdx, x))
        return dp - target_abs

    q_hi = max(1e-10, 0.01 * float(np.mean(area)))
    while residual(q_hi) < 0.0 and q_hi < 1e4:
        q_hi *= 2.0
    q_lo = 0.0
    for _ in range(80):
        q_mid = 0.5 * (q_lo + q_hi)
        if residual(q_mid) > 0.0:
            q_hi = q_mid
        else:
            q_lo = q_mid
    q_abs = 0.5 * (q_lo + q_hi)
    q = sign * q_abs

    u = q / area
    re = reynolds_number(rho, mu, u, diameter)
    f = friction_factor(re, roughness=roughness, diameter=diameter, model=friction_model)
    dpdx_signed = f * rho * u * np.abs(u) / (2.0 * diameter)
    p = _integrate_pressure_from_dpdx(x, p_in, dpdx_signed)
    return p, u, re, f, float(q)


def _spatial_derivative(a: np.ndarray, dx: float) -> np.ndarray:
    out = np.zeros_like(a)
    out[1:-1] = (a[2:] - a[:-2]) / (2.0 * dx)
    out[0] = (a[1] - a[0]) / dx
    out[-1] = (a[-1] - a[-2]) / dx
    return out


def _solve_transient(
    *,
    x: np.ndarray,
    dx: float,
    diameter: np.ndarray,
    area: np.ndarray,
    rho: float,
    mu: float,
    roughness: float,
    friction_model: str,
    p_base: float,
    p_out: float,
    p_amp: float,
    p_freq_hz: float,
    wave_speed: float,
    damping: float,
    t_max: float,
    dt: float,
    sample_every: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfl = wave_speed * dt / dx
    if cfl > 0.95:
        raise ValueError(
            f"Unstable transient configuration: c*dt/dx={cfl:.3f} > 0.95. "
            "Reduce dt or increase nx spacing."
        )

    n_steps = int(np.ceil(t_max / dt))
    sample_indices = list(range(0, n_steps + 1, sample_every))
    if sample_indices[-1] != n_steps:
        sample_indices.append(n_steps)
    n_samples = len(sample_indices)
    t_hist = np.zeros(n_samples, dtype=float)

    p = np.linspace(p_base, p_out, len(x), dtype=float)
    u = np.zeros_like(x, dtype=float)

    p_hist = np.zeros((n_samples, len(x)), dtype=float)
    u_hist = np.zeros((n_samples, len(x)), dtype=float)
    re_hist = np.zeros((n_samples, len(x)), dtype=float)
    f_hist = np.zeros((n_samples, len(x)), dtype=float)
    q_mean_hist = np.zeros(n_samples, dtype=float)
    q_std_hist = np.zeros(n_samples, dtype=float)

    def p_inlet(t: float) -> float:
        return p_base + p_amp * np.sin(2.0 * np.pi * p_freq_hz * t)

    def apply_bc(u_arr: np.ndarray, p_arr: np.ndarray, t_now: float) -> None:
        p_arr[0] = p_inlet(t_now)
        p_arr[-1] = p_out
        u_arr[0] = u_arr[1]
        u_arr[-1] = u_arr[-2]

    def rhs(u_arr: np.ndarray, p_arr: np.ndarray, t_now: float) -> tuple[np.ndarray, np.ndarray]:
        u_loc = u_arr.copy()
        p_loc = p_arr.copy()
        apply_bc(u_loc, p_loc, t_now)

        dpdx = _spatial_derivative(p_loc, dx)
        dudx = _spatial_derivative(u_loc, dx)
        re = reynolds_number(rho, mu, u_loc, diameter)
        ff = friction_factor(re, roughness=roughness, diameter=diameter, model=friction_model)
        wall = ff * u_loc * np.abs(u_loc) / (2.0 * np.maximum(diameter, 1e-9))

        du_dt = -(1.0 / rho) * dpdx - damping * u_loc - wall
        dp_dt = -(rho * wave_speed * wave_speed) * dudx
        du_dt[0] = 0.0
        du_dt[-1] = 0.0
        dp_dt[0] = 0.0
        dp_dt[-1] = 0.0
        return du_dt, dp_dt

    def sample(idx: int, t_now: float) -> None:
        re = reynolds_number(rho, mu, u, diameter)
        ff = friction_factor(re, roughness=roughness, diameter=diameter, model=friction_model)
        q_profile = u * area
        t_hist[idx] = t_now
        p_hist[idx] = p
        u_hist[idx] = u
        re_hist[idx] = re
        f_hist[idx] = ff
        q_mean_hist[idx] = float(np.mean(q_profile))
        q_std_hist[idx] = float(np.std(q_profile))

    sample_pos = 0
    apply_bc(u, p, 0.0)
    sample(sample_pos, 0.0)
    sample_pos += 1

    for step in range(1, n_steps + 1):
        t_now = (step - 1) * dt
        k1_u, k1_p = rhs(u, p, t_now)
        k2_u, k2_p = rhs(u + 0.5 * dt * k1_u, p + 0.5 * dt * k1_p, t_now + 0.5 * dt)
        k3_u, k3_p = rhs(u + 0.5 * dt * k2_u, p + 0.5 * dt * k2_p, t_now + 0.5 * dt)
        k4_u, k4_p = rhs(u + dt * k3_u, p + dt * k3_p, t_now + dt)

        u = u + (dt / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
        p = p + (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
        apply_bc(u, p, step * dt)

        if step in sample_indices:
            sample(sample_pos, step * dt)
            sample_pos += 1

    return t_hist, p_hist, u_hist, re_hist, f_hist, q_mean_hist, q_std_hist


def solve_pipe_flow(
    *,
    model_type: str = "steady",
    length: float = 20.0,
    nx: int = 256,
    profile: str = "constant",
    d_in: float = 0.08,
    d_out: float = 0.05,
    d0: float = 0.06,
    profile_amplitude: float = 0.20,
    profile_waves: float = 2.0,
    custom_diameter_fn=None,
    rho: float = 1000.0,
    mu: float = 1.0e-3,
    roughness: float = 1.0e-5,
    friction_model: str = "auto",
    # Steady settings
    p_in: float = 2.0e5,
    p_out: float = 1.9e5,
    # Transient settings
    p_base: float = 2.0e5,
    p_amp: float = 2.0e3,
    p_freq_hz: float = 2.0,
    wave_speed: float = 200.0,
    damping: float = 0.2,
    t_max: float = 1.0,
    dt: float = 5.0e-4,
    sample_every: int = 10,
) -> PipeFlowResult:
    """Solve steady or transient 1D pipe flow."""
    mt = model_type.strip().lower()
    if mt not in _MODEL_TYPES:
        raise ValueError(f"model_type must be one of {sorted(_MODEL_TYPES)}")
    if rho <= 0 or mu <= 0:
        raise ValueError("rho and mu must be positive.")

    x, dx = build_pipe_grid(length, nx)
    diameter = diameter_profile(
        x,
        profile=profile,
        d_in=d_in,
        d_out=d_out,
        d0=d0,
        amplitude=profile_amplitude,
        n_waves=profile_waves,
        custom_fn=custom_diameter_fn,
    )
    area = area_from_diameter(diameter)

    if mt == "steady":
        p, u, re, ff, q = _solve_steady(
            x=x,
            diameter=diameter,
            area=area,
            rho=rho,
            mu=mu,
            roughness=roughness,
            p_in=p_in,
            p_out=p_out,
            friction_model=friction_model,
        )
        t = np.array([0.0], dtype=float)
        pressure = p[np.newaxis, :]
        velocity = u[np.newaxis, :]
        reynolds = re[np.newaxis, :]
        friction = ff[np.newaxis, :]
        q_mean = np.array([q], dtype=float)
        q_std = np.array([float(np.std(u * area))], dtype=float)
        magnitudes = {
            "flow_rate_m3s": float(q),
            "mean_velocity_ms": float(np.mean(u)),
            "re_max": float(np.max(re)),
            "dp_target_pa": float(p_in - p_out),
            "dp_solved_pa": float(p[0] - p[-1]),
        }
        metadata = {
            "model_type": mt,
            "length": float(length),
            "nx": int(nx),
            "profile": profile,
            "friction_model": friction_model,
            "rho": float(rho),
            "mu": float(mu),
            "roughness": float(roughness),
            "p_in": float(p_in),
            "p_out": float(p_out),
        }
    else:
        t, pressure, velocity, reynolds, friction, q_mean, q_std = _solve_transient(
            x=x,
            dx=dx,
            diameter=diameter,
            area=area,
            rho=rho,
            mu=mu,
            roughness=roughness,
            friction_model=friction_model,
            p_base=p_base,
            p_out=p_out,
            p_amp=p_amp,
            p_freq_hz=p_freq_hz,
            wave_speed=wave_speed,
            damping=damping,
            t_max=t_max,
            dt=dt,
            sample_every=sample_every,
        )
        cfl = wave_speed * dt / dx
        magnitudes = {
            "max_pressure_pa": float(np.max(pressure)),
            "max_velocity_ms": float(np.max(np.abs(velocity))),
            "re_max": float(np.max(reynolds)),
            "cfl": float(cfl),
            "mean_q_std_m3s": float(np.mean(q_std)),
        }
        metadata = {
            "model_type": mt,
            "length": float(length),
            "nx": int(nx),
            "profile": profile,
            "friction_model": friction_model,
            "rho": float(rho),
            "mu": float(mu),
            "roughness": float(roughness),
            "p_base": float(p_base),
            "p_out": float(p_out),
            "p_amp": float(p_amp),
            "p_freq_hz": float(p_freq_hz),
            "wave_speed": float(wave_speed),
            "damping": float(damping),
            "t_max": float(t_max),
            "dt": float(dt),
            "sample_every": int(sample_every),
        }

    logger.info(
        "Solved pipe_flow: model=%s nx=%d profile=%s",
        mt,
        nx,
        profile,
    )
    return PipeFlowResult(
        model_type=mt,
        x=x,
        t=t,
        diameter=diameter,
        area=area,
        pressure=pressure,
        velocity=velocity,
        reynolds=reynolds,
        friction=friction,
        flow_rate_mean=q_mean,
        flow_rate_std=q_std,
        metadata=metadata,
        magnitudes=magnitudes,
    )
