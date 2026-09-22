"""Lightweight 3D incompressible Navier--Stokes solver on a periodic grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from complex_problems.aerodynamics_3d.model import (
    build_obstacle_mask,
    build_periodic_domain,
    ddx_periodic,
    ddy_periodic,
    ddz_periodic,
    laplacian_periodic,
    spectral_divergence_l2,
)
from utils import get_logger

logger = get_logger(__name__)

_APPROXIMATIONS = {"nonlinear_ns", "stokes"}
_SHAPES = {"sphere", "ellipsoid", "box", "naca0012_wing"}


@dataclass
class Aerodynamics3DResult:
    """Core saved histories and compact diagnostics for a 3D solve."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    t: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    pressure: np.ndarray
    obstacle_mask: np.ndarray
    drag_coeff: np.ndarray
    lift_coeff: np.ndarray
    side_force_coeff: np.ndarray
    divergence_l2: np.ndarray
    max_speed: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    magnitudes: dict[str, float] = field(default_factory=dict)


def _wave_numbers(
    *, nx: int, ny: int, nz: int, dx: float, dy: float, dz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return broadcast Fourier wave numbers in ``(z, y, x)`` order."""

    def frequencies(size: int, spacing: float) -> np.ndarray:
        values = 2.0 * np.pi * np.fft.fftfreq(size, d=spacing)
        # The Nyquist coefficient has no distinct negative partner.  Removing
        # it keeps real-valued projected fields Hermitian in every dimension.
        if size % 2 == 0:
            values[size // 2] = 0.0
        return values

    kx = frequencies(nx, dx)[None, None, :]
    ky = frequencies(ny, dy)[None, :, None]
    kz = frequencies(nz, dz)[:, None, None]
    k2 = kx * kx + ky * ky + kz * kz
    return (
        np.broadcast_to(kx, (nz, ny, nx)),
        np.broadcast_to(ky, (nz, ny, nx)),
        np.broadcast_to(kz, (nz, ny, nx)),
        k2,
    )


def project_velocity_fft(
    u_star: np.ndarray,
    v_star: np.ndarray,
    w_star: np.ndarray,
    *,
    dt: float,
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
    k2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the 3D Fourier Helmholtz projection and recover projection pressure.

    The pressure convention is ``u_new = u_star - dt * grad(p)``.  The zero
    Fourier mode is deliberately left unchanged and has zero pressure.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    u_hat = np.fft.fftn(u_star)
    v_hat = np.fft.fftn(v_star)
    w_hat = np.fft.fftn(w_star)
    d = kx * u_hat + ky * v_hat + kz * w_hat
    mask = k2 > 0.0
    u_new_hat = u_hat.copy()
    v_new_hat = v_hat.copy()
    w_new_hat = w_hat.copy()
    u_new_hat[mask] -= kx[mask] * d[mask] / k2[mask]
    v_new_hat[mask] -= ky[mask] * d[mask] / k2[mask]
    w_new_hat[mask] -= kz[mask] * d[mask] / k2[mask]
    pressure_hat = np.zeros_like(d, dtype=complex)
    pressure_hat[mask] = -1j * d[mask] / (dt * k2[mask])
    pressure_hat[0, 0, 0] = 0.0
    return (
        np.real(np.fft.ifftn(u_new_hat)),
        np.real(np.fft.ifftn(v_new_hat)),
        np.real(np.fft.ifftn(w_new_hat)),
        np.real(np.fft.ifftn(pressure_hat)),
    )


def _sample_indices(n_steps: int, sample_every: int) -> list[int]:
    """Return deterministic saved-step indexes including the final step."""
    indices = list(range(0, n_steps + 1, sample_every))
    if indices[-1] != n_steps:
        indices.append(n_steps)
    return indices


def solve_aerodynamics_3d(
    *,
    approximation: str = "nonlinear_ns",
    nx: int = 48,
    ny: int = 32,
    nz: int = 32,
    lx: float = 4.0,
    ly: float = 2.0,
    lz: float = 2.0,
    t_max: float = 1.0,
    dt: float = 0.002,
    sample_every: int = 10,
    rho: float = 1.0,
    nu: float = 0.01,
    u_inf: float = 1.0,
    penalization: float = 0.005,
    obstacle_shape: str = "sphere",
    obstacle_center_x: float = 1.0,
    obstacle_center_y: float = 1.0,
    obstacle_center_z: float = 1.0,
    obstacle_diameter: float = 0.4,
    obstacle_size_x: float = 0.7,
    obstacle_size_y: float = 0.4,
    obstacle_size_z: float = 0.4,
    obstacle_chord: float = 0.8,
    obstacle_span: float = 0.8,
    obstacle_thickness_ratio: float = 0.12,
    obstacle_attack_deg: float = 0.0,
) -> Aerodynamics3DResult:
    """Solve incompressible flow in a periodic Cartesian box with penalization."""
    if approximation not in _APPROXIMATIONS:
        raise ValueError(f"approximation must be one of {sorted(_APPROXIMATIONS)}")
    if obstacle_shape not in _SHAPES:
        raise ValueError(f"obstacle_shape must be one of {sorted(_SHAPES)}")
    if t_max <= 0 or dt <= 0:
        raise ValueError("t_max and dt must be positive.")
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1.")
    if rho <= 0 or nu <= 0 or u_inf <= 0 or penalization <= 0:
        raise ValueError("rho, nu, u_inf, and penalization must be positive.")

    x, y, z, X, Y, Z, dx, dy, dz = build_periodic_domain(nx=nx, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz)
    obstacle_mask, reference_area, reference_length = build_obstacle_mask(
        shape=obstacle_shape,
        X=X,
        Y=Y,
        Z=Z,
        center_x=obstacle_center_x,
        center_y=obstacle_center_y,
        center_z=obstacle_center_z,
        diameter=obstacle_diameter,
        size_x=obstacle_size_x,
        size_y=obstacle_size_y,
        size_z=obstacle_size_z,
        chord=obstacle_chord,
        span=obstacle_span,
        thickness_ratio=obstacle_thickness_ratio,
        attack_deg=obstacle_attack_deg,
    )
    if not np.any(obstacle_mask):
        raise ValueError("Obstacle mask is empty; increase its size or move it into the domain.")

    n_steps = int(np.ceil(t_max / dt))
    t_end = n_steps * dt
    sample_indices = _sample_indices(n_steps, sample_every)
    n_samples = len(sample_indices)
    shape = (nz, ny, nx)
    u = np.full(shape, u_inf, dtype=float)
    v = np.zeros(shape, dtype=float)
    w = np.zeros(shape, dtype=float)
    pressure = np.zeros(shape, dtype=float)
    u[obstacle_mask] = 0.0
    v[obstacle_mask] = 0.0
    w[obstacle_mask] = 0.0
    mask_f = obstacle_mask.astype(float)
    fluid_mask = ~obstacle_mask
    kx, ky, kz, k2 = _wave_numbers(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
    u, v, w, pressure = project_velocity_fft(u, v, w, dt=dt, kx=kx, ky=ky, kz=kz, k2=k2)

    u_hist = np.empty((n_samples, *shape), dtype=float)
    v_hist = np.empty_like(u_hist)
    w_hist = np.empty_like(u_hist)
    p_hist = np.empty_like(u_hist)
    t_hist = np.empty(n_samples, dtype=float)
    cd_hist = np.empty(n_samples, dtype=float)
    cl_hist = np.empty(n_samples, dtype=float)
    cs_hist = np.empty(n_samples, dtype=float)
    div_l2_hist = np.empty(n_samples, dtype=float)
    max_speed_hist = np.empty(n_samples, dtype=float)
    q_ref = 0.5 * rho * u_inf * u_inf
    cell_volume = dx * dy * dz
    tau_drive = 0.2  # Matches the 2D plugin's mean-flow relaxation timescale.
    sample_pos = 0

    def save_sample(step: int) -> None:
        nonlocal sample_pos
        u_hist[sample_pos] = u
        v_hist[sample_pos] = v
        w_hist[sample_pos] = w
        p_hist[sample_pos] = pressure
        t_hist[sample_pos] = step * dt
        reaction = rho * cell_volume / penalization * mask_f
        cd_hist[sample_pos] = float(np.sum(reaction * u) / (q_ref * reference_area))
        cl_hist[sample_pos] = float(np.sum(reaction * v) / (q_ref * reference_area))
        cs_hist[sample_pos] = float(np.sum(reaction * w) / (q_ref * reference_area))
        div_l2_hist[sample_pos] = spectral_divergence_l2(u, v, w, kx, ky, kz)
        max_speed_hist[sample_pos] = float(np.max(np.sqrt(u * u + v * v + w * w)))
        sample_pos += 1

    save_sample(0)
    logger.info(
        "Solving aerodynamics_3d: approx=%s grid=%dx%dx%d t=[0,%g] dt=%g shape=%s",
        approximation,
        nx,
        ny,
        nz,
        t_end,
        dt,
        obstacle_shape,
    )

    for step in range(1, n_steps + 1):
        ux, uy, uz = ddx_periodic(u, dx), ddy_periodic(u, dy), ddz_periodic(u, dz)
        vx, vy, vz = ddx_periodic(v, dx), ddy_periodic(v, dy), ddz_periodic(v, dz)
        wx, wy, wz = ddx_periodic(w, dx), ddy_periodic(w, dy), ddz_periodic(w, dz)
        lap_u = laplacian_periodic(u, dx, dy, dz)
        lap_v = laplacian_periodic(v, dx, dy, dz)
        lap_w = laplacian_periodic(w, dx, dy, dz)
        if approximation == "stokes":
            conv_u = conv_v = conv_w = 0.0
        else:
            conv_u = u * ux + v * uy + w * uz
            conv_v = u * vx + v * vy + w * vz
            conv_w = u * wx + v * wy + w * wz

        mean_u = float(np.mean(u[fluid_mask])) if np.any(fluid_mask) else float(np.mean(u))
        drive_x = (u_inf - mean_u) / tau_drive
        pen_u = -mask_f * u / penalization
        pen_v = -mask_f * v / penalization
        pen_w = -mask_f * w / penalization
        u_star = u + dt * (-conv_u + nu * lap_u + pen_u + drive_x)
        v_star = v + dt * (-conv_v + nu * lap_v + pen_v)
        w_star = w + dt * (-conv_w + nu * lap_w + pen_w)
        u_star[obstacle_mask] = 0.0
        v_star[obstacle_mask] = 0.0
        w_star[obstacle_mask] = 0.0
        u, v, w, pressure = project_velocity_fft(
            u_star, v_star, w_star, dt=dt, kx=kx, ky=ky, kz=kz, k2=k2
        )
        if sample_pos < n_samples and step == sample_indices[sample_pos]:
            save_sample(step)

    tail = max(3, len(cd_hist) // 4)
    magnitudes = {
        "reynolds": float(u_inf * reference_length / nu),
        "mean_cd_tail": float(np.mean(cd_hist[-tail:])),
        "rms_cl": float(np.sqrt(np.mean(cl_hist * cl_hist))),
        "rms_cs": float(np.sqrt(np.mean(cs_hist * cs_hist))),
        "max_divergence_l2": float(np.max(div_l2_hist)),
        "max_speed": float(np.max(max_speed_hist)),
    }
    metadata = {
        "approximation": approximation,
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "lx": float(lx),
        "ly": float(ly),
        "lz": float(lz),
        "dt": float(dt),
        "t_max": float(t_end),
        "sample_every": int(sample_every),
        "rho": float(rho),
        "nu": float(nu),
        "u_inf": float(u_inf),
        "penalization": float(penalization),
        "obstacle_shape": obstacle_shape,
        "obstacle_center_x": float(obstacle_center_x),
        "obstacle_center_y": float(obstacle_center_y),
        "obstacle_center_z": float(obstacle_center_z),
        "obstacle_diameter": float(obstacle_diameter),
        "obstacle_size_x": float(obstacle_size_x),
        "obstacle_size_y": float(obstacle_size_y),
        "obstacle_size_z": float(obstacle_size_z),
        "obstacle_chord": float(obstacle_chord),
        "obstacle_span": float(obstacle_span),
        "obstacle_thickness_ratio": float(obstacle_thickness_ratio),
        "obstacle_attack_deg": float(obstacle_attack_deg),
        "reference_area": float(reference_area),
        "reference_length": float(reference_length),
        "stored_fields": ("u", "v", "w", "pressure"),
        "derived_fields": ("speed", "vorticity_vector", "vorticity_magnitude"),
    }
    return Aerodynamics3DResult(
        x=x,
        y=y,
        z=z,
        t=t_hist,
        u=u_hist,
        v=v_hist,
        w=w_hist,
        pressure=p_hist,
        obstacle_mask=obstacle_mask,
        drag_coeff=cd_hist,
        lift_coeff=cl_hist,
        side_force_coeff=cs_hist,
        divergence_l2=div_l2_hist,
        max_speed=max_speed_hist,
        metadata=metadata,
        magnitudes=magnitudes,
    )
