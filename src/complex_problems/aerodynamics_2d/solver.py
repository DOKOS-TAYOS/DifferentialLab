"""2D incompressible aerodynamics solver with obstacle penalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from complex_problems.aerodynamics_2d.model import (
    build_obstacle_mask,
    build_periodic_domain,
    ddx_periodic,
    ddy_periodic,
    divergence_periodic,
    laplacian_periodic,
    vorticity_periodic,
)
from utils import get_logger

logger = get_logger(__name__)

_APPROXIMATIONS = {"nonlinear_ns", "stokes"}
_SHAPES = {"cylinder", "ellipse", "rectangle", "naca0012"}


@dataclass
class Aerodynamics2DResult:
    """Result bundle for 2D aerodynamics simulation."""

    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    u: np.ndarray
    v: np.ndarray
    pressure: np.ndarray
    speed: np.ndarray
    vorticity: np.ndarray
    obstacle_mask: np.ndarray
    drag_coeff: np.ndarray
    lift_coeff: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    magnitudes: dict[str, float] = field(default_factory=dict)


def _project_div_free_fft(
    u_star: np.ndarray,
    v_star: np.ndarray,
    *,
    dt: float,
    dx: float,
    dy: float,
    kx: np.ndarray,
    ky: np.ndarray,
    k2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project tentative velocity to divergence-free field using periodic FFT."""
    div_star = divergence_periodic(u_star, v_star, dx, dy)
    rhs_hat = np.fft.fft2(div_star / dt)
    phi_hat = np.zeros_like(rhs_hat, dtype=complex)

    mask = k2 > 0.0
    phi_hat[mask] = -rhs_hat[mask] / k2[mask]
    phi_hat[0, 0] = 0.0

    dphi_dx = np.fft.ifft2(1j * kx * phi_hat).real
    dphi_dy = np.fft.ifft2(1j * ky * phi_hat).real

    u_new = u_star - dt * dphi_dx
    v_new = v_star - dt * dphi_dy
    pressure = np.fft.ifft2(phi_hat).real / dt
    return u_new, v_new, pressure


def _snapshot_fields(
    u: np.ndarray,
    v: np.ndarray,
    pressure: np.ndarray,
    *,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(u * u + v * v)
    vort = vorticity_periodic(u, v, dx, dy)
    return speed, vort


def solve_aerodynamics_2d(
    *,
    approximation: str = "nonlinear_ns",
    nx: int = 96,
    ny: int = 64,
    lx: float = 4.0,
    ly: float = 2.0,
    t_max: float = 2.0,
    dt: float = 0.002,
    sample_every: int = 10,
    rho: float = 1.0,
    nu: float = 0.01,
    u_inf: float = 1.0,
    penalization: float = 0.005,
    obstacle_shape: str = "cylinder",
    obstacle_center_x: float = 1.3,
    obstacle_center_y: float = 1.0,
    obstacle_size_x: float = 0.30,
    obstacle_size_y: float = 0.30,
    obstacle_attack_deg: float = 0.0,
) -> Aerodynamics2DResult:
    """Solve 2D incompressible flow in a periodic box with an immersed obstacle."""
    if approximation not in _APPROXIMATIONS:
        raise ValueError(f"approximation must be one of {sorted(_APPROXIMATIONS)}")
    if obstacle_shape not in _SHAPES:
        raise ValueError(f"obstacle_shape must be one of {sorted(_SHAPES)}")
    if t_max <= 0 or dt <= 0:
        raise ValueError("t_max and dt must be positive.")
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1.")
    if rho <= 0 or nu <= 0:
        raise ValueError("rho and nu must be positive.")
    if u_inf <= 0:
        raise ValueError("u_inf must be positive.")
    if penalization <= 0:
        raise ValueError("penalization must be positive.")

    x, y, X, Y, dx, dy = build_periodic_domain(nx=nx, ny=ny, lx=lx, ly=ly)
    obstacle_mask, ref_length, ref_height = build_obstacle_mask(
        shape=obstacle_shape,
        X=X,
        Y=Y,
        center_x=obstacle_center_x,
        center_y=obstacle_center_y,
        size_x=obstacle_size_x,
        size_y=obstacle_size_y,
        attack_deg=obstacle_attack_deg,
    )
    if not np.any(obstacle_mask):
        raise ValueError(
            "Obstacle mask is empty; increase obstacle size or move center into domain."
        )

    n_steps = int(np.ceil(t_max / dt))
    t_end = n_steps * dt
    sample_indices = list(range(0, n_steps + 1, sample_every))
    if sample_indices[-1] != n_steps:
        sample_indices.append(n_steps)
    n_samples = len(sample_indices)

    u = np.full((ny, nx), fill_value=u_inf, dtype=float)
    v = np.zeros((ny, nx), dtype=float)
    p = np.zeros((ny, nx), dtype=float)
    mask_f = obstacle_mask.astype(float)
    u[obstacle_mask] = 0.0
    v[obstacle_mask] = 0.0

    kx_1d = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky_1d = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx, ky = np.meshgrid(kx_1d, ky_1d)
    k2 = kx * kx + ky * ky
    k2[0, 0] = 0.0

    u_hist = np.zeros((n_samples, ny, nx), dtype=float)
    v_hist = np.zeros((n_samples, ny, nx), dtype=float)
    p_hist = np.zeros((n_samples, ny, nx), dtype=float)
    speed_hist = np.zeros((n_samples, ny, nx), dtype=float)
    vort_hist = np.zeros((n_samples, ny, nx), dtype=float)
    t_hist = np.zeros(n_samples, dtype=float)
    cd_hist = np.zeros(n_samples, dtype=float)
    cl_hist = np.zeros(n_samples, dtype=float)
    div_l2_hist = np.zeros(n_samples, dtype=float)

    q_ref = 0.5 * rho * u_inf * u_inf
    area_ref = max(ref_height, 1e-9)
    tau_drive = 0.2

    fluid_mask = ~obstacle_mask
    sample_pos = 0
    speed0, vort0 = _snapshot_fields(u, v, p, dx=dx, dy=dy)
    u_hist[sample_pos] = u
    v_hist[sample_pos] = v
    p_hist[sample_pos] = p
    speed_hist[sample_pos] = speed0
    vort_hist[sample_pos] = vort0
    t_hist[sample_pos] = 0.0
    div0 = divergence_periodic(u, v, dx, dy)
    if np.any(fluid_mask):
        div_l2_hist[sample_pos] = float(np.sqrt(np.mean(div0[fluid_mask] ** 2)))
    else:
        div_l2_hist[sample_pos] = float(np.sqrt(np.mean(div0**2)))
    sample_pos += 1

    logger.info(
        "Solving aerodynamics_2d: approx=%s grid=%dx%d t=[0,%g] dt=%g shape=%s",
        approximation,
        nx,
        ny,
        t_end,
        dt,
        obstacle_shape,
    )

    for step in range(1, n_steps + 1):
        ux = ddx_periodic(u, dx)
        uy = ddy_periodic(u, dy)
        vx = ddx_periodic(v, dx)
        vy = ddy_periodic(v, dy)
        lap_u = laplacian_periodic(u, dx, dy)
        lap_v = laplacian_periodic(v, dx, dy)

        if approximation == "stokes":
            conv_u = 0.0
            conv_v = 0.0
        else:
            conv_u = u * ux + v * uy
            conv_v = u * vx + v * vy

        mean_u = (
            float(np.mean(u[fluid_mask])) if np.any(fluid_mask) else float(np.mean(u))
        )
        force_x = (u_inf - mean_u) / tau_drive

        pen_u = -(mask_f * u) / penalization
        pen_v = -(mask_f * v) / penalization

        u_star = u + dt * (-conv_u + nu * lap_u + pen_u + force_x)
        v_star = v + dt * (-conv_v + nu * lap_v + pen_v)
        u_star[obstacle_mask] = 0.0
        v_star[obstacle_mask] = 0.0

        u, v, p = _project_div_free_fft(
            u_star,
            v_star,
            dt=dt,
            dx=dx,
            dy=dy,
            kx=kx,
            ky=ky,
            k2=k2,
        )
        u[obstacle_mask] = 0.0
        v[obstacle_mask] = 0.0

        if step in sample_indices:
            speed, vort = _snapshot_fields(u, v, p, dx=dx, dy=dy)
            div = divergence_periodic(u, v, dx, dy)
            if np.any(fluid_mask):
                div_l2 = float(np.sqrt(np.mean(div[fluid_mask] ** 2)))
            else:
                div_l2 = float(np.sqrt(np.mean(div**2)))

            # Penalization force integrates obstacle-fluid interaction.
            fx = rho * float(np.sum(mask_f * u / penalization) * dx * dy)
            fy = rho * float(np.sum(mask_f * v / penalization) * dx * dy)
            cd = fx / (q_ref * area_ref)
            cl = fy / (q_ref * area_ref)

            u_hist[sample_pos] = u
            v_hist[sample_pos] = v
            p_hist[sample_pos] = p
            speed_hist[sample_pos] = speed
            vort_hist[sample_pos] = vort
            t_hist[sample_pos] = step * dt
            cd_hist[sample_pos] = cd
            cl_hist[sample_pos] = cl
            div_l2_hist[sample_pos] = div_l2
            sample_pos += 1

    re = u_inf * ref_length / nu
    tail = max(3, len(cd_hist) // 4)
    magnitudes = {
        "reynolds": float(re),
        "mean_cd_tail": float(np.mean(cd_hist[-tail:])),
        "rms_cl": float(np.sqrt(np.mean(cl_hist * cl_hist))),
        "max_speed": float(np.max(speed_hist)),
        "max_divergence_l2": float(np.max(div_l2_hist)),
    }
    metadata = {
        "approximation": approximation,
        "nx": int(nx),
        "ny": int(ny),
        "lx": float(lx),
        "ly": float(ly),
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
        "obstacle_size_x": float(obstacle_size_x),
        "obstacle_size_y": float(obstacle_size_y),
        "obstacle_attack_deg": float(obstacle_attack_deg),
        "reference_length": float(ref_length),
        "reference_height": float(ref_height),
    }
    return Aerodynamics2DResult(
        x=x,
        y=y,
        t=t_hist,
        u=u_hist,
        v=v_hist,
        pressure=p_hist,
        speed=speed_hist,
        vorticity=vort_hist,
        obstacle_mask=obstacle_mask,
        drag_coeff=cd_hist,
        lift_coeff=cl_hist,
        metadata=metadata,
        magnitudes=magnitudes,
    )
