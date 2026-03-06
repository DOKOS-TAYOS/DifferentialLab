"""Geometry and field helpers for 2D aerodynamics."""

from __future__ import annotations

import numpy as np

_OBSTACLE_SHAPES = {"cylinder", "ellipse", "rectangle", "naca0012"}


def build_periodic_domain(
    *,
    nx: int,
    ny: int,
    lx: float,
    ly: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Build periodic domain coordinates and mesh."""
    if nx < 16 or ny < 16:
        raise ValueError("nx and ny must be >= 16.")
    if lx <= 0 or ly <= 0:
        raise ValueError("lx and ly must be positive.")
    x = np.linspace(0.0, lx, nx, endpoint=False)
    y = np.linspace(0.0, ly, ny, endpoint=False)
    dx = float(lx / nx)
    dy = float(ly / ny)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y, dx, dy


def rotate_coordinates(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate coordinates around obstacle center."""
    a = np.deg2rad(angle_deg)
    c = np.cos(a)
    s = np.sin(a)
    xr = X - center_x
    yr = Y - center_y
    xp = c * xr + s * yr
    yp = -s * xr + c * yr
    return xp, yp


def _mask_cylinder(xp: np.ndarray, yp: np.ndarray, *, diameter: float) -> np.ndarray:
    r = 0.5 * diameter
    return xp * xp + yp * yp <= r * r


def _mask_ellipse(xp: np.ndarray, yp: np.ndarray, *, size_x: float, size_y: float) -> np.ndarray:
    ax = 0.5 * size_x
    by = 0.5 * size_y
    return (xp / ax) ** 2 + (yp / by) ** 2 <= 1.0


def _mask_rectangle(xp: np.ndarray, yp: np.ndarray, *, size_x: float, size_y: float) -> np.ndarray:
    return (np.abs(xp) <= 0.5 * size_x) & (np.abs(yp) <= 0.5 * size_y)


def _mask_naca0012(
    xp: np.ndarray,
    yp: np.ndarray,
    *,
    chord: float,
    thickness_ratio: float,
) -> np.ndarray:
    x = xp + 0.5 * chord
    xc = x / chord
    t = thickness_ratio
    yt = 5.0 * t * chord * (
        0.2969 * np.sqrt(np.clip(xc, 0.0, None))
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        - 0.1015 * xc**4
    )
    return (x >= 0.0) & (x <= chord) & (np.abs(yp) <= yt)


def build_obstacle_mask(
    *,
    shape: str,
    X: np.ndarray,
    Y: np.ndarray,
    center_x: float,
    center_y: float,
    size_x: float,
    size_y: float,
    attack_deg: float,
) -> tuple[np.ndarray, float, float]:
    """Build obstacle mask and reference scales."""
    s = shape.strip().lower()
    if s not in _OBSTACLE_SHAPES:
        raise ValueError(f"Unknown obstacle shape '{shape}'.")
    if size_x <= 0 or size_y <= 0:
        raise ValueError("Obstacle sizes must be positive.")

    xp, yp = rotate_coordinates(
        X,
        Y,
        center_x=center_x,
        center_y=center_y,
        angle_deg=attack_deg,
    )
    if s == "cylinder":
        mask = _mask_cylinder(xp, yp, diameter=size_x)
        ref_length = size_x
        ref_height = size_x
    elif s == "ellipse":
        mask = _mask_ellipse(xp, yp, size_x=size_x, size_y=size_y)
        ref_length = size_x
        ref_height = size_y
    elif s == "rectangle":
        mask = _mask_rectangle(xp, yp, size_x=size_x, size_y=size_y)
        ref_length = size_x
        ref_height = size_y
    else:
        mask = _mask_naca0012(
            xp,
            yp,
            chord=size_x,
            thickness_ratio=size_y,
        )
        ref_length = size_x
        ref_height = max(1e-6, size_x * size_y)

    return mask, float(ref_length), float(ref_height)


def ddx_periodic(f: np.ndarray, dx: float) -> np.ndarray:
    """Periodic x-derivative."""
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)


def ddy_periodic(f: np.ndarray, dy: float) -> np.ndarray:
    """Periodic y-derivative."""
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dy)


def laplacian_periodic(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Periodic 2D Laplacian."""
    return (
        (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / (dx * dx)
        + (np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)) / (dy * dy)
    )


def divergence_periodic(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Periodic divergence."""
    return ddx_periodic(u, dx) + ddy_periodic(v, dy)


def vorticity_periodic(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Periodic scalar vorticity."""
    return ddx_periodic(v, dx) - ddy_periodic(u, dy)
