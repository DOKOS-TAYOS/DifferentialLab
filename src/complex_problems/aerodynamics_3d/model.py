"""Geometry and finite-difference helpers for 3D aerodynamics."""

from __future__ import annotations

import numpy as np

_OBSTACLE_SHAPES = {"sphere", "ellipsoid", "box", "naca0012_wing"}


def build_periodic_domain(
    *, nx: int, ny: int, nz: int, lx: float, ly: float, lz: float
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]:
    """Build periodic coordinates and meshes stored in ``(z, y, x)`` order."""
    if min(nx, ny, nz) < 4:
        raise ValueError("nx, ny, and nz must be >= 4.")
    if min(lx, ly, lz) <= 0:
        raise ValueError("lx, ly, and lz must be positive.")
    x = np.linspace(0.0, lx, nx, endpoint=False)
    y = np.linspace(0.0, ly, ny, endpoint=False)
    z = np.linspace(0.0, lz, nz, endpoint=False)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return x, y, z, xx, yy, zz, lx / nx, ly / ny, lz / nz


def rotate_coordinates(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate x/y coordinates about the obstacle center around the z axis."""
    angle = np.deg2rad(angle_deg)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    xr = X - center_x
    yr = Y - center_y
    return cosine * xr + sine * yr, -sine * xr + cosine * yr


def _naca_thickness(x_local: np.ndarray, chord: float, thickness_ratio: float) -> np.ndarray:
    """Return the half-thickness of a symmetric NACA 00xx section."""
    xc = np.clip(x_local / chord, 0.0, 1.0)
    return (
        5.0
        * thickness_ratio
        * chord
        * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1015 * xc**4)
    )


def build_obstacle_mask(
    *,
    shape: str,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    center_x: float,
    center_y: float,
    center_z: float,
    diameter: float = 0.4,
    size_x: float = 0.7,
    size_y: float = 0.4,
    size_z: float = 0.4,
    chord: float = 0.8,
    span: float = 0.8,
    thickness_ratio: float = 0.12,
    attack_deg: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Build an immersed obstacle mask and its reference area and length.

    All dimensions are full dimensions.  The returned mask has the same
    ``(z, y, x)`` shape as the coordinate arrays.
    """
    obstacle_shape = shape.strip().lower()
    if obstacle_shape not in _OBSTACLE_SHAPES:
        raise ValueError(f"Unknown obstacle shape '{shape}'.")
    positive = {
        "diameter": diameter,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
        "chord": chord,
        "span": span,
        "thickness_ratio": thickness_ratio,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError("Obstacle dimensions and thickness ratio must be positive.")

    xp, yp = rotate_coordinates(X, Y, center_x=center_x, center_y=center_y, angle_deg=attack_deg)
    zp = Z - center_z
    if obstacle_shape == "sphere":
        radius = diameter / 2.0
        mask = xp * xp + yp * yp + zp * zp <= radius * radius
        reference_area = np.pi * radius * radius
        reference_length = diameter
    elif obstacle_shape == "ellipsoid":
        mask = (xp / (size_x / 2.0)) ** 2 + (yp / (size_y / 2.0)) ** 2 + (
            zp / (size_z / 2.0)
        ) ** 2 <= 1.0
        reference_area = np.pi * (size_y / 2.0) * (size_z / 2.0)
        reference_length = size_x
    elif obstacle_shape == "box":
        mask = (
            (np.abs(xp) <= size_x / 2.0)
            & (np.abs(yp) <= size_y / 2.0)
            & (np.abs(zp) <= size_z / 2.0)
        )
        reference_area = size_y * size_z
        reference_length = size_x
    else:
        x_local = xp + chord / 2.0
        mask = (
            (x_local >= 0.0)
            & (x_local <= chord)
            & (np.abs(yp) <= _naca_thickness(x_local, chord, thickness_ratio))
            & (np.abs(zp) <= span / 2.0)
        )
        reference_area = chord * span
        reference_length = chord

    return mask.astype(bool), float(reference_area), float(reference_length)


def ddx_periodic(field: np.ndarray, dx: float) -> np.ndarray:
    """Centered periodic derivative along the last (x) axis."""
    result = np.empty_like(field, dtype=float)
    result[..., 1:-1] = (field[..., 2:] - field[..., :-2]) / (2.0 * dx)
    result[..., 0] = (field[..., 1] - field[..., -1]) / (2.0 * dx)
    result[..., -1] = (field[..., 0] - field[..., -2]) / (2.0 * dx)
    return result


def ddy_periodic(field: np.ndarray, dy: float) -> np.ndarray:
    """Centered periodic derivative along the y axis."""
    result = np.empty_like(field, dtype=float)
    result[:, 1:-1, ...] = (field[:, 2:, ...] - field[:, :-2, ...]) / (2.0 * dy)
    result[:, 0, ...] = (field[:, 1, ...] - field[:, -1, ...]) / (2.0 * dy)
    result[:, -1, ...] = (field[:, 0, ...] - field[:, -2, ...]) / (2.0 * dy)
    return result


def ddz_periodic(field: np.ndarray, dz: float) -> np.ndarray:
    """Centered periodic derivative along the first (z) axis."""
    result = np.empty_like(field, dtype=float)
    result[1:-1, ...] = (field[2:, ...] - field[:-2, ...]) / (2.0 * dz)
    result[0, ...] = (field[1, ...] - field[-1, ...]) / (2.0 * dz)
    result[-1, ...] = (field[0, ...] - field[-2, ...]) / (2.0 * dz)
    return result


def laplacian_periodic(field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """Periodic centered finite-difference Laplacian."""
    return (
        (np.roll(field, -1, axis=-1) - 2.0 * field + np.roll(field, 1, axis=-1)) / dx**2
        + (np.roll(field, -1, axis=-2) - 2.0 * field + np.roll(field, 1, axis=-2)) / dy**2
        + (np.roll(field, -1, axis=-3) - 2.0 * field + np.roll(field, 1, axis=-3)) / dz**2
    )


def divergence_periodic(
    u: np.ndarray, v: np.ndarray, w: np.ndarray, dx: float, dy: float, dz: float
) -> np.ndarray:
    """Centered finite-difference divergence for diagnostics outside projection."""
    return ddx_periodic(u, dx) + ddy_periodic(v, dy) + ddz_periodic(w, dz)


def spectral_divergence_l2(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
    *,
    fluid_mask: np.ndarray | None = None,
) -> float:
    """Return the L2 norm of the Fourier divergence."""
    divergence_hat = 1j * (kx * np.fft.fftn(u) + ky * np.fft.fftn(v) + kz * np.fft.fftn(w))
    divergence = np.real(np.fft.ifftn(divergence_hat))
    if fluid_mask is not None and np.any(fluid_mask):
        divergence = divergence[fluid_mask]
    return float(np.sqrt(np.mean(divergence * divergence)))


def vorticity_periodic(
    u: np.ndarray, v: np.ndarray, w: np.ndarray, dx: float, dy: float, dz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the centered finite-difference vorticity vector."""
    return (
        ddy_periodic(w, dy) - ddz_periodic(v, dz),
        ddz_periodic(u, dz) - ddx_periodic(w, dx),
        ddx_periodic(v, dx) - ddy_periodic(u, dy),
    )
