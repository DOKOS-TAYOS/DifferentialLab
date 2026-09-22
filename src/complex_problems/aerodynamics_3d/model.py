"""Geometry and finite-difference helpers for 3D aerodynamics."""

from __future__ import annotations

import numpy as np

_OBSTACLE_SHAPES = {"sphere", "ellipsoid", "box", "naca0012_wing"}
_BOUNDARY_ERROR = (
    "Aerodynamics 3D v1 requires the obstacle to remain fully inside the periodic box; "
    "move the obstacle inward or reduce its dimensions."
)


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


def obstacle_half_extents(
    *,
    shape: str,
    diameter: float = 0.4,
    size_x: float = 0.7,
    size_y: float = 0.4,
    size_z: float = 0.4,
    chord: float = 0.8,
    span: float = 0.8,
    thickness_ratio: float = 0.12,
    attack_deg: float = 0.0,
) -> tuple[float, float, float]:
    """Return world-aligned half extents for an obstacle geometry."""
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

    if obstacle_shape == "sphere":
        radius = diameter / 2.0
        return radius, radius, radius

    if obstacle_shape == "ellipsoid":
        local_x, local_y, local_z = size_x / 2.0, size_y / 2.0, size_z / 2.0
    elif obstacle_shape == "box":
        local_x, local_y, local_z = size_x / 2.0, size_y / 2.0, size_z / 2.0
    else:
        local_x = chord / 2.0
        sample_x = np.linspace(0.0, chord, 2049)
        local_y = float(np.max(_naca_thickness(sample_x, chord, thickness_ratio)))
        local_z = span / 2.0

    angle = np.deg2rad(attack_deg)
    cosine, sine = abs(float(np.cos(angle))), abs(float(np.sin(angle)))
    return cosine * local_x + sine * local_y, sine * local_x + cosine * local_y, local_z


def validate_obstacle_fits_domain(
    *,
    shape: str,
    center_x: float,
    center_y: float,
    center_z: float,
    lx: float,
    ly: float,
    lz: float,
    diameter: float = 0.4,
    size_x: float = 0.7,
    size_y: float = 0.4,
    size_z: float = 0.4,
    chord: float = 0.8,
    span: float = 0.8,
    thickness_ratio: float = 0.12,
    attack_deg: float = 0.0,
) -> None:
    """Reject obstacles whose complete geometry crosses a Cartesian boundary."""
    if min(lx, ly, lz) <= 0:
        raise ValueError("Domain lengths must be positive.")
    half_x, half_y, half_z = obstacle_half_extents(
        shape=shape,
        diameter=diameter,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        chord=chord,
        span=span,
        thickness_ratio=thickness_ratio,
        attack_deg=attack_deg,
    )
    lower = (center_x - half_x, center_y - half_y, center_z - half_z)
    upper = (center_x + half_x, center_y + half_y, center_z + half_z)
    bounds = ((lower[0], upper[0], lx), (lower[1], upper[1], ly), (lower[2], upper[2], lz))
    if any(lo < 0.0 or hi > limit for lo, hi, limit in bounds):
        raise ValueError(_BOUNDARY_ERROR)


def _rotate_surface_xy(
    local_x: np.ndarray,
    local_y: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    attack_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate local surface coordinates back into world x/y coordinates."""
    angle = np.deg2rad(attack_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    return (
        center_x + cosine * local_x - sine * local_y,
        center_y + sine * local_x + cosine * local_y,
    )


def obstacle_surface_mesh(
    *,
    shape: str,
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
    resolution: int = 28,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return continuous surface patches using the same geometry as the mask."""
    if resolution < 4:
        raise ValueError("resolution must be at least 4.")
    obstacle_shape = shape.strip().lower()
    if obstacle_shape not in _OBSTACLE_SHAPES:
        raise ValueError(f"Unknown obstacle shape '{shape}'.")

    patches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if obstacle_shape in {"sphere", "ellipsoid"}:
        angle = np.linspace(0.0, 2.0 * np.pi, resolution)
        polar = np.linspace(0.0, np.pi, max(12, resolution // 2))
        aa, pp = np.meshgrid(angle, polar)
        if obstacle_shape == "sphere":
            radii = (diameter / 2.0, diameter / 2.0, diameter / 2.0)
        else:
            radii = (size_x / 2.0, size_y / 2.0, size_z / 2.0)
        local_x = radii[0] * np.sin(pp) * np.cos(aa)
        local_y = radii[1] * np.sin(pp) * np.sin(aa)
        local_z = center_z + radii[2] * np.cos(pp)
        world_x, world_y = _rotate_surface_xy(
            local_x,
            local_y,
            center_x=center_x,
            center_y=center_y,
            attack_deg=attack_deg,
        )
        return [(world_x, world_y, local_z)]

    if obstacle_shape == "box":
        half_x, half_y, half_z = size_x / 2.0, size_y / 2.0, size_z / 2.0
        values_x = np.linspace(-half_x, half_x, resolution)
        values_y = np.linspace(-half_y, half_y, resolution)
        values_z = np.linspace(-half_z, half_z, resolution)
        face_xy_x, face_xy_y = np.meshgrid(values_x, values_y)
        face_xz_x, face_xz_z = np.meshgrid(values_x, values_z)
        face_yz_y, face_yz_z = np.meshgrid(values_y, values_z)
        for local_x, local_y, local_z in (
            (face_xy_x, face_xy_y, np.full_like(face_xy_x, -half_z)),
            (face_xy_x, face_xy_y, np.full_like(face_xy_x, half_z)),
            (face_xz_x, np.full_like(face_xz_x, -half_y), face_xz_z),
            (face_xz_x, np.full_like(face_xz_x, half_y), face_xz_z),
            (np.full_like(face_yz_y, -half_x), face_yz_y, face_yz_z),
            (np.full_like(face_yz_y, half_x), face_yz_y, face_yz_z),
        ):
            world_x, world_y = _rotate_surface_xy(
                local_x,
                local_y,
                center_x=center_x,
                center_y=center_y,
                attack_deg=attack_deg,
            )
            patches.append((world_x, world_y, center_z + local_z))
        return patches

    local_x_values = np.linspace(0.0, chord, resolution)
    local_z_values = np.linspace(-span / 2.0, span / 2.0, max(12, resolution // 2))
    local_x, local_z = np.meshgrid(local_x_values, local_z_values)
    local_thickness = _naca_thickness(local_x, chord, thickness_ratio)
    for sign in (-1.0, 1.0):
        world_x, world_y = _rotate_surface_xy(
            local_x - chord / 2.0,
            sign * local_thickness,
            center_x=center_x,
            center_y=center_y,
            attack_deg=attack_deg,
        )
        patches.append((world_x, world_y, center_z + local_z))
    return patches


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
