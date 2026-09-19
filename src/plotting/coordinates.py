"""Pure coordinate and field helpers used by PDE visualizations.

The numerical solvers always operate in Cartesian coordinates.  These helpers
only change how already-computed data is displayed.  At coordinate singularities
they use a fixed, documented basis so callers never receive divide-by-zero
warnings: polar/cylindrical origins use ``(e_r, e_phi) = ((1, 0), (0, 1))``;
the spherical origin uses ``(e_r, e_theta, e_phi) = ((0, 0, 1), (1, 0, 0),
(0, 1, 0))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PolarResample:
    """A scalar Cartesian field resampled onto a polar display grid."""

    radius: NDArray[np.float64]
    angle: NDArray[np.float64]
    values: NDArray[np.float64]


@dataclass(frozen=True)
class VectorFieldData:
    """Cartesian vector field arrays ready for a 2D plotting primitive."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    u: NDArray[np.float64]
    v: NDArray[np.float64]
    magnitude: NDArray[np.float64]


@dataclass(frozen=True)
class ScalarSlice3D:
    """An orthogonal scalar slice with its plotted and fixed coordinates."""

    plane: Literal["XY", "XZ", "YZ"]
    axis_1: NDArray[np.float64]
    axis_2: NDArray[np.float64]
    values: NDArray[np.float64]
    fixed_index: int
    fixed_coordinate: float


@dataclass(frozen=True)
class ScalarAxisSweep:
    """Display-independent frames obtained by sweeping one spatial axis."""

    sweep_axis: Literal["x", "y", "z"]
    sweep_coordinates: NDArray[np.float64]
    axis_1: NDArray[np.float64]
    axis_2: NDArray[np.float64] | None
    frames: NDArray[np.float64]


def prepare_scalar_axis_sweep_2d(
    x: ArrayLike,
    y: ArrayLike,
    values: ArrayLike,
    sweep_axis: Literal["x", "y"],
) -> ScalarAxisSweep:
    """Prepare 1D scalar profiles by sweeping one coordinate of a 2D field."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    field = np.asarray(values, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if field.shape != (len(y_arr), len(x_arr)):
        raise ValueError("values must have shape (len(y), len(x))")
    if sweep_axis == "x":
        return ScalarAxisSweep("x", x_arr, y_arr, None, field.T)
    if sweep_axis == "y":
        return ScalarAxisSweep("y", y_arr, x_arr, None, field)
    raise ValueError("sweep_axis must be 'x' or 'y'")


def prepare_scalar_axis_sweep_3d(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    values: ArrayLike,
    sweep_axis: Literal["x", "y", "z"],
) -> ScalarAxisSweep:
    """Prepare orthogonal 2D scalar frames from ``(z, y, x)`` field data."""
    x_arr, y_arr, z_arr = (np.asarray(axis, dtype=float) for axis in (x, y, z))
    field = np.asarray(values, dtype=float)
    if any(axis.ndim != 1 for axis in (x_arr, y_arr, z_arr)):
        raise ValueError("x, y, and z must be one-dimensional")
    if field.shape != (len(z_arr), len(y_arr), len(x_arr)):
        raise ValueError("values must have shape (len(z), len(y), len(x))")
    if sweep_axis == "x":
        return ScalarAxisSweep("x", x_arr, y_arr, z_arr, np.moveaxis(field, 2, 0))
    if sweep_axis == "y":
        return ScalarAxisSweep("y", y_arr, x_arr, z_arr, np.moveaxis(field, 1, 0))
    if sweep_axis == "z":
        return ScalarAxisSweep("z", z_arr, x_arr, y_arr, field)
    raise ValueError("sweep_axis must be 'x', 'y', or 'z'")


def _arrays(*values: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    """Convert and broadcast numeric coordinate inputs without mutating them."""
    return tuple(np.broadcast_arrays(*(np.asarray(value, dtype=float) for value in values)))


def cartesian_to_polar(
    x: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return radius and azimuth (radians) for Cartesian planar coordinates."""
    x_arr, y_arr = _arrays(x, y)
    return np.hypot(x_arr, y_arr), np.arctan2(y_arr, x_arr)


def polar_to_cartesian(
    radius: ArrayLike, angle: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return Cartesian planar coordinates from radius and azimuth."""
    radius_arr, angle_arr = _arrays(radius, angle)
    return radius_arr * np.cos(angle_arr), radius_arr * np.sin(angle_arr)


def cartesian_to_cylindrical(
    x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return cylindrical ``(radius, azimuth, z)`` coordinates."""
    x_arr, y_arr, z_arr = _arrays(x, y, z)
    return np.hypot(x_arr, y_arr), np.arctan2(y_arr, x_arr), z_arr


def cylindrical_to_cartesian(
    radius: ArrayLike, angle: ArrayLike, z: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return Cartesian coordinates from cylindrical ``(radius, azimuth, z)``."""
    radius_arr, angle_arr, z_arr = _arrays(radius, angle, z)
    return radius_arr * np.cos(angle_arr), radius_arr * np.sin(angle_arr), z_arr


def cartesian_to_spherical(
    x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return spherical ``(radius, polar_angle, azimuth)`` coordinates.

    The polar angle is measured from positive z.  At the origin both angles
    are deterministically set to zero.
    """
    x_arr, y_arr, z_arr = _arrays(x, y, z)
    radius = np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)
    cos_polar = np.ones_like(radius)
    np.divide(z_arr, radius, out=cos_polar, where=radius != 0.0)
    polar = np.arccos(np.clip(cos_polar, -1.0, 1.0))
    return radius, polar, np.arctan2(y_arr, x_arr)


def spherical_to_cartesian(
    radius: ArrayLike, polar_angle: ArrayLike, azimuth: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return Cartesian coordinates from spherical values."""
    radius_arr, polar_arr, azimuth_arr = _arrays(radius, polar_angle, azimuth)
    sin_polar = np.sin(polar_arr)
    return (
        radius_arr * sin_polar * np.cos(azimuth_arr),
        radius_arr * sin_polar * np.sin(azimuth_arr),
        radius_arr * np.cos(polar_arr),
    )


def cartesian_vector_to_polar(
    vx: ArrayLike, vy: ArrayLike, x: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Express planar Cartesian vector components in the polar basis."""
    vx_arr, vy_arr, x_arr, y_arr = _arrays(vx, vy, x, y)
    radius = np.hypot(x_arr, y_arr)
    cos_angle = np.ones_like(radius)
    sin_angle = np.zeros_like(radius)
    np.divide(x_arr, radius, out=cos_angle, where=radius != 0.0)
    np.divide(y_arr, radius, out=sin_angle, where=radius != 0.0)
    return vx_arr * cos_angle + vy_arr * sin_angle, -vx_arr * sin_angle + vy_arr * cos_angle


def polar_vector_to_cartesian(
    radial: ArrayLike, tangential: ArrayLike, x: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Express polar vector components in the Cartesian basis."""
    radial_arr, tangential_arr, x_arr, y_arr = _arrays(radial, tangential, x, y)
    radius = np.hypot(x_arr, y_arr)
    cos_angle = np.ones_like(radius)
    sin_angle = np.zeros_like(radius)
    np.divide(x_arr, radius, out=cos_angle, where=radius != 0.0)
    np.divide(y_arr, radius, out=sin_angle, where=radius != 0.0)
    return (
        radial_arr * cos_angle - tangential_arr * sin_angle,
        radial_arr * sin_angle + tangential_arr * cos_angle,
    )


def cartesian_vector_to_cylindrical(
    vx: ArrayLike, vy: ArrayLike, vz: ArrayLike, x: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Express a Cartesian vector in the cylindrical basis.

    Every returned component has the common broadcast shape of the vector and
    coordinate inputs.
    """
    vx_arr, vy_arr, axial_arr, x_arr, y_arr = _arrays(vx, vy, vz, x, y)
    radial, tangential = cartesian_vector_to_polar(vx_arr, vy_arr, x_arr, y_arr)
    return radial, tangential, axial_arr


def cylindrical_vector_to_cartesian(
    radial: ArrayLike, tangential: ArrayLike, axial: ArrayLike, x: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Express cylindrical vector components in the Cartesian basis.

    Every returned component has the common broadcast shape of the vector and
    coordinate inputs.
    """
    radial_arr, tangential_arr, axial_arr, x_arr, y_arr = _arrays(radial, tangential, axial, x, y)
    vx, vy = polar_vector_to_cartesian(radial_arr, tangential_arr, x_arr, y_arr)
    return vx, vy, axial_arr


def cartesian_vector_to_spherical(
    vx: ArrayLike, vy: ArrayLike, vz: ArrayLike, x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Express a Cartesian vector in the spherical basis."""
    vx_arr, vy_arr, vz_arr, x_arr, y_arr, z_arr = _arrays(vx, vy, vz, x, y, z)
    radius, polar, azimuth = cartesian_to_spherical(x_arr, y_arr, z_arr)
    sin_polar, cos_polar = np.sin(polar), np.cos(polar)
    cos_azimuth, sin_azimuth = np.cos(azimuth), np.sin(azimuth)
    return (
        vx_arr * sin_polar * cos_azimuth + vy_arr * sin_polar * sin_azimuth + vz_arr * cos_polar,
        vx_arr * cos_polar * cos_azimuth + vy_arr * cos_polar * sin_azimuth - vz_arr * sin_polar,
        -vx_arr * sin_azimuth + vy_arr * cos_azimuth,
    )


def spherical_vector_to_cartesian(
    radial: ArrayLike,
    polar: ArrayLike,
    azimuthal: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Express spherical vector components in the Cartesian basis."""
    radial_arr, polar_arr, azimuthal_arr, x_arr, y_arr, z_arr = _arrays(
        radial, polar, azimuthal, x, y, z
    )
    _radius, polar_angle, azimuth = cartesian_to_spherical(x_arr, y_arr, z_arr)
    sin_polar, cos_polar = np.sin(polar_angle), np.cos(polar_angle)
    cos_azimuth, sin_azimuth = np.cos(azimuth), np.sin(azimuth)
    return (
        radial_arr * sin_polar * cos_azimuth
        + polar_arr * cos_polar * cos_azimuth
        - azimuthal_arr * sin_azimuth,
        radial_arr * sin_polar * sin_azimuth
        + polar_arr * cos_polar * sin_azimuth
        + azimuthal_arr * cos_azimuth,
        radial_arr * cos_polar - polar_arr * sin_polar,
    )


def resample_scalar_to_polar(
    x: ArrayLike,
    y: ArrayLike,
    values: ArrayLike,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    radial_points: int = 160,
    angular_points: int = 180,
) -> PolarResample:
    """Resample a Cartesian scalar field onto a polar grid with NaN masking.

    Points outside the source rectangle and interpolation neighborhoods touching
    NaN/masked source samples are emitted as NaN.
    """
    from scipy.interpolate import RegularGridInterpolator

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    field = np.ma.filled(np.ma.asarray(values, dtype=float), np.nan)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or field.shape != (len(y_arr), len(x_arr)):
        raise ValueError("values must have shape (len(y), len(x)) on one-dimensional grids")
    if radial_points < 2 or angular_points < 3:
        raise ValueError("polar resampling requires at least 2 radial and 3 angular points")
    ox, oy = origin
    corners_x, corners_y = np.meshgrid([x_arr[0], x_arr[-1]], [y_arr[0], y_arr[-1]])
    radius_max = float(np.max(np.hypot(corners_x - ox, corners_y - oy)))
    radius = np.linspace(0.0, radius_max, radial_points)
    angle = np.linspace(-np.pi, np.pi, angular_points, endpoint=False)
    radius_grid, angle_grid = np.meshgrid(radius, angle)
    sample_x = ox + radius_grid * np.cos(angle_grid)
    sample_y = oy + radius_grid * np.sin(angle_grid)
    interpolator = RegularGridInterpolator(
        (y_arr, x_arr), field, bounds_error=False, fill_value=np.nan
    )
    sampled = interpolator(np.column_stack((sample_y.ravel(), sample_x.ravel()))).reshape(
        angle_grid.shape
    )
    return PolarResample(radius=radius, angle=angle, values=sampled)


def vector_field_data(x: ArrayLike, y: ArrayLike, components: ArrayLike) -> VectorFieldData:
    """Validate an exactly two-component planar field and compute its magnitude."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    field = np.asarray(components, dtype=float)
    expected_shape = (len(y_arr), len(x_arr))
    if field.ndim != 3 or field.shape[0] != 2 or field.shape[1:] != expected_shape:
        raise ValueError("components must have shape (2, len(y), len(x))")
    u, v = field[0], field[1]
    return VectorFieldData(x=x_arr, y=y_arr, u=u, v=v, magnitude=np.hypot(u, v))


def extract_scalar_3d_slice(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    values: ArrayLike,
    plane: Literal["XY", "XZ", "YZ"],
    index: int,
) -> ScalarSlice3D:
    """Extract an index-clamped orthogonal slice from ``(nz, ny, nx)`` data."""
    x_arr, y_arr, z_arr = (np.asarray(axis, dtype=float) for axis in (x, y, z))
    field = np.asarray(values, dtype=float)
    if field.shape != (len(z_arr), len(y_arr), len(x_arr)):
        raise ValueError("values must have shape (len(z), len(y), len(x))")
    if plane == "XY":
        fixed = int(np.clip(index, 0, len(z_arr) - 1))
        return ScalarSlice3D(plane, x_arr, y_arr, field[fixed], fixed, float(z_arr[fixed]))
    if plane == "XZ":
        fixed = int(np.clip(index, 0, len(y_arr) - 1))
        return ScalarSlice3D(plane, x_arr, z_arr, field[:, fixed, :], fixed, float(y_arr[fixed]))
    if plane == "YZ":
        fixed = int(np.clip(index, 0, len(x_arr) - 1))
        return ScalarSlice3D(plane, y_arr, z_arr, field[:, :, fixed], fixed, float(x_arr[fixed]))
    raise ValueError("plane must be 'XY', 'XZ', or 'YZ'")
