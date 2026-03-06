"""Physical model helpers for the 2D nonlinear membrane."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

_SHAPES = {"gaussian", "mode", "random", "custom"}


def laplacian_2d(u: np.ndarray, *, boundary: str) -> np.ndarray:
    """Compute the discrete 2D Laplacian."""
    if boundary == "periodic":
        return (
            np.roll(u, -1, axis=0)
            + np.roll(u, 1, axis=0)
            + np.roll(u, -1, axis=1)
            + np.roll(u, 1, axis=1)
            - 4.0 * u
        )

    # Fixed boundary with u=0 outside domain.
    p = np.pad(u, pad_width=1, mode="constant", constant_values=0.0)
    return (
        p[2:, 1:-1]
        + p[:-2, 1:-1]
        + p[1:-1, 2:]
        + p[1:-1, :-2]
        - 4.0 * p[1:-1, 1:-1]
    )


def apply_fixed_boundary(u: np.ndarray, v: np.ndarray | None = None) -> None:
    """Clamp edges to zero in-place for fixed boundaries."""
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    if v is not None:
        v[0, :] = 0.0
        v[-1, :] = 0.0
        v[:, 0] = 0.0
        v[:, -1] = 0.0


def acceleration_field(
    u: np.ndarray,
    *,
    mass: float,
    k_linear: float,
    boundary: str,
    alpha: float = 0.0,
    beta: float = 0.0,
    high_order_coeff: float = 0.0,
    high_order_power: int = 5,
) -> np.ndarray:
    """Compute acceleration field from linear and nonlinear membrane terms."""
    lap = laplacian_2d(u, boundary=boundary)
    force = k_linear * lap
    if alpha != 0.0:
        force += alpha * lap**2
    if beta != 0.0:
        force += beta * lap**3
    if high_order_coeff != 0.0:
        force += high_order_coeff * np.sign(lap) * np.abs(lap) ** high_order_power

    a = force / mass
    if boundary == "fixed":
        a = a.copy()
        a[0, :] = 0.0
        a[-1, :] = 0.0
        a[:, 0] = 0.0
        a[:, -1] = 0.0
    return a


def _build_mesh(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Build normalized x/y meshes in [0, 1]."""
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y


def build_initial_displacement(
    *,
    nx: int,
    ny: int,
    shape: str,
    amplitude: float,
    sigma: float,
    mode_x: int = 1,
    mode_y: int = 1,
    center_x: float = 0.5,
    center_y: float = 0.5,
    custom_fn: Callable[[float, float], float] | None = None,
    random_seed: int = 0,
    boundary: str = "fixed",
) -> np.ndarray:
    """Create an initial displacement field."""
    if shape not in _SHAPES:
        raise ValueError(f"Unknown shape '{shape}'.")
    if nx < 2 or ny < 2:
        raise ValueError("Grid must be at least 2x2.")
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    X, Y = _build_mesh(nx, ny)

    if shape == "gaussian":
        u = amplitude * np.exp(
            -(((X - center_x) ** 2 + (Y - center_y) ** 2) / (2.0 * sigma**2))
        )
    elif shape == "mode":
        if boundary == "periodic":
            u = amplitude * np.cos(2.0 * np.pi * mode_x * X) * np.cos(
                2.0 * np.pi * mode_y * Y
            )
        else:
            j = np.arange(1, ny + 1)[:, np.newaxis]
            i = np.arange(1, nx + 1)[np.newaxis, :]
            u = amplitude * np.sin(mode_x * np.pi * i / (nx + 1)) * np.sin(
                mode_y * np.pi * j / (ny + 1)
            )
    elif shape == "random":
        rng = np.random.default_rng(random_seed)
        u = amplitude * rng.standard_normal((ny, nx))
    else:
        if custom_fn is None:
            raise ValueError("Custom shape selected but no expression provided.")
        u = np.array(
            [[custom_fn(float(X[j, i]), float(Y[j, i])) for i in range(nx)] for j in range(ny)],
            dtype=float,
        )

    if boundary == "fixed":
        u = u.copy()
        u[0, :] = 0.0
        u[-1, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0
    return u


def compute_energy_terms(
    u: np.ndarray,
    v: np.ndarray,
    *,
    mass: float,
    k_linear: float,
    boundary: str,
    alpha: float = 0.0,
    beta: float = 0.0,
    high_order_coeff: float = 0.0,
    high_order_power: int = 5,
) -> tuple[float, float, float]:
    """Compute kinetic, potential, total energies."""
    kinetic = 0.5 * mass * float(np.sum(v**2))

    if boundary == "periodic":
        dx = np.roll(u, -1, axis=1) - u
        dy = np.roll(u, -1, axis=0) - u
    else:
        p = np.pad(u, pad_width=1, mode="constant", constant_values=0.0)
        dx = p[:, 1:] - p[:, :-1]
        dy = p[1:, :] - p[:-1, :]
    potential = 0.5 * k_linear * float(np.sum(dx**2) + np.sum(dy**2))

    if alpha != 0.0 or beta != 0.0 or high_order_coeff != 0.0:
        lap = laplacian_2d(u, boundary=boundary)
        if alpha != 0.0:
            potential += (alpha / 3.0) * float(np.sum(lap**3))
        if beta != 0.0:
            potential += (beta / 4.0) * float(np.sum(lap**4))
        if high_order_coeff != 0.0:
            potential += (high_order_coeff / (high_order_power + 1.0)) * float(
                np.sum(np.sign(lap) * np.abs(lap) ** (high_order_power + 1))
            )

    total = kinetic + potential
    return kinetic, potential, total


def compute_fft_power_2d(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute shifted 2D power spectrum of a field."""
    ny, nx = u.shape
    spectrum = np.fft.fftshift(np.fft.fft2(u))
    power = np.abs(spectrum) ** 2
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    return kx, ky, power

