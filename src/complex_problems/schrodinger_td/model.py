"""Model helpers for TDSE (1D/2D) with configurable potentials."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

_POTENTIALS = {"free", "harmonic", "square_well", "barrier", "double_well", "lattice", "custom"}
_PACKETS = {"gaussian", "superposition", "custom"}


def normalize_wavefunction_1d(psi: np.ndarray, dx: float) -> np.ndarray:
    """Normalize 1D wavefunction."""
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm <= 0:
        return psi
    return psi / norm


def normalize_wavefunction_2d(psi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Normalize 2D wavefunction."""
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx * dy)
    if norm <= 0:
        return psi
    return psi / norm


def build_absorbing_mask_1d(nx: int, *, ratio: float = 0.1, strength: float = 1.0) -> np.ndarray:
    """Build a smooth 1D absorbing boundary mask."""
    ratio = max(0.0, min(0.49, ratio))
    n_edge = int(ratio * nx)
    mask = np.ones(nx, dtype=float)
    if n_edge <= 0:
        return mask
    idx = np.arange(nx)
    dist_left = idx
    dist_right = nx - 1 - idx
    dist = np.minimum(dist_left, dist_right).astype(float)
    s = np.clip(dist / max(1, n_edge), 0.0, 1.0)
    taper = np.sin(0.5 * np.pi * s) ** strength
    return taper


def build_absorbing_mask_2d(
    nx: int, ny: int, *, ratio: float = 0.1, strength: float = 1.0
) -> np.ndarray:
    """Build 2D absorbing mask as outer product of 1D masks."""
    mx = build_absorbing_mask_1d(nx, ratio=ratio, strength=strength)
    my = build_absorbing_mask_1d(ny, ratio=ratio, strength=strength)
    return my[:, np.newaxis] * mx[np.newaxis, :]


def potential_1d(
    x: np.ndarray,
    *,
    potential_type: str,
    omega: float = 1.0,
    v0: float = 5.0,
    width: float = 2.0,
    barrier_sigma: float = 0.3,
    lattice_k: float = 2.0,
    a_dw: float = 1.0,
    b_dw: float = 1.0,
    custom_fn: Callable[[float], float] | None = None,
) -> np.ndarray:
    """Build 1D potential profile."""
    p = potential_type.strip().lower()
    if p not in _POTENTIALS:
        raise ValueError(f"Unknown potential type '{potential_type}'.")
    if p == "free":
        return np.zeros_like(x)
    if p == "harmonic":
        return 0.5 * omega**2 * x**2
    if p == "square_well":
        out = np.full_like(x, fill_value=v0)
        out[np.abs(x) <= 0.5 * width] = 0.0
        return out
    if p == "barrier":
        return v0 * np.exp(-(x**2) / (2.0 * barrier_sigma**2))
    if p == "double_well":
        return a_dw * (x**2 - b_dw**2) ** 2
    if p == "lattice":
        return v0 * np.cos(lattice_k * x) ** 2
    if custom_fn is None:
        raise ValueError("Custom potential selected but no custom expression provided.")
    return np.array([custom_fn(float(xi)) for xi in x], dtype=float)


def potential_2d(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    potential_type: str,
    omega: float = 1.0,
    v0: float = 5.0,
    width: float = 2.0,
    barrier_sigma: float = 0.3,
    lattice_k: float = 2.0,
    a_dw: float = 1.0,
    b_dw: float = 1.0,
    custom_fn: Callable[[float, float], float] | None = None,
) -> np.ndarray:
    """Build 2D potential surface."""
    p = potential_type.strip().lower()
    if p not in _POTENTIALS:
        raise ValueError(f"Unknown potential type '{potential_type}'.")
    if p == "free":
        return np.zeros_like(X)
    if p == "harmonic":
        return 0.5 * omega**2 * (X**2 + Y**2)
    if p == "square_well":
        inside = (np.abs(X) <= 0.5 * width) & (np.abs(Y) <= 0.5 * width)
        out = np.full_like(X, fill_value=v0)
        out[inside] = 0.0
        return out
    if p == "barrier":
        return v0 * np.exp(-((X**2 + Y**2) / (2.0 * barrier_sigma**2)))
    if p == "double_well":
        return a_dw * ((X**2 - b_dw**2) ** 2 + (Y**2 - b_dw**2) ** 2)
    if p == "lattice":
        return v0 * (np.cos(lattice_k * X) ** 2 + np.cos(lattice_k * Y) ** 2)
    if custom_fn is None:
        raise ValueError("Custom potential selected but no custom expression provided.")
    out = np.zeros_like(X, dtype=float)
    for j in range(X.shape[0]):
        for i in range(X.shape[1]):
            out[j, i] = custom_fn(float(X[j, i]), float(Y[j, i]))
    return out


def initial_packet_1d(
    x: np.ndarray,
    *,
    packet_type: str,
    sigma: float,
    x0: float,
    k0x: float,
    separation: float = 2.0,
    custom_fn: Callable[[float], float] | None = None,
) -> np.ndarray:
    """Build initial 1D packet."""
    p = packet_type.strip().lower()
    if p not in _PACKETS:
        raise ValueError(f"Unknown packet type '{packet_type}'.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    def _gauss(center: float) -> np.ndarray:
        return np.exp(-((x - center) ** 2) / (2.0 * sigma**2))

    if p == "gaussian":
        amp = _gauss(x0)
    elif p == "superposition":
        amp = _gauss(x0 - separation * 0.5) + _gauss(x0 + separation * 0.5)
    else:
        if custom_fn is None:
            raise ValueError("Custom packet selected but no custom expression provided.")
        amp = np.array([custom_fn(float(xi)) for xi in x], dtype=float)
    psi = amp.astype(complex) * np.exp(1j * k0x * x)
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    return normalize_wavefunction_1d(psi, dx)


def initial_packet_2d(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    packet_type: str,
    sigma: float,
    x0: float,
    y0: float,
    k0x: float,
    k0y: float,
    separation: float = 2.0,
    custom_fn: Callable[[float, float], float] | None = None,
) -> np.ndarray:
    """Build initial 2D packet."""
    p = packet_type.strip().lower()
    if p not in _PACKETS:
        raise ValueError(f"Unknown packet type '{packet_type}'.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    def _gauss(cx: float, cy: float) -> np.ndarray:
        return np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma**2)))

    if p == "gaussian":
        amp = _gauss(x0, y0)
    elif p == "superposition":
        amp = _gauss(x0 - separation * 0.5, y0) + _gauss(x0 + separation * 0.5, y0)
    else:
        if custom_fn is None:
            raise ValueError("Custom packet selected but no custom expression provided.")
        amp = np.zeros_like(X, dtype=float)
        for j in range(X.shape[0]):
            for i in range(X.shape[1]):
                amp[j, i] = custom_fn(float(X[j, i]), float(Y[j, i]))

    psi = amp.astype(complex) * np.exp(1j * (k0x * X + k0y * Y))
    dx = float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 1.0
    dy = float(Y[1, 0] - Y[0, 0]) if Y.shape[0] > 1 else 1.0
    return normalize_wavefunction_2d(psi, dx, dy)

