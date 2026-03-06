"""Model helpers for 1D pipe flow."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

_PROFILES = {"constant", "converging", "diverging", "sinusoidal", "custom"}
_FRICTION_MODELS = {"laminar", "blasius", "swamee_jain", "auto"}


def build_pipe_grid(length: float, nx: int) -> tuple[np.ndarray, float]:
    """Build 1D pipe grid."""
    if length <= 0:
        raise ValueError("length must be positive.")
    if nx < 16:
        raise ValueError("nx must be >= 16.")
    x = np.linspace(0.0, length, nx)
    dx = float(x[1] - x[0]) if nx > 1 else length
    return x, dx


def diameter_profile(
    x: np.ndarray,
    *,
    profile: str,
    d_in: float,
    d_out: float,
    d0: float,
    amplitude: float,
    n_waves: float,
    custom_fn: Callable[[float], float] | None = None,
) -> np.ndarray:
    """Build diameter profile along x."""
    p = profile.strip().lower()
    if p not in _PROFILES:
        raise ValueError(f"Unknown profile '{profile}'.")
    if d_in <= 0 or d_out <= 0 or d0 <= 0:
        raise ValueError("Pipe diameters must be positive.")
    length = float(x[-1] - x[0]) if len(x) > 1 else 1.0
    xn = (x - x[0]) / max(length, 1e-12)

    if p == "constant":
        d = np.full_like(x, fill_value=d0, dtype=float)
    elif p == "converging":
        d = d_in + (d_out - d_in) * xn
    elif p == "diverging":
        d = d_out + (d_in - d_out) * (1.0 - xn)
    elif p == "sinusoidal":
        d = d0 * (1.0 + amplitude * np.sin(2.0 * np.pi * n_waves * xn))
    else:
        if custom_fn is None:
            raise ValueError("Custom profile selected but no custom expression provided.")
        d = np.array([custom_fn(float(xi)) for xi in x], dtype=float)

    if np.any(d <= 0):
        raise ValueError("Diameter profile must remain positive for all x.")
    return d


def area_from_diameter(diameter: np.ndarray) -> np.ndarray:
    """Pipe cross-sectional area."""
    return 0.25 * np.pi * diameter**2


def reynolds_number(
    rho: float,
    mu: float,
    velocity: np.ndarray,
    diameter: np.ndarray,
) -> np.ndarray:
    """Compute Reynolds number profile."""
    if rho <= 0 or mu <= 0:
        raise ValueError("rho and mu must be positive.")
    return rho * np.abs(velocity) * diameter / mu


def friction_factor(
    re: np.ndarray,
    *,
    roughness: float,
    diameter: np.ndarray,
    model: str,
) -> np.ndarray:
    """Compute Darcy friction factor."""
    m = model.strip().lower()
    if m not in _FRICTION_MODELS:
        raise ValueError(f"Unknown friction model '{model}'.")
    if roughness < 0:
        raise ValueError("roughness must be non-negative.")

    re_safe = np.maximum(re, 1e-8)
    eps_rel = roughness / np.maximum(diameter, 1e-8)
    lam = 64.0 / np.maximum(re_safe, 1.0)
    blasius = 0.3164 / np.maximum(re_safe, 1.0) ** 0.25
    swj = 0.25 / (
        np.log10(np.maximum(eps_rel / 3.7 + 5.74 / np.maximum(re_safe, 1.0) ** 0.9, 1e-10)) ** 2
    )

    if m == "laminar":
        return lam
    if m == "blasius":
        return blasius
    if m == "swamee_jain":
        return swj
    return np.where(re_safe < 2300.0, lam, swj)
