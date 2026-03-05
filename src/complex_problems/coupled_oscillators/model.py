"""Physics model for coupled harmonic oscillators."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from utils import get_logger

logger = get_logger(__name__)

_UNIFORM_RTOL = 1e-12
_UNIFORM_ATOL = 1e-12
_EPS_SIGN = 1e-12


def _resolve_mass(masses_spec: Any, i: int, n: int) -> float:
    """Resolve mass for oscillator i."""
    if callable(masses_spec):
        return float(masses_spec(i))
    if isinstance(masses_spec, (list, tuple, np.ndarray)):
        if len(masses_spec) == 0:
            raise ValueError("Mass specification list cannot be empty.")
        return float(masses_spec[i] if i < len(masses_spec) else masses_spec[-1])
    return float(masses_spec)


def _resolve_k(k_spec: Any, i: int, n: int) -> float:
    """Resolve coupling constant at spring index i."""
    if callable(k_spec):
        return float(k_spec(i))
    if isinstance(k_spec, (list, tuple, np.ndarray)):
        if len(k_spec) == 0:
            raise ValueError("Coupling specification list cannot be empty.")
        idx = min(i, len(k_spec) - 1)
        return float(k_spec[idx])
    return float(k_spec)


def _resolve_mass_array(
    masses_spec: float | list[float] | Callable[[int], float],
    n: int,
) -> np.ndarray:
    masses = np.array([_resolve_mass(masses_spec, i, n) for i in range(n)], dtype=float)
    if np.any(masses <= 0):
        raise ValueError("All masses must be positive.")
    return masses


def _resolve_k_array(
    k_spec: float | list[float] | Callable[[int], float],
    n_springs: int,
    n: int,
) -> np.ndarray:
    if n_springs <= 0:
        return np.zeros(0, dtype=float)
    k_arr = np.array([_resolve_k(k_spec, i, n) for i in range(n_springs)], dtype=float)
    if np.any(k_arr < 0):
        raise ValueError("Coupling constants must be non-negative.")
    return k_arr


def _neighbor_values(x: np.ndarray, dist: int, boundary: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (left, right) neighbor values for distance dist."""
    if boundary == "periodic":
        return np.roll(x, dist), np.roll(x, -dist)

    left = np.zeros_like(x)
    right = np.zeros_like(x)
    if dist < x.size:
        left[dist:] = x[:-dist]
        right[:-dist] = x[dist:]
    return left, right


def _build_nearest_k_arrays(k_nearest: np.ndarray, boundary: str) -> tuple[np.ndarray, np.ndarray]:
    """Build nearest-neighbor spring arrays (left/right spring for each mass)."""
    n = len(k_nearest) + 1 if boundary == "fixed" else len(k_nearest)
    if boundary == "periodic":
        k_right = k_nearest.copy()
        k_left = np.roll(k_nearest, 1)
        return k_left, k_right

    if len(k_nearest) != n - 1:
        raise ValueError("Fixed boundary requires n-1 nearest-neighbor coupling values.")

    k_left = np.empty(n, dtype=float)
    k_right = np.empty(n, dtype=float)
    k_left[0] = k_nearest[0]
    k_left[1:] = k_nearest
    k_right[:-1] = k_nearest
    k_right[-1] = k_nearest[-1]
    return k_left, k_right


def build_ode_function(
    n_oscillators: int,
    masses: float | list[float] | Callable[[int], float],
    k_coupling: float | list[float] | Callable[[int], float],
    boundary: str = "fixed",
    coupling_types: list[str] | None = None,
    nonlinear_coeff: float = 0.0,
    nonlinear_fput_alpha: float = 0.0,
    nonlinear_quartic: float = 0.0,
    nonlinear_quintic: float = 0.0,
    k_2nn: float = 0.0,
    k_3nn: float = 0.0,
    k_4nn: float = 0.0,
    external_amplitude: float = 0.0,
    external_frequency: float = 1.0,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Build the RHS for coupled oscillators with optional nonlinear terms."""
    if boundary not in {"fixed", "periodic"}:
        raise ValueError("Boundary must be 'fixed' or 'periodic'.")
    if n_oscillators < 2:
        raise ValueError("At least 2 oscillators are required.")

    coupling_types = coupling_types or ["linear"]
    n = n_oscillators
    n_springs = n - 1 if boundary == "fixed" else n

    masses_arr = _resolve_mass_array(masses, n)
    k_nearest = _resolve_k_array(k_coupling, n_springs, n)
    k_left, k_right = _build_nearest_k_arrays(k_nearest, boundary)

    has_linear = "linear" in coupling_types
    has_beta = "nonlinear" in coupling_types and nonlinear_coeff != 0.0
    has_alpha = "nonlinear_fput_alpha" in coupling_types and nonlinear_fput_alpha != 0.0
    has_quartic = "nonlinear_quartic" in coupling_types and nonlinear_quartic != 0.0
    has_quintic = "nonlinear_quintic" in coupling_types and nonlinear_quintic != 0.0
    has_external = "external_force" in coupling_types and external_amplitude != 0.0

    long_range: list[tuple[int, float]] = [
        (2, float(k_2nn)),
        (3, float(k_3nn)),
        (4, float(k_4nn)),
    ]
    long_range = [(dist, kval) for dist, kval in long_range if kval != 0.0]

    def ode_func(t: float, y: np.ndarray) -> np.ndarray:
        x = y[:n]
        v = y[n:]

        left_1, right_1 = _neighbor_values(x, 1, boundary)
        delta_left = x - left_1
        delta_right = right_1 - x
        lap_1 = right_1 + left_1 - 2.0 * x

        force = np.zeros(n, dtype=float)

        if has_linear:
            force += k_right * delta_right - k_left * delta_left

        for dist, kval in long_range:
            left_d, right_d = _neighbor_values(x, dist, boundary)
            force += kval * (right_d + left_d - 2.0 * x)

        # beta-FPUT: force from quartic spring potential (delta^4 / 4)
        if has_beta:
            force += nonlinear_coeff * (delta_right**3 - delta_left**3)

        # alpha-FPUT: force from cubic spring potential (delta^3 / 3)
        if has_alpha:
            force += nonlinear_fput_alpha * (delta_right**2 - delta_left**2)

        if has_quartic:
            force += nonlinear_quartic * np.sign(lap_1) * np.abs(lap_1) ** 4

        if has_quintic:
            force += nonlinear_quintic * lap_1**5

        if has_external:
            force += external_amplitude * np.cos(external_frequency * t)

        dydt = np.empty_like(y)
        dydt[:n] = v
        dydt[n:] = force / masses_arr
        return dydt

    return ode_func


def _is_uniform(
    masses: float | list[float] | Callable[[int], float],
    k_coupling: float | list[float] | Callable[[int], float],
    n: int,
) -> bool:
    """True when masses and nearest-neighbor couplings are uniform."""
    if callable(masses) or callable(k_coupling):
        return False
    masses_arr = _resolve_mass_array(masses, n)
    k_arr = _resolve_k_array(k_coupling, n - 1, n)
    return (
        np.allclose(masses_arr, masses_arr[0], rtol=_UNIFORM_RTOL, atol=_UNIFORM_ATOL)
        and np.allclose(k_arr, k_arr[0], rtol=_UNIFORM_RTOL, atol=_UNIFORM_ATOL)
    )


def _build_stiffness_matrix(
    n: int,
    boundary: str,
    k_left: np.ndarray,
    k_right: np.ndarray,
    long_range: list[tuple[int, float]],
) -> np.ndarray:
    """Build stiffness matrix K for x'' = -M^-1 K x."""
    K = np.zeros((n, n), dtype=float)

    for i in range(n):
        K[i, i] += k_left[i] + k_right[i]
        if boundary == "periodic":
            K[i, (i - 1) % n] -= k_left[i]
            K[i, (i + 1) % n] -= k_right[i]
        else:
            if i > 0:
                K[i, i - 1] -= k_left[i]
            if i < n - 1:
                K[i, i + 1] -= k_right[i]

    for dist, kval in long_range:
        if boundary == "periodic":
            for i in range(n):
                K[i, i] += 2.0 * kval
                K[i, (i - dist) % n] -= kval
                K[i, (i + dist) % n] -= kval
        else:
            for i in range(n):
                K[i, i] += 2.0 * kval
                j_left = i - dist
                j_right = i + dist
                if j_left >= 0:
                    K[i, j_left] -= kval
                if j_right < n:
                    K[i, j_right] -= kval

    return K


def compute_normal_modes(
    n_oscillators: int,
    masses: float | list[float] | Callable[[int], float],
    k_coupling: float | list[float] | Callable[[int], float],
    boundary: str = "fixed",
    k_2nn: float = 0.0,
    k_3nn: float = 0.0,
    k_4nn: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normal modes and angular frequencies for the linearized system."""
    if boundary not in {"fixed", "periodic"}:
        raise ValueError("Boundary must be 'fixed' or 'periodic'.")
    if n_oscillators < 2:
        raise ValueError("At least 2 oscillators are required.")

    n = n_oscillators
    n_springs = n - 1 if boundary == "fixed" else n

    masses_arr = _resolve_mass_array(masses, n)
    k_nearest = _resolve_k_array(k_coupling, n_springs, n)
    k_left, k_right = _build_nearest_k_arrays(k_nearest, boundary)
    long_range = [(2, float(k_2nn)), (3, float(k_3nn)), (4, float(k_4nn))]
    long_range = [(dist, kval) for dist, kval in long_range if kval != 0.0]

    uniform_mass = np.allclose(
        masses_arr, masses_arr[0], rtol=_UNIFORM_RTOL, atol=_UNIFORM_ATOL
    )
    uniform_k = np.allclose(
        k_nearest, k_nearest[0], rtol=_UNIFORM_RTOL, atol=_UNIFORM_ATOL
    )
    if boundary == "fixed" and uniform_mass and uniform_k and not long_range:
        m0 = float(masses_arr[0])
        k0 = float(k_nearest[0])

        j = np.arange(1, n + 1, dtype=float)[:, np.newaxis]
        mode_idx = np.arange(1, n + 1, dtype=float)[np.newaxis, :]
        modes = np.sin(np.pi * j * mode_idx / (n + 1))

        # M-orthonormalization (here M = m0 I).
        norms = np.sqrt(m0 * np.sum(modes**2, axis=0))
        modes = modes / norms[np.newaxis, :]

        omega = 2.0 * np.sqrt(k0 / m0) * np.sin(np.pi * mode_idx.ravel() / (2.0 * (n + 1)))
        return modes, omega

    K_mat = _build_stiffness_matrix(n, boundary, k_left, k_right, long_range)
    M_mat = np.diag(masses_arr)

    try:
        from scipy.linalg import eigh

        omega_sq, modes = eigh(K_mat, b=M_mat, check_finite=False)
        omega_sq = np.maximum(omega_sq, 0.0)
        omega = np.sqrt(omega_sq)

        # Deterministic sign convention for display stability.
        for idx in range(n):
            non_zero = np.flatnonzero(np.abs(modes[:, idx]) > _EPS_SIGN)
            if non_zero.size and modes[non_zero[0], idx] < 0:
                modes[:, idx] *= -1.0

        return modes, omega
    except Exception:  # pragma: no cover - fallback path
        logger.warning("Could not compute normal modes; returning identity basis")
        return np.eye(n), np.ones(n)

