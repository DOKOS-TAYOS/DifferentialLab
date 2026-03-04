"""Physics model for coupled harmonic oscillators."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from utils import get_logger

logger = get_logger(__name__)


def _resolve_mass(masses_spec: Any, i: int, n: int) -> float:
    """Resolve mass for oscillator i.

    Args:
        masses_spec: Constant float, list of length n, or callable(i) -> float.
        i: Oscillator index.
        n: Total number of oscillators.

    Returns:
        Mass value for oscillator i.
    """
    if callable(masses_spec):
        return float(masses_spec(i))
    if isinstance(masses_spec, (list, tuple, np.ndarray)):
        return float(masses_spec[i] if i < len(masses_spec) else masses_spec[-1])
    return float(masses_spec)


def _resolve_k(k_spec: Any, i: int, n: int) -> float:
    """Resolve coupling constant for spring between oscillator i and i+1.

    Args:
        k_spec: Constant float, list of length n-1 (or n), or callable(i) -> float.
        i: Spring index (between oscillator i and i+1).
        n: Total number of oscillators.

    Returns:
        Coupling constant for spring i.
    """
    if callable(k_spec):
        return float(k_spec(i))
    if isinstance(k_spec, (list, tuple, np.ndarray)):
        idx = min(i, len(k_spec) - 1) if k_spec else 0
        return float(k_spec[idx])
    return float(k_spec)


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
    """Build the ODE right-hand side for coupled oscillators.

    State vector y has shape (2*N,): [x_0, ..., x_{N-1}, v_0, ..., v_{N-1}].

    Args:
        n_oscillators: Number of oscillators N.
        masses: Mass per oscillator (constant, list, or callable).
        k_coupling: Coupling constant for 1st neighbors.
        boundary: "fixed" (x_{-1}=x_N=0) or "periodic".
        coupling_types: List of "linear", "nonlinear", "nonlinear_fput_alpha",
            "nonlinear_quartic", "nonlinear_quintic", "external_force".
        nonlinear_coeff: Coefficient for cubic nonlinear coupling (FPUT-β).
        nonlinear_fput_alpha: Coefficient for FPUT-α: α·(x_{i+1}+x_{i-1}-2x_i)·(x_{i+1}-x_{i-1}).
        nonlinear_quartic: Coefficient for quartic nonlinear (F ∝ sign(Δ)·Δ⁴).
        nonlinear_quintic: Coefficient for quintic nonlinear (F ∝ Δ⁵).
        k_2nn: Coupling for 2nd neighbors (0 = disabled).
        k_3nn: Coupling for 3rd neighbors (0 = disabled).
        k_4nn: Coupling for 4th neighbors (0 = disabled).
        external_amplitude: Amplitude of external driving force.
        external_frequency: Angular frequency of external force.

    Returns:
        ODE function (t, y) -> dydt.
    """
    coupling_types = coupling_types or ["linear"]
    n = n_oscillators

    def ode_func(t: float, y: np.ndarray) -> np.ndarray:
        x = y[:n]
        v = y[n:]

        def _x(idx: int) -> float:
            """Position at index, with boundary: x[-j]=x[n+j]=0 for fixed; wrap for periodic."""
            if boundary == "fixed":
                if idx < 0 or idx >= n:
                    return 0.0
            else:
                idx = idx % n
            return float(x[idx])

        def _laplacian(i: int, d: int) -> float:
            """Centered discrete Laplacian: x[i+d] + x[i-d] - 2*x[i]."""
            return _x(i + d) + _x(i - d) - 2.0 * x[i]

        dydt = np.zeros_like(y)
        dydt[:n] = v

        for i in range(n):
            m_i = _resolve_mass(masses, i, n)
            if m_i <= 0:
                m_i = 1.0

            force = 0.0

            # Linear coupling: 1st neighbors (Laplacian, centered)
            if "linear" in coupling_types:
                L1 = _laplacian(i, 1)
                k_eff = (
                    0.5 * (_resolve_k(k_coupling, i - 1, n) + _resolve_k(k_coupling, i, n))
                    if 0 < i < n - 1
                    else _resolve_k(k_coupling, 0 if i == 0 else n - 2, n)
                )
                force += k_eff * L1

            # Linear coupling: 2nd, 3rd, 4th neighbors (Laplacian, centered)
            for dist, k_val in [(2, k_2nn), (3, k_3nn), (4, k_4nn)]:
                if k_val != 0:
                    force += k_val * _laplacian(i, dist)

            # Cubic nonlinear: ε·L³ (centered)
            if "nonlinear" in coupling_types and nonlinear_coeff != 0:
                L1 = _laplacian(i, 1)
                force += nonlinear_coeff * L1**3

            # FPUT-α: α·L·(x_{i+1}-x_{i-1}) — already centered
            if "nonlinear_fput_alpha" in coupling_types and nonlinear_fput_alpha != 0:
                L1 = _laplacian(i, 1)
                gradient = _x(i + 1) - _x(i - 1)
                force += nonlinear_fput_alpha * L1 * gradient

            # Quartic nonlinear: ε·sign(L)·|L|⁴ (centered)
            if "nonlinear_quartic" in coupling_types and nonlinear_quartic != 0:
                L1 = _laplacian(i, 1)
                force += nonlinear_quartic * np.sign(L1) * np.abs(L1) ** 4

            # Quintic nonlinear: ε·L⁵ (centered)
            if "nonlinear_quintic" in coupling_types and nonlinear_quintic != 0:
                L1 = _laplacian(i, 1)
                force += nonlinear_quintic * L1**5

            # External driving force
            if "external_force" in coupling_types and external_amplitude != 0:
                force += external_amplitude * np.cos(external_frequency * t)

            dydt[n + i] = force / m_i

        return dydt

    return ode_func


def _is_uniform(
    masses: float | list[float] | Callable[[int], float],
    k_coupling: float | list[float] | Callable[[int], float],
    n: int,
) -> bool:
    """True if masses and k are constant (uniform chain)."""
    if callable(masses) or callable(k_coupling):
        return False
    if isinstance(masses, (list, tuple, np.ndarray)):
        if len(masses) < n:
            return False
        m0 = float(masses[0])
        if any(float(m) != m0 for m in masses[:n]):
            return False
    if isinstance(k_coupling, (list, tuple, np.ndarray)):
        n_springs = n - 1
        if len(k_coupling) < n_springs:
            return False
        k0 = float(k_coupling[0])
        if any(float(k) != k0 for k in k_coupling[:n_springs]):
            return False
    return True


def compute_normal_modes(
    n_oscillators: int,
    masses: float | list[float] | Callable[[int], float],
    k_coupling: float | list[float] | Callable[[int], float],
    boundary: str = "fixed",
    k_2nn: float = 0.0,
    k_3nn: float = 0.0,
    k_4nn: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normal modes for linear system (fixed or periodic boundary).

    For fixed ends and uniform chain with only 1st-neighbor coupling: uses analytic formula
    u_j^(k) = sin((k+1)*pi*(j+1)/(N+1)) for mode k, oscillator j.

    Otherwise solves K @ v = omega^2 * M @ v.

    Returns:
        Tuple of (M_modes, omega_modes). M_modes has shape (n, n), columns are
        mode vectors. omega_modes is 1D array of angular frequencies.
    """
    n = n_oscillators
    has_long_range = k_2nn != 0 or k_3nn != 0 or k_4nn != 0

    if (
        boundary == "fixed"
        and not has_long_range
        and _is_uniform(masses, k_coupling, n)
    ):
        m = float(_resolve_mass(masses, 0, n))
        k_val = float(_resolve_k(k_coupling, 0, n))
        # Analytic: mode k (0-indexed), oscillator j (0-indexed)
        # u_j = sin((k+1) * pi * (j+1) / (N+1))
        modes = np.zeros((n, n))
        for k in range(n):
            for j in range(n):
                modes[j, k] = np.sin((k + 1) * np.pi * (j + 1) / (n + 1))
        # Normalize so that modes^T M modes = I (M = m*I for uniform)
        for k in range(n):
            col = modes[:, k]
            col /= np.sqrt(np.sum(m * col**2))
        # omega_k = 2*sqrt(k/m) * sin((k+1)*pi/(2*(N+1)))
        omega = np.array(
            [
                2 * np.sqrt(k_val / m) * np.sin((k + 1) * np.pi / (2 * (n + 1)))
                for k in range(n)
            ]
        )
        return modes, omega

    M_mat = np.diag([_resolve_mass(masses, i, n) for i in range(n)])
    K_mat = np.zeros((n, n))

    for i in range(n):
        if boundary == "fixed":
            # Wall springs at x_{-1}=x_N=0 add diagonal terms for end oscillators
            if i == 0:
                k_wall = _resolve_k(k_coupling, 0, n)
                K_mat[i, i] += k_wall
            if i > 0:
                k_prev = _resolve_k(k_coupling, i - 1, n)
                K_mat[i, i] += k_prev
                K_mat[i, i - 1] -= k_prev
            if i < n - 1:
                k_curr = _resolve_k(k_coupling, i, n)
                K_mat[i, i] += k_curr
                K_mat[i, i + 1] -= k_curr
            if i == n - 1:
                k_wall = _resolve_k(k_coupling, n - 2, n)
                K_mat[i, i] += k_wall
        else:
            k_prev = _resolve_k(k_coupling, (i - 1) % n, n)
            k_curr = _resolve_k(k_coupling, i, n)
            K_mat[i, i] += k_prev + k_curr
            K_mat[i, (i - 1) % n] -= k_prev
            K_mat[i, (i + 1) % n] -= k_curr

        # 2nd, 3rd, 4th neighbor coupling
        for dist, k_val in [(2, k_2nn), (3, k_3nn), (4, k_4nn)]:
            if k_val == 0:
                continue
            if boundary == "fixed":
                if i < dist:
                    K_mat[i, i] += k_val  # left wall
                else:
                    K_mat[i, i] += k_val
                    K_mat[i, i - dist] -= k_val
                if i + dist < n:
                    K_mat[i, i] += k_val
                    K_mat[i, i + dist] -= k_val
                else:
                    K_mat[i, i] += k_val  # right wall
            else:
                j_prev = (i - dist) % n
                j_next = (i + dist) % n
                K_mat[i, i] += 2 * k_val
                K_mat[i, j_prev] -= k_val
                K_mat[i, j_next] -= k_val

    try:
        from scipy.linalg import eigh

        omega_sq, modes = eigh(K_mat, b=M_mat)
        omega_sq = np.maximum(omega_sq, 0.0)
        omega = np.sqrt(omega_sq)
        # Align sign: first oscillator (j=0) should be positive for mode 0
        for k in range(n):
            if modes[0, k] < 0:
                modes[:, k] *= -1
        return modes, omega
    except np.linalg.LinAlgError:
        logger.warning("Could not compute normal modes; returning identity")
        return np.eye(n), np.ones(n)
