"""ODE functions for complex differential equations defined in code.

Functions here are callable as f(x, y, **params) and return dy/dx as a 1-D numpy array.
They can be referenced from equations.yaml via function_name.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def harmonic_oscillator(x: float, y: np.ndarray, omega: float = 1.0, **kwargs: Any) -> np.ndarray:
    """y'' + ω²y = 0 — Simple harmonic oscillator."""
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = -omega**2 * y[0]
    return dydt


def damped_oscillator(
    x: float, y: np.ndarray, omega: float = 1.0, gamma: float = 0.1, **kwargs: Any
) -> np.ndarray:
    """y'' + 2γy' + ω²y = 0 — Damped oscillator."""
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = -2 * gamma * y[1] - omega**2 * y[0]
    return dydt


def exponential_growth(x: float, y: np.ndarray, k: float = 0.5, **kwargs: Any) -> np.ndarray:
    """y' = ky — Exponential growth or decay."""
    dydt = np.empty(1)
    dydt[0] = k * y[0]
    return dydt


def logistic_equation(
    x: float, y: np.ndarray, r: float = 1.0, K: float = 10.0, **kwargs: Any
) -> np.ndarray:
    """y' = ry(1 - y/K) — Logistic population growth."""
    dydt = np.empty(1)
    dydt[0] = r * y[0] * (1 - y[0] / K)
    return dydt


def van_der_pol(x: float, y: np.ndarray, mu: float = 1.0, **kwargs: Any) -> np.ndarray:
    """y'' - μ(1 - y²)y' + y = 0 — Van der Pol oscillator."""
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = mu * (1 - y[0] ** 2) * y[1] - y[0]
    return dydt


def simple_pendulum(
    x: float, y: np.ndarray, g: float = -9.81, L: float = 1.0, **kwargs: Any
) -> np.ndarray:
    """y'' + (g/L)sin(y) = 0 — Nonlinear pendulum."""
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = -(g / L) * np.sin(y[0])
    return dydt


def rc_circuit(
    x: float, y: np.ndarray, R: float = 1000.0, C: float = 0.001, **kwargs: Any
) -> np.ndarray:
    """y' = -y/(RC) — RC circuit discharge."""
    dydt = np.empty(1)
    dydt[0] = -y[0] / (R * C)
    return dydt


def free_fall_drag(
    x: float, y: np.ndarray, g: float = -9.81, b: float = 0.5, m: float = 1.0, **kwargs: Any
) -> np.ndarray:
    """y'' = g - (b/m)y' — Free fall with linear drag."""
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = g - (b / m) * y[1]
    return dydt


# --- Schrödinger equation (time-dependent Hamiltonian) ---


def _default_hamiltonian(x: float, a: float) -> float:
    """Default time-dependent Hamiltonian: H(x) = a * (1 + 0.1*sin(x))."""
    return a * (1.0 + 0.1 * np.sin(x))


def _default_potential(x: float, b: float) -> float:
    """Default time-dependent potential: V(x) = b * x² * exp(-0.1*x²)."""
    return b * x**2 * np.exp(-0.1 * x**2)


def schrodinger_equation(
    x: float,
    y: np.ndarray,
    a: float = 1.0,
    b: float = 0.5,
    hamiltonian_function: Any = None,
    potential_function: Any = None,
    **kwargs: Any,
) -> np.ndarray:
    """Time-dependent Schrödinger equation: iℏ ∂ψ/∂t = Ĥ(t)ψ.

    Uses Hamiltonian H(x) and potential V(x) as functions of time x.
    y = [Re(ψ), Im(ψ)] for a single-component wave function.
    """
    H = (_default_hamiltonian if hamiltonian_function is None else hamiltonian_function)(x, a)
    V = (_default_potential if potential_function is None else potential_function)(x, b)
    # i dψ/dt = (H + V)ψ  →  d(Reψ)/dt = (H+V)*Imψ,  d(Imψ)/dt = -(H+V)*Reψ
    # Using ℏ=1
    omega = H + V
    dydt = np.empty(2)
    dydt[0] = omega * y[1]   # d(Reψ)/dt
    dydt[1] = -omega * y[0]  # d(Imψ)/dt
    return dydt


# --- Additional physics equations ---


def lorentz_system(
    x: float,
    y: np.ndarray,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    **kwargs: Any,
) -> np.ndarray:
    """Lorenz system: chaotic attractor."""
    dydt = np.empty(3)
    dydt[0] = sigma * (y[1] - y[0])
    dydt[1] = y[0] * (rho - y[2]) - y[1]
    dydt[2] = y[0] * y[1] - beta * y[2]
    return dydt


def duffing_oscillator(
    x: float,
    y: np.ndarray,
    delta: float = 0.5,
    alpha: float = -1.0,
    beta: float = 1.0,
    gamma: float = 0.3,
    omega: float = 1.2,
    **kwargs: Any,
) -> np.ndarray:
    """Duffing oscillator: y'' + δy' + αy + βy³ = γcos(ωt)."""
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = -delta * y[1] - alpha * y[0] - beta * y[0] ** 3 + gamma * np.cos(omega * x)
    return dydt


def lotka_volterra(
    x: float,
    y: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.1,
    gamma: float = 1.5,
    delta: float = 0.075,
    **kwargs: Any,
) -> np.ndarray:
    """Lotka-Volterra predator-prey: dx/dt = αx - βxy, dy/dt = δxy - γy."""
    dydt = np.empty(2)
    dydt[0] = alpha * y[0] - beta * y[0] * y[1]
    dydt[1] = delta * y[0] * y[1] - gamma * y[1]
    return dydt


def rigid_body_euler(
    x: float,
    y: np.ndarray,
    I1: float = 1.0,
    I2: float = 2.0,
    I3: float = 3.0,
    **kwargs: Any,
) -> np.ndarray:
    """Euler equations for rigid body: dω/dt = I⁻¹(τ - ω×(Iω)). No torque."""
    omega1, omega2, omega3 = y[0], y[1], y[2]
    dydt = np.empty(3)
    dydt[0] = ((I2 - I3) / I1) * omega2 * omega3
    dydt[1] = ((I3 - I1) / I2) * omega3 * omega1
    dydt[2] = ((I1 - I2) / I3) * omega1 * omega2
    return dydt


def bloch_equations(
    x: float,
    y: np.ndarray,
    gamma: float = 1.0,
    Bx: float = 0.0,
    By: float = 0.0,
    Bz: float = 1.0,
    T1: float = 1e6,
    T2: float = 1e6,
    **kwargs: Any,
) -> np.ndarray:
    """Bloch equations for magnetization: dM/dt = γ(M×B) - relaxation."""
    Mx, My, Mz = y[0], y[1], y[2]
    dydt = np.empty(3)
    dydt[0] = gamma * (My * Bz - Mz * By) - Mx / T2
    dydt[1] = gamma * (Mz * Bx - Mx * Bz) - My / T2
    dydt[2] = gamma * (Mx * By - My * Bx) - (Mz - 1.0) / T1
    return dydt
