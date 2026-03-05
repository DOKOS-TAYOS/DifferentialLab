"""ODE functions for complex differential equations defined in code.

Functions here are callable as ``f(x, y, **params)`` and return dy/dx as a 1-D
numpy array. They can be referenced from config/equations/*.yaml via function_name.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# --- Schrödinger equation (time-dependent Hamiltonian) ---


def _default_hamiltonian(x: float, a: float) -> float:
    """Default time-dependent Hamiltonian: H(x) = a * (1 + 0.1*sin(x)).

    Args:
        x: Time variable.
        a: Hamiltonian scale parameter.

    Returns:
        Hamiltonian value at time x.
    """
    return a * (1.0 + 0.1 * np.sin(x))


def _default_potential(x: float, b: float) -> float:
    """Default time-dependent potential: V(x) = b * x² * exp(-0.1*x²).

    Args:
        x: Position variable.
        b: Potential scale parameter.

    Returns:
        Potential value at position x.
    """
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

    Args:
        x: Independent variable (time).
        y: State vector ``[Re(ψ), Im(ψ)]``.
        a: Hamiltonian scale parameter.
        b: Potential scale parameter.
        hamiltonian_function: Custom H(x, a). If None, uses default.
        potential_function: Custom V(x, b). If None, uses default.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    H = (_default_hamiltonian if hamiltonian_function is None else hamiltonian_function)(x, a)
    V = (_default_potential if potential_function is None else potential_function)(x, b)
    # i dψ/dt = (H + V)ψ  →  d(Reψ)/dt = (H+V)*Imψ,  d(Imψ)/dt = -(H+V)*Reψ
    # Using ℏ=1
    omega = H + V
    dydt = np.empty(2)
    dydt[0] = omega * y[1]  # d(Reψ)/dt
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
    """Lorenz system: chaotic attractor.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        sigma: Prandtl number.
        rho: Rayleigh number.
        beta: Geometric factor.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
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
    """Duffing oscillator: y'' + δy' + αy + βy³ = γcos(ωt).

    Args:
        x: Independent variable (time).
        y: State vector ``[y, y']``.
        delta: Damping.
        alpha: Linear stiffness.
        beta: Cubic stiffness.
        gamma: Forcing amplitude.
        omega: Forcing frequency.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
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
    """Lotka-Volterra predator-prey: dx/dt = αx - βxy, dy/dt = δxy - γy.

    Args:
        x: Independent variable (time).
        y: State vector ``[x, y]`` (prey, predator).
        alpha: Prey growth rate.
        beta: Predation rate.
        gamma: Predator death rate.
        delta: Predator growth from prey.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
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
    """Euler equations for rigid body: dω/dt = I⁻¹(τ - ω×(Iω)). No torque.

    Args:
        x: Independent variable (time).
        y: State vector ``[ω₁, ω₂, ω₃]`` (angular velocities).
        I1: Principal moment of inertia 1.
        I2: Principal moment of inertia 2.
        I3: Principal moment of inertia 3.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    omega1, omega2, omega3 = y[0], y[1], y[2]
    dydt = np.empty(3)
    dydt[0] = ((I2 - I3) / I1) * omega2 * omega3
    dydt[1] = ((I3 - I1) / I2) * omega3 * omega1
    dydt[2] = ((I1 - I2) / I3) * omega1 * omega2
    return dydt


def rlc_circuit(
    x: float,
    y: np.ndarray,
    R: float = 1.0,
    L: float = 1.0,
    C: float = 1.0,
    **kwargs: Any,
) -> np.ndarray:
    """y'' + (R/L)y' + (1/LC)y = 0 — RLC series circuit (zero forcing).

    Args:
        x: Independent variable (time).
        y: State vector ``[q, q']`` (charge, current).
        R: Resistance.
        L: Inductance.
        C: Capacitance.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = -(R / L) * y[1] - (1.0 / (L * C)) * y[0]
    return dydt


def gompertz_growth(
    x: float, y: np.ndarray, r: float = 0.5, K: float = 10.0, **kwargs: Any
) -> np.ndarray:
    """y' = ry·ln(K/y) — Gompertz growth (tumor, cell populations).

    Args:
        x: Independent variable (time).
        y: State vector ``[y]``.
        r: Growth rate.
        K: Carrying capacity.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = r * y[0] * np.log(K / np.maximum(y[0], 1e-10))
    return dydt


def newton_cooling(
    x: float, y: np.ndarray, k: float = 0.1, T_env: float = 20.0, **kwargs: Any
) -> np.ndarray:
    """y' = -k(y - T_env) — Newton's law of cooling.

    Args:
        x: Independent variable (time).
        y: State vector ``[T]`` (temperature).
        k: Cooling coefficient.
        T_env: Ambient temperature.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = -k * (y[0] - T_env)
    return dydt


def linear_decay_source(
    x: float, y: np.ndarray, a: float = 1.0, b: float = 0.5, **kwargs: Any
) -> np.ndarray:
    """y' = a - by — Linear decay with constant source (e.g. drug concentration).

    Args:
        x: Independent variable (time).
        y: State vector ``[y]``.
        a: Source/inflow rate.
        b: Decay rate.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = a - b * y[0]
    return dydt


def airy_equation(x: float, y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """y'' = x·y — Airy equation (quantum mechanics, optics).

    Args:
        x: Independent variable.
        y: State vector ``[y, y']``.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = x * y[0]
    return dydt


# --- Additional quantum equations ---


def hermite_ode(x: float, y: np.ndarray, n: float = 2.0, **kwargs: Any) -> np.ndarray:
    """y'' - 2xy' + 2ny = 0 — Hermite equation (QHO eigenfunctions).

    Args:
        x: Independent variable.
        y: State vector ``[y, y']``.
        n: Quantum number / eigenvalue parameter.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = 2 * x * y[1] - 2 * n * y[0]
    return dydt


def laguerre_ode(x: float, y: np.ndarray, n: float = 1.0, **kwargs: Any) -> np.ndarray:
    """xy'' + (1-x)y' + ny = 0 — Laguerre equation (hydrogen radial).

    Args:
        x: Independent variable (r).
        y: State vector ``[y, y']``.
        n: Parameter (integer for Laguerre polynomials).
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    xx = max(x, 1e-8)
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = (-(1 - xx) * y[1] - n * y[0]) / xx
    return dydt


def bessel_ode(x: float, y: np.ndarray, n: float = 0.0, **kwargs: Any) -> np.ndarray:
    """y'' + y'/x + (1 - n²/x²)y = 0 — Bessel equation (radial QM).

    Args:
        x: Independent variable (x > 0).
        y: State vector ``[y, y']``.
        n: Order (integer for Bessel functions).
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    xx = max(x, 1e-8)
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = -y[1] / xx - (1 - n**2 / xx**2) * y[0]
    return dydt


def stationary_schrodinger_ho(x: float, y: np.ndarray, E: float = 1.5, **kwargs: Any) -> np.ndarray:
    """y'' + (E - x²)y = 0 — 1D Schrödinger in harmonic well (ℏ=m=ω=1).

    Args:
        x: Position.
        y: State vector ``[ψ, ψ']`` (wave function).
        E: Energy eigenvalue.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = -(E - x**2) * y[0]
    return dydt


def stationary_schrodinger_well(
    x: float, y: np.ndarray, E: float = 2.0, V0: float = 10.0, a: float = 1.0, **kwargs: Any
) -> np.ndarray:
    """y'' + (E - V(x))y = 0 — 1D Schrödinger, finite square well.

    V(x) = V0 for ``|x| > a``, 0 otherwise. Units ℏ²/(2m)=1.

    Args:
        x: Position.
        y: State vector ``[ψ, ψ']`` (wave function).
        E: Energy eigenvalue.
        V0: Well depth.
        a: Half-width of the well.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    v_x = V0 if abs(x) > a else 0.0
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = -(E - v_x) * y[0]
    return dydt


def kummer_ode(
    x: float, y: np.ndarray, a: float = 1.0, b: float = 1.0, **kwargs: Any
) -> np.ndarray:
    """xy'' + (b-x)y' - ay = 0 — Kummer (confluent hypergeometric); hydrogen.

    Args:
        x: Independent variable.
        y: State vector ``[y, y']``.
        a: Kummer parameter a.
        b: Kummer parameter b.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    xx = max(x, 1e-8)
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = ((xx - b) * y[1] + a * y[0]) / xx
    return dydt


def rabi_oscillations(
    x: float,
    y: np.ndarray,
    Omega: float = 1.0,
    delta: float = 0.0,
    omega_drive: float = 1.0,
    **kwargs: Any,
) -> np.ndarray:
    """Rabi oscillations: two-level system. y = [Re(c_g), Im(c_g), Re(c_e), Im(c_e)].

    dc_g/dt = -i(Ω/2)cos(ωt)c_e, dc_e/dt = -i(Ω/2)cos(ωt)c_g - iΔ·c_e.

    Args:
        x: Independent variable (time).
        y: State vector ``[Re(c_g), Im(c_g), Re(c_e), Im(c_e)]``.
        Omega: Rabi frequency.
        delta: Detuning.
        omega_drive: Drive frequency.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    drive = (Omega / 2) * np.cos(omega_drive * x)
    dydt = np.empty(4)
    dydt[0] = drive * y[3]  # d(Re(c_g))/dt = (Ω/2)cos(ωt) Im(c_e)
    dydt[1] = -drive * y[2]  # d(Im(c_g))/dt = -(Ω/2)cos(ωt) Re(c_e)
    dydt[2] = drive * y[1] + delta * y[3]  # d(Re(c_e))/dt
    dydt[3] = -drive * y[0] - delta * y[2]  # d(Im(c_e))/dt
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
    """Bloch equations for magnetization: dM/dt = γ(M×B) - relaxation.

    Args:
        x: Independent variable (time).
        y: State vector ``[Mx, My, Mz]``.
        gamma: Gyromagnetic ratio.
        Bx: Magnetic field x-component.
        By: Magnetic field y-component.
        Bz: Magnetic field z-component.
        T1: Longitudinal relaxation time.
        T2: Transverse relaxation time.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    Mx, My, Mz = y[0], y[1], y[2]
    dydt = np.empty(3)
    dydt[0] = gamma * (My * Bz - Mz * By) - Mx / T2
    dydt[1] = gamma * (Mz * Bx - Mx * Bz) - My / T2
    dydt[2] = gamma * (Mx * By - My * Bx) - (Mz - 1.0) / T1
    return dydt


# --- Additional vector ODEs ---


def rossler_attractor(
    x: float,
    y: np.ndarray,
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
    **kwargs: Any,
) -> np.ndarray:
    """Rössler attractor: chaotic system.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        a: Parameter a.
        b: Parameter b.
        c: Parameter c.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = -y[1] - y[2]
    dydt[1] = y[0] + a * y[1]
    dydt[2] = b + y[2] * (y[0] - c)
    return dydt


def sir_epidemic(
    x: float,
    y: np.ndarray,
    beta: float = 0.5,
    gamma: float = 0.2,
    **kwargs: Any,
) -> np.ndarray:
    """SIR epidemic model: dS/dt = -βSI, dI/dt = βSI - γI, dR/dt = γI.

    Args:
        x: Independent variable (time).
        y: State vector ``[S, I, R]`` (susceptible, infected, recovered).
        beta: Transmission rate.
        gamma: Recovery rate.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    susceptible, infected, _ = y[0], y[1], y[2]
    dydt = np.empty(3)
    dydt[0] = -beta * susceptible * infected
    dydt[1] = beta * susceptible * infected - gamma * infected
    dydt[2] = gamma * infected
    return dydt


def fitzhugh_nagumo(
    x: float,
    y: np.ndarray,
    a: float = 0.7,
    b: float = 0.8,
    I_ext: float = 0.0,
    **kwargs: Any,
) -> np.ndarray:
    """FitzHugh-Nagumo neuron model: simplified Hodgkin-Huxley.

    Args:
        x: Independent variable (time).
        y: State vector ``[v, w]`` (membrane potential, recovery variable).
        a: Parameter a.
        b: Parameter b.
        I_ext: External current.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    v, w = y[0], y[1]
    dydt = np.empty(2)
    dydt[0] = v - v**3 / 3.0 - w + I_ext
    dydt[1] = (v + a - b * w) / a
    return dydt


def chen_system(
    x: float,
    y: np.ndarray,
    a: float = 35.0,
    b: float = 3.0,
    c: float = 28.0,
    **kwargs: Any,
) -> np.ndarray:
    """Chen system: chaotic attractor (dual to Lorenz).

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        a: Parameter a.
        b: Parameter b.
        c: Parameter c.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = a * (y[1] - y[0])
    dydt[1] = (c - a) * y[0] - y[0] * y[2] + c * y[1]
    dydt[2] = y[0] * y[1] - b * y[2]
    return dydt


def brusselator(
    x: float,
    y: np.ndarray,
    A: float = 1.0,
    B: float = 3.0,
    **kwargs: Any,
) -> np.ndarray:
    """Brusselator: autocatalytic chemical reaction (limit cycle).

    Args:
        x: Independent variable (time).
        y: State vector ``[u, v]`` (concentrations).
        A: Parameter A.
        B: Parameter B.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    u, v = y[0], y[1]
    dydt = np.empty(2)
    dydt[0] = A - (B + 1) * u + u**2 * v
    dydt[1] = B * u - u**2 * v
    return dydt


# --- Additional scalar ODEs ---


def michaelis_menten(
    x: float,
    y: np.ndarray,
    V_max: float = 1.0,
    K_m: float = 0.5,
    **kwargs: Any,
) -> np.ndarray:
    """y' = -V_max·y/(K_m + y) — Michaelis-Menten enzyme kinetics.

    Args:
        x: Independent variable (time).
        y: State vector ``[S]`` (substrate concentration).
        V_max: Maximum reaction rate.
        K_m: Michaelis constant.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = -V_max * y[0] / (K_m + y[0])
    return dydt


def riccati_ode(
    x: float,
    y: np.ndarray,
    a: float = 1.0,
    b: float = 0.5,
    c: float = -0.1,
    **kwargs: Any,
) -> np.ndarray:
    """y' = a + b·y + c·y² — Riccati equation (general form).

    Args:
        x: Independent variable.
        y: State vector ``[y]``.
        a: Constant term.
        b: Linear coefficient.
        c: Quadratic coefficient.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = a + b * y[0] + c * y[0] ** 2
    return dydt


def rayleigh_oscillator(x: float, y: np.ndarray, mu: float = 1.0, **kwargs: Any) -> np.ndarray:
    """y'' - μ(1 - y'²)y' + y = 0 — Rayleigh oscillator (self-excited).

    Args:
        x: Independent variable (time).
        y: State vector ``[y, y']``.
        mu: Nonlinearity parameter.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = mu * (1 - y[1] ** 2) * y[1] - y[0]
    return dydt


def forced_harmonic_oscillator(
    x: float,
    y: np.ndarray,
    omega: float = 1.0,
    F: float = 0.5,
    Omega: float = 1.2,
    **kwargs: Any,
) -> np.ndarray:
    """y'' + ω²y = F·cos(Ωt) — Forced harmonic oscillator.

    Args:
        x: Independent variable (time).
        y: State vector ``[y, y']``.
        omega: Natural frequency.
        F: Forcing amplitude.
        Omega: Forcing frequency.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = -(omega**2) * y[0] + F * np.cos(Omega * x)
    return dydt


def allee_effect(
    x: float,
    y: np.ndarray,
    r: float = 1.0,
    K: float = 10.0,
    m: float = 0.2,
    **kwargs: Any,
) -> np.ndarray:
    """y' = ry(1 - y/K)(y/K - m) — Allee effect (critical threshold).

    Args:
        x: Independent variable (time).
        y: State vector ``[y]``.
        r: Growth rate.
        K: Carrying capacity.
        m: Allee threshold (0 < m < 1).
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    yy = y[0] / K
    dydt[0] = r * y[0] * (1 - yy) * (yy - m)
    return dydt


def langmuir_adsorption(
    x: float,
    y: np.ndarray,
    k_a: float = 1.0,
    k_d: float = 0.2,
    **kwargs: Any,
) -> np.ndarray:
    """y' = k_a(1 - y) - k_d·y — Langmuir adsorption/desorption.

    Args:
        x: Independent variable (time).
        y: State vector ``[θ]`` (surface coverage 0–1).
        k_a: Adsorption rate.
        k_d: Desorption rate.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = k_a * (1 - y[0]) - k_d * y[0]
    return dydt


# --- Additional vector ODEs ---


def thomas_attractor(
    x: float,
    y: np.ndarray,
    b: float = 0.208186,
    **kwargs: Any,
) -> np.ndarray:
    """Thomas' cyclically symmetric attractor: chaotic 3D system.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        b: Parameter (typically ~0.2).
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = np.sin(y[1]) - b * y[0]
    dydt[1] = np.sin(y[2]) - b * y[1]
    dydt[2] = np.sin(y[0]) - b * y[2]
    return dydt


def hindmarsh_rose(
    x: float,
    y: np.ndarray,
    a: float = 1.0,
    b: float = 3.0,
    c: float = 1.0,
    d: float = 5.0,
    r: float = 0.01,
    s: float = 4.0,
    x1: float = -1.6,
    I_ext: float = 3.0,
    **kwargs: Any,
) -> np.ndarray:
    """Hindmarsh-Rose neuron model: 3D bursting dynamics.

    Args:
        x: Independent variable (time).
        y: State vector ``[x, y, z]`` (membrane, recovery, slow).
        a: Model parameter.
        b: Model parameter.
        c: Model parameter.
        d: Model parameter.
        r: Model parameter.
        s: Model parameter.
        x1: Model parameter.
        I_ext: External current.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    xx, yy, zz = y[0], y[1], y[2]
    dydt = np.empty(3)
    dydt[0] = yy - a * xx**3 + b * xx**2 - zz + I_ext
    dydt[1] = c - d * xx**2 - yy
    dydt[2] = r * (s * (xx - x1) - zz)
    return dydt


def rabinovich_fabrikant(
    x: float,
    y: np.ndarray,
    gamma: float = 0.87,
    alpha: float = 1.1,
    **kwargs: Any,
) -> np.ndarray:
    """Rabinovich-Fabrikant system: chaotic attractor.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        gamma: Parameter γ.
        alpha: Parameter α.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = y[1] * (y[2] - 1 + y[0] ** 2) + gamma * y[0]
    dydt[1] = y[0] * (3 * y[2] + 1 - y[0] ** 2) + gamma * y[1]
    dydt[2] = -2 * y[2] * (alpha + y[0] * y[1])
    return dydt


def chua_circuit(
    x: float,
    y: np.ndarray,
    alpha: float = 15.6,
    beta: float = 28.0,
    m0: float = -1.143,
    m1: float = -0.714,
    **kwargs: Any,
) -> np.ndarray:
    """Chua circuit: electronic chaotic oscillator.

    Args:
        x: Independent variable (time).
        y: State vector ``[x, y, z]`` (voltages, current).
        alpha: Parameter α.
        beta: Parameter β.
        m0: Piecewise slope 1.
        m1: Piecewise slope 2.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    v1, v2, i3 = y[0], y[1], y[2]
    h = m1 * v1 + 0.5 * (m0 - m1) * (np.abs(v1 + 1) - np.abs(v1 - 1))
    dydt = np.empty(3)
    dydt[0] = alpha * (v2 - v1 - h)
    dydt[1] = v1 - v2 + i3
    dydt[2] = -beta * v2
    return dydt


def selkov_glycolysis(
    x: float,
    y: np.ndarray,
    a: float = 0.05,
    b: float = 0.5,
    **kwargs: Any,
) -> np.ndarray:
    """Sel'kov model: glycolysis oscillations.

    Args:
        x: Independent variable (time).
        y: State vector ``[x, y]`` (ADP, F6P concentrations).
        a: Parameter a.
        b: Parameter b.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    xx, yy = y[0], y[1]
    dydt = np.empty(2)
    dydt[0] = -xx + yy + xx**2 * yy
    dydt[1] = a - yy - xx**2 * yy
    return dydt


def competitive_lotka_volterra_3(
    x: float,
    y: np.ndarray,
    r1: float = 1.0,
    r2: float = 1.0,
    r3: float = 1.0,
    a12: float = 0.5,
    a13: float = 0.5,
    a21: float = 0.5,
    a23: float = 0.5,
    a31: float = 0.5,
    a32: float = 0.5,
    **kwargs: Any,
) -> np.ndarray:
    """Competitive Lotka-Volterra: 3 species competition.

    dN_i/dt = r_i·N_i·(1 - N_i - Σ a_ij·N_j). f₀=N₁, f₁=N₂, f₂=N₃.

    Args:
        x: Independent variable (time).
        y: State vector ``[N₁, N₂, N₃]`` (species populations).
        r1: Growth rate of species 1.
        r2: Growth rate of species 2.
        r3: Growth rate of species 3.
        a12: Competition coefficient of 2 on 1.
        a13: Competition coefficient of 3 on 1.
        a21: Competition coefficient of 1 on 2.
        a23: Competition coefficient of 3 on 2.
        a31: Competition coefficient of 1 on 3.
        a32: Competition coefficient of 2 on 3.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    n1, n2, n3 = y[0], y[1], y[2]
    dydt = np.empty(3)
    dydt[0] = r1 * n1 * (1 - n1 - a12 * n2 - a13 * n3)
    dydt[1] = r2 * n2 * (1 - n2 - a21 * n1 - a23 * n3)
    dydt[2] = r3 * n3 * (1 - n3 - a31 * n1 - a32 * n2)
    return dydt


def lu_chen(
    x: float,
    y: np.ndarray,
    a: float = 36.0,
    b: float = 3.0,
    c: float = 20.0,
    **kwargs: Any,
) -> np.ndarray:
    """Lü-Chen system: bridge between Lorenz and Chen attractors.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        a: Parameter a.
        b: Parameter b.
        c: Parameter c.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = a * (y[1] - y[0])
    dydt[1] = -y[0] * y[2] + c * y[1]
    dydt[2] = y[0] * y[1] - b * y[2]
    return dydt


def aizawa_attractor(
    x: float,
    y: np.ndarray,
    a: float = 0.95,
    b: float = 0.7,
    c: float = 0.6,
    d: float = 3.5,
    e: float = 0.25,
    f_param: float = 0.1,
    **kwargs: Any,
) -> np.ndarray:
    """Aizawa attractor: 3D chaotic system.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        a: Parameter a.
        b: Parameter b.
        c: Parameter c.
        d: Parameter d.
        e: Parameter e.
        f_param: Parameter f.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    xx, yy, zz = y[0], y[1], y[2]
    dydt = np.empty(3)
    dydt[0] = (zz - b) * xx - d * yy
    dydt[1] = d * xx + (zz - b) * yy
    dydt[2] = c + a * zz - zz**3 / 3 - (xx**2 + yy**2) * (1 + e * zz) + f_param * zz * xx**3
    return dydt


# --- 20+ additional scalar ODEs ---


def cubic_decay(x: float, y: np.ndarray, k: float = 1.0, **kwargs: Any) -> np.ndarray:
    """y' = -k·y³ — Cubic decay (nonlinear).

    Args:
        x: Independent variable.
        y: State vector ``[y]``.
        k: Decay rate.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = -k * y[0] ** 3
    return dydt


def sqrt_growth(x: float, y: np.ndarray, k: float = 0.5, **kwargs: Any) -> np.ndarray:
    """y' = k·√y — Square-root growth (e.g. droplet).

    Args:
        x: Independent variable.
        y: State vector ``[y]``.
        k: Growth rate.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = k * np.sqrt(np.maximum(y[0], 0))
    return dydt


def bistable(x: float, y: np.ndarray, a: float = 0.5, **kwargs: Any) -> np.ndarray:
    """y' = y(1-y)(y-a) — Bistable (cubic with 3 equilibria).

    Args:
        x: Independent variable.
        y: State vector ``[y]``.
        a: Bistability parameter.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = y[0] * (1 - y[0]) * (y[0] - a)
    return dydt


def landau(x: float, y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """y' = y - y³ — Landau potential (pitchfork).

    Args:
        x: Independent variable.
        y: State vector ``[y]``.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = y[0] - y[0] ** 3
    return dydt


def cubic_oscillator(x: float, y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """y'' = -y³ — Pure cubic oscillator.

    Args:
        x: Independent variable.
        y: State vector ``[y, y']``.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = -(y[0] ** 3)
    return dydt


def bernoulli_decay(
    x: float, y: np.ndarray, k: float = 1.0, n: float = 2.0, **kwargs: Any
) -> np.ndarray:
    """y' = -k·y^n — Bernoulli-type decay.

    Args:
        x: Independent variable.
        y: State vector ``[y]``.
        k: Decay rate.
        n: Exponent.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = -k * (y[0] ** n)
    return dydt


def holling_type2(
    x: float,
    y: np.ndarray,
    r: float = 1.0,
    K: float = 10.0,
    c: float = 1.0,
    **kwargs: Any,
) -> np.ndarray:
    """y' = r·y·(K-y)/(K+c·y) — Holling type II growth.

    Args:
        x: Independent variable.
        y: State vector ``[y]``.
        r: Growth rate.
        K: Carrying capacity.
        c: Holling parameter.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = r * y[0] * (K - y[0]) / (K + c * y[0])
    return dydt


def soft_spring(
    x: float, y: np.ndarray, omega: float = 1.0, eps: float = 0.1, **kwargs: Any
) -> np.ndarray:
    """y'' = -ω²y - εy³ — Soft spring (Duffing).

    Args:
        x: Independent variable.
        y: State vector ``[y, y']``.
        omega: Natural frequency.
        eps: Nonlinearity parameter.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = -(omega**2) * y[0] - eps * y[0] ** 3
    return dydt


def quadratic_decay(
    x: float, y: np.ndarray, a: float = 1.0, b: float = 0.5, **kwargs: Any
) -> np.ndarray:
    """y' = a - b·y² — Quadratic decay with source.

    Args:
        x: Independent variable.
        y: State vector ``[y]``.
        a: Source rate.
        b: Decay coefficient.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = a - b * y[0] ** 2
    return dydt


def cubic_landau(x: float, y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """y' = y(1-y²) — Cubic (symmetric bistable).

    Args:
        x: Independent variable.
        y: State vector ``[y]``.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = y[0] * (1 - y[0] ** 2)
    return dydt


def smooth_decay(x: float, y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """y' = -y/(1+y²) — Smooth decay.

    Args:
        x: Independent variable.
        y: State vector ``[y]``.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = -y[0] / (1 + y[0] ** 2)
    return dydt


def gompertz_harvesting(
    x: float,
    y: np.ndarray,
    r: float = 0.5,
    K: float = 10.0,
    d: float = 0.1,
    **kwargs: Any,
) -> np.ndarray:
    """y' = r·y·ln(K/y) - d·y — Gompertz with harvesting.

    Args:
        x: Independent variable.
        y: State vector ``[y]``.
        r: Growth rate.
        K: Carrying capacity.
        d: Harvesting rate.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = r * y[0] * np.log(K / np.maximum(y[0], 1e-10)) - d * y[0]
    return dydt


def damped_pendulum(
    x: float, y: np.ndarray, g: float = 9.81, L: float = 1.0, gamma: float = 0.5, **kwargs: Any
) -> np.ndarray:
    """y'' + (g/L)sin(y) + γ·y' = 0 — Damped pendulum.

    Args:
        x: Independent variable (time).
        y: State vector ``[θ, θ']``.
        g: Gravitational acceleration.
        L: Pendulum length.
        gamma: Damping coefficient.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = -(g / L) * np.sin(y[0]) - gamma * y[1]
    return dydt


def lienard(x: float, y: np.ndarray, mu: float = 1.0, **kwargs: Any) -> np.ndarray:
    """y'' + μ(y²-1)y' + y = 0 — Liénard (Van der Pol-like).

    Args:
        x: Independent variable.
        y: State vector ``[y, y']``.
        mu: Nonlinearity parameter.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = -mu * (y[0] ** 2 - 1) * y[1] - y[0]
    return dydt


def matthew_equation(
    x: float, y: np.ndarray, a: float = 1.0, q: float = 0.5, **kwargs: Any
) -> np.ndarray:
    """y'' + (a - 2q·cos(2x))y = 0 — Mathieu equation.

    Args:
        x: Independent variable.
        y: State vector ``[y, y']``.
        a: Mathieu parameter.
        q: Mathieu parameter.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = -(a - 2 * q * np.cos(2 * x)) * y[0]
    return dydt


def legendre_ode(x: float, y: np.ndarray, n: float = 2.0, **kwargs: Any) -> np.ndarray:
    """(1-x²)y'' - 2xy' + n(n+1)y = 0 — Legendre."""
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = (2 * x * y[1] - n * (n + 1) * y[0]) / (1 - x**2) if abs(x) < 0.999 else 0
    return dydt


def blasius_type(x: float, y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """y''' = -y·y''/2 — Blasius (simplified). y=[f,f',f''].

    Args:
        x: Independent variable.
        y: State vector ``[f, f', f'']``.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = y[1]
    dydt[1] = y[2]
    dydt[2] = -0.5 * y[0] * y[2]
    return dydt


def emden_fowler(x: float, y: np.ndarray, n: float = 5.0, **kwargs: Any) -> np.ndarray:
    """y'' + (2/x)y' + y^n = 0 — Emden-Fowler (polytropic).

    Args:
        x: Independent variable (x > 0).
        y: State vector ``[y, y']``.
        n: Polytropic index.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    xx = max(x, 1e-8)
    dydt = np.empty(2)
    dydt[0] = y[1]
    dydt[1] = -2 * y[1] / xx - y[0] ** n
    return dydt


def fisher_kpp(x: float, y: np.ndarray, r: float = 1.0, **kwargs: Any) -> np.ndarray:
    """y' = r·y(1-y) — Fisher-KPP (spatial would need diffusion).

    Args:
        x: Independent variable.
        y: State vector ``[y]``.
        r: Growth rate.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = r * y[0] * (1 - y[0])
    return dydt


def malthus_harvesting(
    x: float, y: np.ndarray, r: float = 0.5, h: float = 0.1, **kwargs: Any
) -> np.ndarray:
    """y' = r·y - h — Malthus with constant harvesting.

    Args:
        x: Independent variable.
        y: State vector ``[y]``.
        r: Growth rate.
        h: Harvesting rate.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = r * y[0] - h
    return dydt


def relu_activation(x: float, y: np.ndarray, k: float = 1.0, **kwargs: Any) -> np.ndarray:
    """y' = k·max(0, y) — ReLU-like activation (linear for y>0).

    Args:
        x: Independent variable.
        y: State vector ``[y]``.
        k: Slope for y > 0.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = k * np.maximum(0, y[0])
    return dydt


def tanh_decay(x: float, y: np.ndarray, k: float = 1.0, **kwargs: Any) -> np.ndarray:
    """y' = -k·tanh(y) — Tanh decay (smooth).

    Args:
        x: Independent variable.
        y: State vector ``[y]``.
        k: Decay rate.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(1)
    dydt[0] = -k * np.tanh(y[0])
    return dydt


# --- 20+ additional vector ODEs ---


def halvorsen_attractor(x: float, y: np.ndarray, a: float = 1.89, **kwargs: Any) -> np.ndarray:
    """Halvorsen attractor: chaotic 3D.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        a: Parameter.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = -a * y[0] - 4 * y[1] - 4 * y[2] - y[1] ** 2
    dydt[1] = -a * y[1] - 4 * y[2] - 4 * y[0] - y[2] ** 2
    dydt[2] = -a * y[2] - 4 * y[0] - 4 * y[1] - y[0] ** 2
    return dydt


def dadras_attractor(
    x: float,
    y: np.ndarray,
    a: float = 3.0,
    b: float = 2.7,
    c: float = 1.7,
    d: float = 2.0,
    e: float = 9.0,
    **kwargs: Any,
) -> np.ndarray:
    """Dadras attractor: chaotic 3D.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        a: Parameter a.
        b: Parameter b.
        c: Parameter c.
        d: Parameter d.
        e: Parameter e.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = y[1] - a * y[0] + b * y[1] * y[2]
    dydt[1] = c * y[1] - y[0] * y[2] + y[2]
    dydt[2] = d * y[0] * y[1] - e * y[2]
    return dydt


def sprott_s(x: float, y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Sprott S system: chaotic 3D."""
    dydt = np.empty(3)
    dydt[0] = y[1] * y[2]
    dydt[1] = y[0] - y[1]
    dydt[2] = 1 - y[0] * y[1]
    return dydt


def sprott_a(x: float, y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Sprott A system: chaotic 3D.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = y[1]
    dydt[1] = -y[0] + y[1] * y[2]
    dydt[2] = 1 - y[1] ** 2
    return dydt


def three_species_food_chain(
    x: float,
    y: np.ndarray,
    r: float = 1.0,
    a1: float = 2.0,
    a2: float = 2.0,
    b1: float = 0.1,
    b2: float = 0.1,
    d1: float = 0.4,
    d2: float = 0.1,
    **kwargs: Any,
) -> np.ndarray:
    """Rosenzweig-MacArthur 3-species food chain. f₀=prey, f₁=pred1, f₂=pred2."""
    dydt = np.empty(3)
    dydt[0] = r * y[0] * (1 - y[0]) - a1 * y[0] * y[1] / (1 + b1 * y[0])
    dydt[1] = a1 * y[0] * y[1] / (1 + b1 * y[0]) - d1 * y[1] - a2 * y[1] * y[2] / (1 + b2 * y[1])
    dydt[2] = a2 * y[1] * y[2] / (1 + b2 * y[1]) - d2 * y[2]
    return dydt


def sirs_epidemic(
    x: float,
    y: np.ndarray,
    beta: float = 0.5,
    gamma: float = 0.2,
    xi: float = 0.1,
    **kwargs: Any,
) -> np.ndarray:
    """SIRS: dS/dt=-βSI+ξR, dI/dt=βSI-γI, dR/dt=γI-ξR.

    Args:
        x: Independent variable (time).
        y: State vector ``[S, I, R]`` (susceptible, infected, recovered).
        beta: Transmission rate.
        gamma: Recovery rate.
        xi: Waning immunity rate.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    s, i, r = y[0], y[1], y[2]
    dydt = np.empty(3)
    dydt[0] = -beta * s * i + xi * r
    dydt[1] = beta * s * i - gamma * i
    dydt[2] = gamma * i - xi * r
    return dydt


def seir_epidemic(
    x: float,
    y: np.ndarray,
    beta: float = 0.5,
    sigma: float = 0.1,
    gamma: float = 0.2,
    **kwargs: Any,
) -> np.ndarray:
    """SEIR: S→E→I→R. f₀=S, f₁=E, f₂=I, f₃=R."""
    s, e, i, _ = y[0], y[1], y[2], y[3]
    dydt = np.empty(4)
    dydt[0] = -beta * s * i
    dydt[1] = beta * s * i - sigma * e
    dydt[2] = sigma * e - gamma * i
    dydt[3] = gamma * i
    return dydt


def oregonator(
    x: float,
    y: np.ndarray,
    q: float = 0.0008,
    f: float = 1.0,
    eps: float = 0.01,
    **kwargs: Any,
) -> np.ndarray:
    """Oregonator: Belousov-Zhabotinsky reaction.

    Args:
        x: Independent variable (time).
        y: State vector ``[u, v, w]`` (concentrations).
        q: Parameter q.
        f: Parameter f.
        eps: Parameter eps.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    u, v, w = y[0], y[1], y[2]
    dydt = np.empty(3)
    dydt[0] = (u - u * u - f * v * (u - q) / (u + q)) / eps
    dydt[1] = u - v
    dydt[2] = (u - w) / eps
    return dydt


def morris_lecar(
    x: float,
    y: np.ndarray,
    phi: float = 0.04,
    g_ca: float = 1.1,
    g_k: float = 2.0,
    g_l: float = 0.5,
    v_ca: float = 1.0,
    v_k: float = -0.7,
    v_l: float = -0.5,
    v1: float = -0.01,
    v2: float = 0.15,
    v3: float = 0.1,
    v4: float = 0.145,
    I_ext: float = 0.0,
    **kwargs: Any,
) -> np.ndarray:
    """Morris-Lecar neuron model."""
    v, w = y[0], y[1]
    m_inf = 0.5 * (1 + np.tanh((v - v1) / v2))
    w_inf = 0.5 * (1 + np.tanh((v - v3) / v4))
    tau_w = 1 / np.cosh((v - v3) / (2 * v4))
    dydt = np.empty(2)
    dydt[0] = I_ext - g_ca * m_inf * (v - v_ca) - g_k * w * (v - v_k) - g_l * (v - v_l)
    dydt[1] = phi * (w_inf - w) / tau_w
    return dydt


def wilson_cowan(
    x: float,
    y: np.ndarray,
    tau_e: float = 1.0,
    tau_i: float = 1.0,
    w_ee: float = 12.0,
    w_ei: float = 4.0,
    w_ie: float = 13.0,
    w_ii: float = 11.0,
    I_e: float = 1.0,
    I_i: float = 0.0,
    **kwargs: Any,
) -> np.ndarray:
    """Wilson-Cowan: excitatory-inhibitory neural populations.

    Args:
        x: Independent variable (time).
        y: State vector ``[e, i]`` (excitatory, inhibitory).
        tau_e: Excitatory time constant.
        tau_i: Inhibitory time constant.
        w_ee, w_ei, w_ie, w_ii: Connection weights.
        I_e, I_i: External inputs.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    e, i = y[0], y[1]
    s_e = 1 / (1 + np.exp(-e))
    s_i = 1 / (1 + np.exp(-i))
    dydt = np.empty(2)
    dydt[0] = (-e + w_ee * s_e - w_ei * s_i + I_e) / tau_e
    dydt[1] = (-i + w_ie * s_e - w_ii * s_i + I_i) / tau_i
    return dydt


def goodwin_oscillator(
    x: float,
    y: np.ndarray,
    k: float = 1.0,
    b: float = 1.0,
    n: float = 9.0,
    **kwargs: Any,
) -> np.ndarray:
    """Goodwin oscillator: gene regulation."""
    dydt = np.empty(3)
    dydt[0] = 1 / (1 + y[2] ** n) - b * y[0]
    dydt[1] = y[0] - b * y[1]
    dydt[2] = y[1] - b * y[2]
    return dydt


def t_system(
    x: float,
    y: np.ndarray,
    a: float = 2.0,
    b: float = 0.5,
    c: float = 1.0,
    **kwargs: Any,
) -> np.ndarray:
    """T system: chaotic 3D.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        a: Parameter a.
        b: Parameter b.
        c: Parameter c.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = a * (y[1] - y[0])
    dydt[1] = (c - a) * y[0] - a * y[0] * y[2]
    dydt[2] = y[0] * y[1] - b * y[2]
    return dydt


def finance_chaos(
    x: float,
    y: np.ndarray,
    a: float = 0.001,
    b: float = 0.2,
    c: float = 1.1,
    **kwargs: Any,
) -> np.ndarray:
    """Finance system: chaotic."""
    dydt = np.empty(3)
    dydt[0] = y[2] + (y[1] - a) * y[0]
    dydt[1] = 1 - b * y[1] - y[0] ** 2
    dydt[2] = -y[0] - c * y[2]
    return dydt


def arneodo(
    x: float,
    y: np.ndarray,
    a: float = -5.5,
    b: float = 3.5,
    c: float = -1.0,
    **kwargs: Any,
) -> np.ndarray:
    """Arneodo system: chaotic.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        a: Parameter a.
        b: Parameter b.
        c: Parameter c.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = y[1]
    dydt[1] = y[2]
    dydt[2] = -a * y[0] - b * y[1] - c * y[2] + y[0] ** 2
    return dydt


def bouali_attractor(
    x: float,
    y: np.ndarray,
    a: float = 0.3,
    s: float = 1.0,
    **kwargs: Any,
) -> np.ndarray:
    """Bouali attractor: chaotic.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        a: Parameter a.
        s: Parameter s.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = y[0] * (1 - y[1] ** 2) - a * y[2]
    dydt[1] = s * y[2]
    dydt[2] = -y[1] + y[0] * y[1] * y[2]
    return dydt


def nose_hoover(x: float, y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Nose-Hoover: thermostat (Hamiltonian-like).

    Args:
        x: Independent variable.
        y: State vector ``[q, p, ζ]``.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = y[1]
    dydt[1] = y[2] * y[0] - y[1]
    dydt[2] = 1 - y[0] ** 2
    return dydt


def dee_attractor(
    x: float, y: np.ndarray, a: float = 1.0, b: float = 0.1, **kwargs: Any
) -> np.ndarray:
    """Dee attractor: chaotic.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        a: Parameter a.
        b: Parameter b.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = y[1]
    dydt[1] = -y[0] - a * y[2]
    dydt[2] = a * y[1] + b * y[0] ** 2
    return dydt


def four_wing(
    x: float,
    y: np.ndarray,
    a: float = 4.0,
    b: float = 6.0,
    c: float = 10.0,
    **kwargs: Any,
) -> np.ndarray:
    """Four-wing attractor.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        a: Parameter a.
        b: Parameter b.
        c: Parameter c.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = a * y[0] - y[1] * y[2]
    dydt[1] = -b * y[1] + y[0] * y[2]
    dydt[2] = -c * y[2] + y[0] * y[1]
    return dydt


def genesi_attractor(
    x: float,
    y: np.ndarray,
    a: float = 0.44,
    b: float = 1.1,
    c: float = 1.0,
    **kwargs: Any,
) -> np.ndarray:
    """Genesiş attractor: chaotic.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        a: Parameter a.
        b: Parameter b.
        c: Parameter c.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = -y[1]
    dydt[1] = y[0] + a * y[2]
    dydt[2] = b + y[2] * (y[0] - c)
    return dydt


def qi_chaos(
    x: float,
    y: np.ndarray,
    a: float = 14.0,
    b: float = 5.0,
    c: float = 1.0,
    **kwargs: Any,
) -> np.ndarray:
    """Qi system: chaotic."""
    dydt = np.empty(3)
    dydt[0] = a * (y[1] - y[0]) + y[1] * y[2]
    dydt[1] = c * y[0] + y[1] - y[0] * y[2]
    dydt[2] = y[0] * y[1] - b * y[2]
    return dydt


def wang_sun_chaos(
    x: float,
    y: np.ndarray,
    a: float = 10.0,
    b: float = 40.0,
    c: float = 2.5,
    d: float = 5.0,
    **kwargs: Any,
) -> np.ndarray:
    """Wang-Sun chaotic system.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z]``.
        a: Parameter a.
        b: Parameter b.
        c: Parameter c.
        d: Parameter d.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(3)
    dydt[0] = a * (y[1] - y[0])
    dydt[1] = y[0] * y[2] - y[1]
    dydt[2] = b - c * y[0] * y[1] - d * y[2]
    return dydt


def predator_prey_ratio(
    x: float,
    y: np.ndarray,
    r: float = 1.0,
    a: float = 1.0,
    b: float = 0.5,
    e: float = 0.5,
    m: float = 0.2,
    **kwargs: Any,
) -> np.ndarray:
    """Ratio-dependent predator-prey: f₀=prey, f₁=predator.

    Args:
        x: Independent variable (time).
        y: State vector ``[prey, predator]``.
        r: Prey growth rate.
        a: Predation rate.
        b: Ratio parameter.
        e: Conversion efficiency.
        m: Predator mortality.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(2)
    dydt[0] = r * y[0] * (1 - y[0]) - a * y[0] * y[1] / (y[0] + b * y[1])
    dydt[1] = e * a * y[0] * y[1] / (y[0] + b * y[1]) - m * y[1]
    return dydt


def leslie_gower(
    x: float,
    y: np.ndarray,
    r: float = 1.0,
    a: float = 1.0,
    c: float = 0.5,
    d: float = 0.1,
    **kwargs: Any,
) -> np.ndarray:
    """Leslie-Gower predator-prey: modified carrying capacity."""
    dydt = np.empty(2)
    dydt[0] = r * y[0] * (1 - y[0]) - a * y[0] * y[1]
    denom = np.maximum(y[0], 1e-10)
    dydt[1] = y[1] * (c - d * y[1] / denom)
    return dydt


def holling_tanner(
    x: float,
    y: np.ndarray,
    r: float = 1.0,
    a: float = 1.0,
    b: float = 0.5,
    e: float = 0.5,
    m: float = 0.2,
    **kwargs: Any,
) -> np.ndarray:
    """Holling-Tanner: predator-prey with prey-dependent growth.

    Args:
        x: Independent variable (time).
        y: State vector ``[prey, predator]``.
        r: Prey growth rate.
        a: Predation rate.
        b: Half-saturation.
        e: Conversion efficiency.
        m: Predator mortality.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(2)
    dydt[0] = r * y[0] * (1 - y[0]) - a * y[0] * y[1] / (y[0] + b)
    dydt[1] = y[1] * (e * a * y[0] / (y[0] + b) - m)
    return dydt


def rossler_hyperchaos(
    x: float,
    y: np.ndarray,
    a: float = 0.25,
    b: float = 3.0,
    c: float = 0.5,
    d: float = 0.05,
    **kwargs: Any,
) -> np.ndarray:
    """Rössler hyperchaos: 4D.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z, w]``.
        a: Parameter a.
        b: Parameter b.
        c: Parameter c.
        d: Parameter d.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(4)
    dydt[0] = -y[1] - y[2]
    dydt[1] = y[0] + a * y[1] + y[3]
    dydt[2] = b + y[0] * y[2]
    dydt[3] = -c * y[2] + d * y[3]
    return dydt


def hyperchaos_lorenz(
    x: float,
    y: np.ndarray,
    a: float = 10.0,
    b: float = 28.0,
    c: float = 8.0 / 3.0,
    r: float = 0.5,
    **kwargs: Any,
) -> np.ndarray:
    """Lorenz 4D hyperchaos.

    Args:
        x: Independent variable.
        y: State vector ``[x, y, z, w]``.
        a: Parameter a (Prandtl-like).
        b: Parameter b (Rayleigh-like).
        c: Parameter c.
        r: Fourth dimension coupling.
        **kwargs: Ignored.

    Returns:
        dy/dx as 1-D numpy array.
    """
    dydt = np.empty(4)
    dydt[0] = a * (y[1] - y[0]) + y[3]
    dydt[1] = y[0] * (b - y[2]) - y[1]
    dydt[2] = y[0] * y[1] - c * y[2]
    dydt[3] = -y[0] * y[2] + r * y[3]
    return dydt
