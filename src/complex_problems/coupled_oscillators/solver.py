"""Solver for coupled harmonic oscillators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp

from complex_problems.coupled_oscillators.model import (
    build_ode_function,
    compute_normal_modes,
)
from config import get_default_solver_method, get_env_from_schema
from utils import SolverFailedError, get_logger

logger = get_logger(__name__)


@dataclass
class CoupledOscillatorsResult:
    """Result from solving coupled oscillators.

    Attributes:
        x: Time values (1D).
        y: State array shape (2*N, n_points): [x_0..x_{N-1}, v_0..v_{N-1}].
        n_oscillators: Number of oscillators.
        masses: Mass array (n_oscillators,).
        k_coupling: Coupling array (n_oscillators-1 or n_oscillators for periodic).
        M_modes: Mode matrix (n, n), columns are eigenvectors.
        omega_modes: Angular frequencies of modes (1D).
        has_modes: True if normal modes were computed (linear only).
        metadata: Additional solver metadata.
    """

    x: np.ndarray
    y: np.ndarray
    n_oscillators: int
    masses: np.ndarray
    k_coupling: np.ndarray
    M_modes: np.ndarray
    omega_modes: np.ndarray
    has_modes: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def solve_coupled_oscillators(
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
    t_min: float = 0.0,
    t_max: float = 30.0,
    n_points: int | None = None,
    y0: list[float] | None = None,
    method: str | None = None,
) -> CoupledOscillatorsResult:
    """Solve the coupled oscillators system.

    Args:
        n_oscillators: Number of oscillators.
        masses: Mass specification.
        k_coupling: Coupling specification.
        boundary: "fixed" or "periodic".
        coupling_types: List of coupling types.
        nonlinear_coeff: Cubic nonlinear coupling coefficient (FPUT-β).
        nonlinear_fput_alpha: FPUT-α quadratic nonlinear coefficient.
        nonlinear_quartic: Quartic nonlinear coefficient.
        nonlinear_quintic: Quintic nonlinear coefficient.
        k_2nn: 2nd-neighbor linear coupling (0 = disabled).
        k_3nn: 3rd-neighbor linear coupling (0 = disabled).
        k_4nn: 4th-neighbor linear coupling (0 = disabled).
        external_amplitude: External force amplitude.
        external_frequency: External force frequency.
        t_min: Start time.
        t_max: End time.
        n_points: Number of output points (default from env).
        y0: Initial conditions [x_0, ..., x_{N-1}, v_0, ..., v_{N-1}].
        method: Solver method.

    Returns:
        CoupledOscillatorsResult with solution and metadata.

    Raises:
        SolverFailedError: If integration fails.
    """
    coupling_types = coupling_types or ["linear"]
    n = n_oscillators

    if n_points is None:
        n_points = int(get_env_from_schema("SOLVER_NUM_POINTS"))
    t_eval = np.linspace(t_min, t_max, n_points)

    if y0 is None:
        y0 = [0.0] * (2 * n)
        y0[0] = 1.0  # Excite first oscillator
    y0_arr = np.array(y0, dtype=float)

    ode_func = build_ode_function(
        n_oscillators=n,
        masses=masses,
        k_coupling=k_coupling,
        boundary=boundary,
        coupling_types=coupling_types,
        nonlinear_coeff=nonlinear_coeff,
        nonlinear_fput_alpha=nonlinear_fput_alpha,
        nonlinear_quartic=nonlinear_quartic,
        nonlinear_quintic=nonlinear_quintic,
        k_2nn=k_2nn,
        k_3nn=k_3nn,
        k_4nn=k_4nn,
        external_amplitude=external_amplitude,
        external_frequency=external_frequency,
    )

    method = method or get_default_solver_method()
    rtol = get_env_from_schema("SOLVER_RTOL")
    atol = get_env_from_schema("SOLVER_ATOL")
    max_step = get_env_from_schema("SOLVER_MAX_STEP")
    effective_max_step = np.inf if max_step <= 0 else max_step

    logger.info(
        "Solving coupled oscillators: n=%d, t=[%g, %g], method=%s",
        n,
        t_min,
        t_max,
        method,
    )

    sol = solve_ivp(
        fun=ode_func,
        t_span=(t_min, t_max),
        y0=y0_arr,
        method=method,
        t_eval=t_eval,
        max_step=effective_max_step,
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )

    if not sol.success:
        logger.error("Solver failed: %s", sol.message)
        raise SolverFailedError(f"Solver failed ({method}): {sol.message}")

    # Resolve masses and k to arrays for result
    from complex_problems.coupled_oscillators.model import _resolve_k, _resolve_mass

    masses_arr = np.array([_resolve_mass(masses, i, n) for i in range(n)])
    n_springs = n if boundary == "periodic" else n - 1
    k_arr = np.array([_resolve_k(k_coupling, i, n) for i in range(n_springs)])

    # Compute linear modes whenever linear coupling is present; use as projection basis
    # even for nonlinear/external systems (energy per mode not conserved in that case)
    has_modes = "linear" in coupling_types
    M_modes = np.eye(n)
    omega_modes = np.ones(n)
    if has_modes:
        M_modes, omega_modes = compute_normal_modes(
            n, masses, k_coupling, boundary,
            k_2nn=k_2nn, k_3nn=k_3nn, k_4nn=k_4nn,
        )

    metadata = {
        "method": method,
        "n_eval": getattr(sol, "nfev", 0),
        "boundary": boundary,
        "coupling_types": coupling_types,
        "nonlinear_coeff": nonlinear_coeff,
        "nonlinear_fput_alpha": nonlinear_fput_alpha,
        "nonlinear_quartic": nonlinear_quartic,
        "nonlinear_quintic": nonlinear_quintic,
        "k_2nn": k_2nn,
        "k_3nn": k_3nn,
        "k_4nn": k_4nn,
    }

    return CoupledOscillatorsResult(
        x=sol.t,
        y=sol.y,
        n_oscillators=n,
        masses=masses_arr,
        k_coupling=k_arr,
        M_modes=M_modes,
        omega_modes=omega_modes,
        has_modes=has_modes,
        metadata=metadata,
    )
