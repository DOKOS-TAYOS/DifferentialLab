"""Solver for coupled harmonic oscillators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp

from complex_problems.coupled_oscillators.model import (
    _resolve_k,
    _resolve_mass,
    build_ode_function,
    compute_normal_modes,
)
from config import DEFAULT_SOLVER_METHOD, get_env_from_schema
from utils import SolverFailedError, get_logger

logger = get_logger(__name__)


@dataclass
class CoupledOscillatorsResult:
    """Result from solving coupled oscillators."""

    x: np.ndarray
    y: np.ndarray
    n_oscillators: int
    masses: np.ndarray
    k_coupling: np.ndarray
    M_modes: np.ndarray
    omega_modes: np.ndarray
    has_modes: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def _resolve_state_arrays(
    n: int,
    boundary: str,
    masses: float | list[float] | Callable[[int], float],
    k_coupling: float | list[float] | Callable[[int], float],
) -> tuple[np.ndarray, np.ndarray]:
    masses_arr = np.array([_resolve_mass(masses, i, n) for i in range(n)], dtype=float)
    if np.any(masses_arr <= 0):
        raise ValueError("All masses must be positive.")

    n_springs = n if boundary == "periodic" else n - 1
    k_arr = np.array([_resolve_k(k_coupling, i, n) for i in range(n_springs)], dtype=float)
    if np.any(k_arr < 0):
        raise ValueError("All coupling constants must be non-negative.")
    return masses_arr, k_arr


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
    """Solve the coupled oscillators system."""
    if boundary not in {"fixed", "periodic"}:
        raise ValueError("Boundary must be 'fixed' or 'periodic'.")

    n = int(n_oscillators)
    if n < 2:
        raise ValueError("Number of oscillators must be at least 2.")
    if t_max <= t_min:
        raise ValueError("t_max must be greater than t_min.")

    coupling_types = coupling_types or ["linear"]
    method = method or DEFAULT_SOLVER_METHOD

    if n_points is None:
        n_points = int(get_env_from_schema("SOLVER_NUM_POINTS"))
    if n_points < 2:
        raise ValueError("Resolution points must be at least 2.")

    if y0 is None:
        y0 = [0.0] * (2 * n)
        y0[0] = 1.0
    y0_arr = np.asarray(y0, dtype=float)
    if y0_arr.shape != (2 * n,):
        raise ValueError(
            f"Initial state must have exactly {2 * n} values, got {y0_arr.size}."
        )

    masses_arr, k_arr = _resolve_state_arrays(n, boundary, masses, k_coupling)

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

    t_eval = np.linspace(t_min, t_max, n_points)
    rtol = float(get_env_from_schema("SOLVER_RTOL"))
    atol = float(get_env_from_schema("SOLVER_ATOL"))
    max_step = float(get_env_from_schema("SOLVER_MAX_STEP"))
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
        dense_output=False,
    )

    if not sol.success:
        logger.error("Solver failed: %s", sol.message)
        raise SolverFailedError(f"Solver failed ({method}): {sol.message}")

    # Keep modal analysis in the linear basis by design, even for nonlinear runs.
    has_modes = "linear" in coupling_types
    M_modes = np.eye(n)
    omega_modes = np.ones(n)
    if has_modes:
        M_modes, omega_modes = compute_normal_modes(
            n,
            masses,
            k_coupling,
            boundary,
            k_2nn=k_2nn,
            k_3nn=k_3nn,
            k_4nn=k_4nn,
        )

    metadata = {
        "method": method,
        "n_eval": int(getattr(sol, "nfev", 0)),
        "boundary": boundary,
        "coupling_types": coupling_types,
        "nonlinear_coeff": float(nonlinear_coeff),
        "nonlinear_fput_alpha": float(nonlinear_fput_alpha),
        "nonlinear_quartic": float(nonlinear_quartic),
        "nonlinear_quintic": float(nonlinear_quintic),
        "k_2nn": float(k_2nn),
        "k_3nn": float(k_3nn),
        "k_4nn": float(k_4nn),
        "external_amplitude": float(external_amplitude),
        "external_frequency": float(external_frequency),
        "rtol": rtol,
        "atol": atol,
        "max_step": max_step,
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

