"""solve_ivp solver and diagnostics for classical gravitational N-body dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from complex_problems.gravitational_n_body.model import (
    gravitational_accelerations,
    gravitational_potential_energy,
    validate_n_body_state,
)
from config import get_env_from_schema
from utils import SolverFailedError


@dataclass(slots=True)
class NBodyResult:
    """Sampled N-body solution and derived conservation diagnostics."""

    t: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    dimension: int
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray
    total_energy: np.ndarray
    linear_momentum: np.ndarray
    angular_momentum: np.ndarray
    center_of_mass: np.ndarray
    center_of_mass_velocity: np.ndarray
    moment_of_inertia: np.ndarray
    virial_ratio: np.ndarray
    min_pair_distance: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    magnitudes: dict[str, float] = field(default_factory=dict)


def _angular_momentum(
    positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray
) -> np.ndarray:
    """Return 3-vector angular momentum for each sampled time, including planar data."""
    if positions.shape[-1] == 2:
        cross_z = positions[..., 0] * velocities[..., 1] - positions[..., 1] * velocities[..., 0]
        angular = np.zeros((positions.shape[0], 3))
        angular[:, 2] = np.sum(cross_z * masses[np.newaxis, :], axis=1)
        return angular
    return np.sum(np.cross(positions, velocities) * masses[np.newaxis, :, np.newaxis], axis=1)


def _safe_relative_drift(values: np.ndarray) -> float:
    """Return a finite relative drift, falling back to absolute drift at zero reference."""
    reference = np.asarray(values[0], dtype=float)
    drift = np.asarray(values, dtype=float) - reference
    scale = float(np.linalg.norm(reference))
    max_drift = float(np.max(np.linalg.norm(drift.reshape(drift.shape[0], -1), axis=1)))
    return max_drift / scale if scale > np.finfo(float).eps else max_drift


def _compute_diagnostics(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    gravitational_constant: float,
    epsilon: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compute scalar/vector histories without retaining pair matrices across time."""
    kinetic = 0.5 * np.sum(masses[np.newaxis, :, np.newaxis] * velocities**2, axis=(1, 2))
    potential = np.asarray(
        [
            gravitational_potential_energy(frame, masses, gravitational_constant, epsilon)
            for frame in positions
        ]
    )
    total = kinetic + potential
    momentum = np.sum(masses[np.newaxis, :, np.newaxis] * velocities, axis=1)
    total_mass = float(np.sum(masses))
    center_of_mass = np.sum(masses[np.newaxis, :, np.newaxis] * positions, axis=1) / total_mass
    relative_positions = positions - center_of_mass[:, np.newaxis, :]
    moment_of_inertia = np.sum(
        masses[np.newaxis, :] * np.sum(relative_positions**2, axis=2), axis=1
    )
    angular = _angular_momentum(positions, velocities, masses)
    virial = np.full_like(kinetic, np.nan)
    nonzero_potential = np.abs(potential) > np.finfo(float).eps
    virial[nonzero_potential] = (
        2.0 * kinetic[nonzero_potential] / np.abs(potential[nonzero_potential])
    )
    min_distances = np.empty(positions.shape[0])
    for index, frame in enumerate(positions):
        displacement = frame[np.newaxis, :, :] - frame[:, np.newaxis, :]
        squared = np.einsum("ijk,ijk->ij", displacement, displacement)
        min_distances[index] = np.sqrt(np.min(squared[np.triu_indices(frame.shape[0], k=1)]))
    return (
        kinetic,
        potential,
        total,
        momentum,
        center_of_mass,
        angular,
        moment_of_inertia,
        virial,
        min_distances,
    )


def solve_n_body(
    *,
    masses: np.ndarray | list[float],
    positions: np.ndarray | list[list[float]],
    velocities: np.ndarray | list[list[float]],
    dimension: int | None = None,
    gravitational_constant: float = 1.0,
    epsilon: float = 0.0,
    t_min: float = 0.0,
    t_max: float = 10.0,
    n_points: int | None = None,
    method: str = "DOP853",
) -> NBodyResult:
    """Integrate a finite 2D/3D Newtonian point-mass state with solve_ivp.

    Values use one self-consistent user-selected unit system.  A positive epsilon
    selects Plummer-softened gravity; epsilon=0 is exact point gravity.
    """
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        raise ValueError("Time interval must be finite and have t_max greater than t_min.")
    if n_points is None:
        n_points = int(get_env_from_schema("SOLVER_NUM_POINTS"))
    if n_points < 2:
        raise ValueError("Output sample count must be at least 2.")
    masses_array, positions_array, velocities_array, actual_dimension = validate_n_body_state(
        masses,
        positions,
        velocities,
        dimension=dimension,
        gravitational_constant=gravitational_constant,
        epsilon=epsilon,
    )
    n_bodies = masses_array.size
    initial_state = np.concatenate((positions_array.ravel(), velocities_array.ravel()))

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("N-body state became non-finite during integration.")
        current_positions = state[: n_bodies * actual_dimension].reshape(n_bodies, actual_dimension)
        current_velocities = state[n_bodies * actual_dimension :].reshape(
            n_bodies, actual_dimension
        )
        acceleration = gravitational_accelerations(
            current_positions, masses_array, gravitational_constant, epsilon
        )
        derivative = np.concatenate((current_velocities.ravel(), acceleration.ravel()))
        if not np.all(np.isfinite(derivative)):
            raise FloatingPointError("N-body derivative became non-finite during integration.")
        return derivative

    max_step = float(get_env_from_schema("SOLVER_MAX_STEP"))
    solution = solve_ivp(
        rhs,
        (t_min, t_max),
        initial_state,
        method=method,
        t_eval=np.linspace(t_min, t_max, n_points),
        rtol=float(get_env_from_schema("SOLVER_RTOL")),
        atol=float(get_env_from_schema("SOLVER_ATOL")),
        max_step=np.inf if max_step <= 0.0 else max_step,
    )
    if not solution.success:
        raise SolverFailedError(f"N-body solver failed ({method}): {solution.message}")
    if not np.all(np.isfinite(solution.y)):
        raise SolverFailedError("N-body solver produced non-finite output.")
    positions_history = solution.y[: n_bodies * actual_dimension].T.reshape(
        -1, n_bodies, actual_dimension
    )
    velocities_history = solution.y[n_bodies * actual_dimension :].T.reshape(
        -1, n_bodies, actual_dimension
    )
    (
        kinetic,
        potential,
        total,
        momentum,
        center_of_mass,
        angular,
        inertia,
        virial,
        min_distances,
    ) = _compute_diagnostics(
        positions_history, velocities_history, masses_array, gravitational_constant, epsilon
    )
    expected_com = center_of_mass[0] + (solution.t - solution.t[0])[:, np.newaxis] * (
        momentum[0] / np.sum(masses_array)
    )
    magnitudes = {
        "max_relative_total_energy_drift": _safe_relative_drift(total),
        "max_linear_momentum_drift": _safe_relative_drift(momentum),
        "max_angular_momentum_drift": _safe_relative_drift(angular),
        "max_com_displacement_from_uniform_motion": float(
            np.max(np.linalg.norm(center_of_mass - expected_com, axis=1))
        ),
        "minimum_pair_separation": float(np.min(min_distances)),
    }
    return NBodyResult(
        t=solution.t,
        positions=positions_history,
        velocities=velocities_history,
        masses=masses_array,
        dimension=actual_dimension,
        kinetic_energy=kinetic,
        potential_energy=potential,
        total_energy=total,
        linear_momentum=momentum,
        angular_momentum=angular,
        center_of_mass=center_of_mass,
        center_of_mass_velocity=momentum[0] / np.sum(masses_array),
        moment_of_inertia=inertia,
        virial_ratio=virial,
        min_pair_distance=min_distances,
        metadata={
            "method": method,
            "gravitational_constant": float(gravitational_constant),
            "epsilon": float(epsilon),
            "n_bodies": int(n_bodies),
            "nfev": int(solution.nfev),
            "rtol": float(get_env_from_schema("SOLVER_RTOL")),
            "atol": float(get_env_from_schema("SOLVER_ATOL")),
        },
        magnitudes=magnitudes,
    )


solve_gravitational_n_body = solve_n_body
