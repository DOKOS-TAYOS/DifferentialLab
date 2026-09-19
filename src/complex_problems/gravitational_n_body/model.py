"""Validated initial states and numerical kernels for Newtonian N-body gravity."""
# ruff: noqa: E501

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class NBodyState:
    """Finite initial state for a 2D or 3D point-mass system."""

    masses: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    dimension: int
    name: str = "Custom"
    recommended_t_max: float = 10.0
    recommended_n_points: int = 500
    recommended_epsilon: float = 0.0


def validate_n_body_state(
    masses: np.ndarray | list[float],
    positions: np.ndarray | list[list[float]],
    velocities: np.ndarray | list[list[float]],
    *,
    dimension: int | None = None,
    gravitational_constant: float = 1.0,
    epsilon: float = 0.0,
    min_bodies: int = 2,
    max_bodies: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Validate and copy an N-body state, rejecting singular point-gravity starts."""
    masses_array = np.asarray(masses, dtype=float).reshape(-1)
    positions_array = np.asarray(positions, dtype=float)
    velocities_array = np.asarray(velocities, dtype=float)
    if dimension is None:
        if positions_array.ndim != 2:
            raise ValueError("Positions must be a two-dimensional N by dimension array.")
        dimension = int(positions_array.shape[1])
    if dimension not in (2, 3):
        raise ValueError("Dimension must be 2 or 3.")
    n_bodies = masses_array.size
    if not min_bodies <= n_bodies <= max_bodies:
        raise ValueError(f"Number of bodies must be between {min_bodies} and {max_bodies}.")
    expected_shape = (n_bodies, dimension)
    if positions_array.shape != expected_shape or velocities_array.shape != expected_shape:
        raise ValueError(
            f"Positions and velocities must both have shape {expected_shape}; "
            f"got {positions_array.shape} and {velocities_array.shape}."
        )
    if (
        not np.all(np.isfinite(masses_array))
        or not np.all(np.isfinite(positions_array))
        or not np.all(np.isfinite(velocities_array))
    ):
        raise ValueError("Masses, positions, and velocities must all be finite.")
    if np.any(masses_array <= 0.0):
        raise ValueError("Every body mass must be positive.")
    if not np.isfinite(gravitational_constant) or gravitational_constant <= 0.0:
        raise ValueError("Gravitational constant G must be finite and positive.")
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("Softening epsilon must be finite and non-negative.")
    if epsilon == 0.0:
        displacement = positions_array[np.newaxis, :, :] - positions_array[:, np.newaxis, :]
        distances_squared = np.einsum("ijk,ijk->ij", displacement, displacement)
        upper = distances_squared[np.triu_indices(n_bodies, k=1)]
        if np.any(upper == 0.0):
            raise ValueError(
                "Coincident bodies are singular for epsilon=0. Set a positive softening "
                "epsilon or provide distinct initial positions."
            )
    return masses_array.copy(), positions_array.copy(), velocities_array.copy(), dimension


def gravitational_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    gravitational_constant: float = 1.0,
    epsilon: float = 0.0,
) -> np.ndarray:
    """Return accelerations using the softened Newtonian pair-force convention."""
    displacement = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    distance_squared = np.einsum("ijk,ijk->ij", displacement, displacement)
    softened_squared = distance_squared + epsilon**2
    np.fill_diagonal(softened_squared, np.inf)
    inverse_distance_cubed = softened_squared**-1.5
    acceleration = gravitational_constant * np.sum(
        displacement
        * (masses[np.newaxis, :, np.newaxis] * inverse_distance_cubed[:, :, np.newaxis]),
        axis=1,
    )
    if not np.all(np.isfinite(acceleration)):
        raise ValueError("Gravitational acceleration became non-finite.")
    return acceleration


def gravitational_potential_energy(
    positions: np.ndarray,
    masses: np.ndarray,
    gravitational_constant: float = 1.0,
    epsilon: float = 0.0,
) -> float:
    """Return U = -G sum_{i<j} m_i m_j / sqrt(r_ij^2 + epsilon^2)."""
    displacement = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    distance_squared = np.einsum("ijk,ijk->ij", displacement, displacement)
    i, j = np.triu_indices(masses.size, k=1)
    distances = np.sqrt(distance_squared[i, j] + epsilon**2)
    if np.any(distances == 0.0) or not np.all(np.isfinite(distances)):
        raise ValueError("Potential energy is undefined for coincident unsoftened bodies.")
    return float(-gravitational_constant * np.sum(masses[i] * masses[j] / distances))


def pair_distances(positions: np.ndarray) -> np.ndarray:
    """Return one distance per unordered pair for a single position array."""
    displacement = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    distance_squared = np.einsum("ijk,ijk->ij", displacement, displacement)
    return np.sqrt(distance_squared[np.triu_indices(positions.shape[0], k=1)])


def figure_eight_state() -> NBodyState:
    """Return the normalized equal-mass Chenciner--Montgomery benchmark state."""
    return NBodyState(
        masses=np.ones(3),
        positions=np.array([[-0.97000436, 0.24308753], [0.97000436, -0.24308753], [0.0, 0.0]]),
        velocities=np.array(
            [
                [0.4662036850, 0.4323657300],
                [0.4662036850, 0.4323657300],
                [-0.9324073700, -0.8647314600],
            ]
        ),
        dimension=2,
        name="Figure-eight equal-mass orbit",
        recommended_t_max=6.3259,
        recommended_n_points=600,
    )


def lagrange_equilateral_state(
    *, mass: float = 1.0, radius: float = 1.0, gravitational_constant: float = 1.0
) -> NBodyState:
    """Generate the rigidly rotating equal-mass Lagrange equilateral solution."""
    if not all(
        np.isfinite(value) and value > 0.0 for value in (mass, radius, gravitational_constant)
    ):
        raise ValueError("Lagrange mass, radius, and G must be finite and positive.")
    angles = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    positions = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    angular_speed = np.sqrt(gravitational_constant * mass / (np.sqrt(3.0) * radius**3))
    velocities = angular_speed * np.column_stack((-positions[:, 1], positions[:, 0]))
    period = 2.0 * np.pi / angular_speed
    return NBodyState(
        masses=np.full(3, mass),
        positions=positions,
        velocities=velocities,
        dimension=2,
        name="Lagrange equilateral orbit",
        recommended_t_max=float(period),
        recommended_n_points=600,
    )


def pythagorean_state() -> NBodyState:
    """Return the classical zero-velocity 3-4-5 Pythagorean configuration."""
    return NBodyState(
        masses=np.array([3.0, 4.0, 5.0]),
        positions=np.array([[1.0, 3.0], [-2.0, -1.0], [1.0, -1.0]]),
        velocities=np.zeros((3, 2)),
        dimension=2,
        name="Pythagorean three-body problem",
        recommended_t_max=10.0,
        recommended_n_points=800,
        recommended_epsilon=0.02,
    )


def random_bound_cluster_state(
    *,
    n_bodies: int = 8,
    dimension: int = 2,
    seed: int = 1234,
    position_scale: float = 1.0,
    mass_min: float = 0.8,
    mass_max: float = 1.2,
    target_virial_ratio: float = 1.0,
    gravitational_constant: float = 1.0,
    epsilon: float = 0.02,
) -> NBodyState:
    """Create a deterministic, COM-frame random cluster with requested initial virial ratio."""
    if dimension not in (2, 3) or not 2 <= n_bodies <= 100:
        raise ValueError("Random clusters require 2--100 bodies in 2D or 3D.")
    if position_scale <= 0.0 or mass_min <= 0.0 or mass_max < mass_min or target_virial_ratio < 0.0:
        raise ValueError(
            "Cluster scales and masses must be positive; virial ratio must be non-negative."
        )
    rng = np.random.default_rng(seed)
    masses = rng.uniform(mass_min, mass_max, n_bodies)
    positions = rng.normal(size=(n_bodies, dimension)) * position_scale
    positions -= np.average(positions, axis=0, weights=masses)
    # Deterministic small perturbations resolve the astronomically unlikely duplicate draw.
    for index in range(1, n_bodies):
        if np.any(
            np.all(np.isclose(positions[index], positions[:index], rtol=0.0, atol=1e-12), axis=1)
        ):
            positions[index, 0] += (index + 1) * 1e-8 * position_scale
    velocities = rng.normal(size=(n_bodies, dimension))
    velocities -= np.average(velocities, axis=0, weights=masses)
    potential = gravitational_potential_energy(positions, masses, gravitational_constant, epsilon)
    kinetic_unscaled = 0.5 * float(np.sum(masses[:, np.newaxis] * velocities**2))
    if kinetic_unscaled > 0.0 and target_virial_ratio > 0.0:
        velocities *= np.sqrt(target_virial_ratio * abs(potential) / (2.0 * kinetic_unscaled))
    else:
        velocities.fill(0.0)
    return NBodyState(
        masses, positions, velocities, dimension, "Random bound cluster", 10.0, 500, epsilon
    )


def ring_state(
    *,
    n_bodies: int = 8,
    dimension: int = 2,
    radius: float = 1.0,
    mass: float = 1.0,
    gravitational_constant: float = 1.0,
    epsilon: float = 0.0,
) -> NBodyState:
    """Generate a planar equal-mass ring using its actual discrete radial acceleration."""
    if dimension not in (2, 3) or not 2 <= n_bodies <= 100:
        raise ValueError("Rings require 2--100 bodies in 2D or 3D.")
    if not all(
        np.isfinite(value) and value > 0.0 for value in (radius, mass, gravitational_constant)
    ):
        raise ValueError("Ring radius, mass, and G must be finite and positive.")
    angles = 2.0 * np.pi * np.arange(n_bodies) / n_bodies
    planar = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    positions = planar if dimension == 2 else np.column_stack((planar, np.zeros(n_bodies)))
    masses = np.full(n_bodies, mass)
    acceleration = gravitational_accelerations(positions, masses, gravitational_constant, epsilon)
    radial_unit = positions[0] / radius
    inward_acceleration = max(0.0, -float(np.dot(acceleration[0], radial_unit)))
    speed = np.sqrt(inward_acceleration * radius)
    planar_velocity = speed * np.column_stack((-np.sin(angles), np.cos(angles)))
    velocities = (
        planar_velocity
        if dimension == 2
        else np.column_stack((planar_velocity, np.zeros(n_bodies)))
    )
    return NBodyState(masses, positions, velocities, dimension, "Rotating ring", 10.0, 500, epsilon)
