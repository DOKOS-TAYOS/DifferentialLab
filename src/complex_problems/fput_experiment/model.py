"""Physical model and initial states for a fixed-end FPUT chain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

FPUTModel = Literal["alpha", "beta"]


@dataclass(frozen=True, slots=True)
class FPUTPreset:
    """Editable settings supplied by an FPUT study preset."""

    name: str
    n_particles: int
    model: FPUTModel
    coefficient: float
    amplitude: float
    mode: int
    t_end: float
    dt: float
    sample_every: int
    initial_state: str = "single_mode"


FPUT_PRESETS: dict[str, FPUTPreset] = {
    "Interactive alpha recurrence": FPUTPreset(
        "Interactive alpha recurrence", 16, "alpha", 0.25, 1.0, 1, 2200.0, 0.02, 20
    ),
    "Historical alpha recurrence": FPUTPreset(
        "Historical alpha recurrence", 32, "alpha", 0.25, 1.0, 1, 11500.0, 0.01, 100
    ),
    "Historical beta recurrence": FPUTPreset(
        "Historical beta recurrence", 32, "beta", 10.0, 1.0, 2, 600.0, 1.0e-4, 400
    ),
    "Historical alpha superrecurrence": FPUTPreset(
        "Historical alpha superrecurrence", 32, "alpha", 0.5, 1.0, 1, 240000.0, 0.02, 200
    ),
    "Historical phase-space study": FPUTPreset(
        "Historical phase-space study", 3, "alpha", 0.25, 1.0, 1, 15000.0, 0.01, 10
    ),
    "Alpha strain / solitary-wave study": FPUTPreset(
        "Alpha strain / solitary-wave study",
        32,
        "alpha",
        0.25,
        1.0,
        1,
        11500.0,
        0.001,
        500,
        "legacy_kink_pair",
    ),
    "Beta strain / solitary-wave study": FPUTPreset(
        "Beta strain / solitary-wave study", 32, "beta", 50.0, 1.0, 1, 11500.0, 0.001, 500
    ),
    "Custom": FPUTPreset("Custom", 16, "alpha", 0.25, 1.0, 1, 100.0, 0.02, 10),
}


def _validate_chain(n_particles: int, model: FPUTModel, coefficient: float) -> None:
    """Validate parameters shared by every physical-model routine."""
    if int(n_particles) != n_particles or n_particles < 2:
        raise ValueError("FPUT requires at least two moving particles.")
    if model not in {"alpha", "beta"}:
        raise ValueError("model must be 'alpha' or 'beta'.")
    if not np.isfinite(coefficient):
        raise ValueError("The nonlinear coefficient must be finite.")
    if model == "beta" and coefficient < 0.0:
        raise ValueError("beta must be non-negative for the intended bounded FPUT-beta model.")


def bond_strain(displacement: np.ndarray) -> np.ndarray:
    """Return all N+1 fixed-end bond strains x[j+1]-x[j]."""
    x = np.asarray(displacement, dtype=float)
    if x.shape[-1] < 2:
        raise ValueError("Displacement must contain at least two particles.")
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(1, 1)])
    return np.diff(padded, axis=-1)


def fput_force(displacement: np.ndarray, *, model: FPUTModel, coefficient: float) -> np.ndarray:
    """Evaluate the O(N) fixed-end FPUT acceleration without an interaction matrix."""
    x = np.asarray(displacement, dtype=float)
    _validate_chain(x.size, model, coefficient)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("Displacement must be a finite one-dimensional state.")
    strain = bond_strain(x)
    linear = strain[1:] - strain[:-1]
    nonlinear = strain[1:] ** (2 if model == "alpha" else 3) - strain[:-1] ** (
        2 if model == "alpha" else 3
    )
    return linear + coefficient * nonlinear


def energy_components(
    displacement: np.ndarray, velocity: np.ndarray, *, model: FPUTModel, coefficient: float
) -> tuple[float, float, float, float, float]:
    """Return kinetic, harmonic, nonlinear, potential and exact Hamiltonian energy."""
    x = np.asarray(displacement, dtype=float)
    v = np.asarray(velocity, dtype=float)
    _validate_chain(x.size, model, coefficient)
    if (
        x.shape != v.shape
        or x.ndim != 1
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(v))
    ):
        raise ValueError("Displacement and velocity must be finite equal-length vectors.")
    strain = bond_strain(x)
    kinetic = float(0.5 * np.sum(v**2))
    harmonic = float(0.5 * np.sum(strain**2))
    power = 3 if model == "alpha" else 4
    nonlinear = float(coefficient / power * np.sum(strain**power))
    potential = harmonic + nonlinear
    return kinetic, harmonic, nonlinear, potential, kinetic + potential


def normal_mode_basis(n_particles: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the orthonormal fixed-end sine basis Phi and linear frequencies."""
    _validate_chain(n_particles, "alpha", 0.0)
    indices = np.arange(1, n_particles + 1, dtype=float)
    basis = np.sqrt(2.0 / (n_particles + 1.0)) * np.sin(
        np.pi * np.outer(indices, indices) / (n_particles + 1.0)
    )
    omega = 2.0 * np.sin(np.pi * indices / (2.0 * (n_particles + 1.0)))
    return basis, omega


def modal_coordinates(
    displacement: np.ndarray, velocity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transform particle displacement and velocity to orthonormal modal coordinates."""
    x = np.asarray(displacement, dtype=float)
    v = np.asarray(velocity, dtype=float)
    if x.shape != v.shape or x.ndim != 1:
        raise ValueError("Particle displacement and velocity must be equal-length vectors.")
    basis, _ = normal_mode_basis(x.size)
    return basis.T @ x, basis.T @ v


def particle_coordinates(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transform orthonormal modal coordinates back to particle coordinates."""
    q_arr = np.asarray(q, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    if q_arr.shape != p_arr.shape or q_arr.ndim != 1:
        raise ValueError("Modal Q and P must be equal-length vectors.")
    basis, _ = normal_mode_basis(q_arr.size)
    return basis @ q_arr, basis @ p_arr


def single_mode_initial_state(
    n_particles: int, amplitude: float, mode: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create the historical physical-displacement single-mode initial state."""
    if not np.isfinite(amplitude) or not 1 <= mode <= n_particles:
        raise ValueError("Amplitude must be finite and mode must be between 1 and N.")
    j = np.arange(1, n_particles + 1, dtype=float)
    return amplitude * np.sin(np.pi * mode * j / (n_particles + 1.0)), np.zeros(n_particles)


def two_adjacent_modes_initial_state(
    n_particles: int, amplitude: float, mode: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create the historical equal-amplitude adjacent-mode initial state."""
    if not np.isfinite(amplitude) or not 1 <= mode < n_particles:
        raise ValueError("Amplitude must be finite and mode must be between 1 and N-1.")
    j = np.arange(1, n_particles + 1, dtype=float)
    x = (
        0.5
        * amplitude
        * (
            np.sin(np.pi * mode * j / (n_particles + 1.0))
            + np.sin(np.pi * (mode + 1) * j / (n_particles + 1.0))
        )
    )
    return x, np.zeros(n_particles)


def legacy_kink_pair_initial_state(
    n_particles: int,
    amplitude: float,
    alpha: float,
    *,
    width: float = 0.5,
    first_center: float = 6.0,
    second_center: float = 26.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Port the historical CC0 alpha-FPUT kink-pair initial condition.

    This is a legacy study state, not asserted to be an exact discrete FPUT solution.
    """
    _validate_chain(n_particles, "alpha", alpha)
    if (
        alpha == 0.0
        or width <= 0.0
        or not all(np.isfinite(value) for value in (amplitude, width, first_center, second_center))
    ):
        raise ValueError("Legacy kink-pair needs non-zero alpha and finite positive width.")
    i = np.arange(1, n_particles + 1, dtype=float)
    a = width
    first = i - first_center
    second = i - second_center
    # Algebraically identical to the original Fortran formulas, evaluated for moving particles.
    x = -np.log((1.0 + np.exp(2.0 * a * (first - 1.0))) / (1.0 + np.exp(2.0 * a * first)))
    x += np.log((1.0 + np.exp(2.0 * a * (second - 1.0))) / (1.0 + np.exp(2.0 * a * second)))
    velocity = 1.0 / (np.cosh(a * first) * (np.exp(-a * first) + np.exp(a * first - 2.0 * a)))
    velocity -= 1.0 / (np.cosh(a * second) * (np.exp(-a * second) + np.exp(a * second - 2.0 * a)))
    return x * (0.5 * amplitude / alpha), velocity * (np.sinh(a) ** 2 * np.exp(-a) / alpha)


def alpha_nonconvex_bonds(displacement: np.ndarray, alpha: float) -> np.ndarray:
    """Return a mask for initial alpha bonds with non-positive local curvature."""
    return 1.0 + 2.0 * alpha * bond_strain(np.asarray(displacement, dtype=float)) <= 0.0
