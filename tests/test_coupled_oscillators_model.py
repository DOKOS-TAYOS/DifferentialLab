"""Tests for coupled oscillators model and solver."""

from __future__ import annotations

import numpy as np
import pytest

from complex_problems.coupled_oscillators.model import (
    build_ode_function,
    compute_normal_modes,
)
from complex_problems.coupled_oscillators.solver import solve_coupled_oscillators


def test_build_ode_linear_nonuniform_fixed_matches_spring_balance() -> None:
    n = 4
    x = np.array([0.3, -0.2, 0.5, 0.1], dtype=float)
    y = np.concatenate([x, np.zeros(n)])
    k = np.array([1.0, 2.0, 4.0], dtype=float)

    ode = build_ode_function(
        n_oscillators=n,
        masses=1.0,
        k_coupling=k.tolist(),
        boundary="fixed",
        coupling_types=["linear"],
    )
    dydt = ode(0.0, y)
    acc = dydt[n:]

    expected = np.array(
        [
            k[0] * (x[1] - x[0]) - k[0] * (x[0] - 0.0),
            k[1] * (x[2] - x[1]) - k[0] * (x[1] - x[0]),
            k[2] * (x[3] - x[2]) - k[1] * (x[2] - x[1]),
            k[2] * (0.0 - x[3]) - k[2] * (x[3] - x[2]),
        ],
        dtype=float,
    )
    np.testing.assert_allclose(acc, expected, rtol=1e-12, atol=1e-12)


def test_build_ode_linear_nonuniform_periodic_matches_spring_balance() -> None:
    n = 4
    x = np.array([0.2, -0.1, 0.4, -0.3], dtype=float)
    y = np.concatenate([x, np.zeros(n)])
    k = np.array([1.0, 2.0, 4.0, 3.0], dtype=float)

    ode = build_ode_function(
        n_oscillators=n,
        masses=1.0,
        k_coupling=k.tolist(),
        boundary="periodic",
        coupling_types=["linear"],
    )
    dydt = ode(0.0, y)
    acc = dydt[n:]

    expected = np.zeros(n)
    for i in range(n):
        expected[i] = (
            k[i] * (x[(i + 1) % n] - x[i])
            - k[(i - 1) % n] * (x[i] - x[(i - 1) % n])
        )
    np.testing.assert_allclose(acc, expected, rtol=1e-12, atol=1e-12)


def test_build_ode_beta_nonlinearity_uses_difference_of_cubes() -> None:
    n = 3
    x = np.array([0.3, -0.2, 0.1], dtype=float)
    y = np.concatenate([x, np.zeros(n)])
    beta = 2.5

    ode = build_ode_function(
        n_oscillators=n,
        masses=1.0,
        k_coupling=0.0,
        boundary="periodic",
        coupling_types=["nonlinear"],
        nonlinear_coeff=beta,
    )
    dydt = ode(0.0, y)
    acc = dydt[n:]

    expected = np.zeros(n)
    for i in range(n):
        delta_right = x[(i + 1) % n] - x[i]
        delta_left = x[i] - x[(i - 1) % n]
        expected[i] = beta * (delta_right**3 - delta_left**3)
    np.testing.assert_allclose(acc, expected, rtol=1e-12, atol=1e-12)


def test_compute_normal_modes_uniform_fixed_matches_analytic_frequencies() -> None:
    n = 6
    m = 1.5
    k = 2.0
    modes, omega = compute_normal_modes(
        n_oscillators=n,
        masses=m,
        k_coupling=k,
        boundary="fixed",
    )
    expected = np.array(
        [2.0 * np.sqrt(k / m) * np.sin((idx + 1) * np.pi / (2.0 * (n + 1))) for idx in range(n)]
    )

    np.testing.assert_allclose(omega, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(m * modes.T @ modes, np.eye(n), atol=1e-10)


def test_solver_rejects_non_positive_mass() -> None:
    with pytest.raises(ValueError):
        solve_coupled_oscillators(
            n_oscillators=3,
            masses=[1.0, -1.0, 1.0],
            k_coupling=1.0,
            t_max=1.0,
            n_points=20,
        )

