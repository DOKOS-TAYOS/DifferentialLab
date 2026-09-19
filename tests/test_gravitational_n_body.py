"""Focused numerical and plotting tests for gravitational N-body dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from complex_problems.gravitational_n_body.model import (
    figure_eight_state,
    gravitational_accelerations,
    gravitational_potential_energy,
    lagrange_equilateral_state,
    pythagorean_state,
    random_bound_cluster_state,
    ring_state,
    validate_n_body_state,
)
from complex_problems.gravitational_n_body.result_dialog import (
    create_orbit_animation_figure,
    orbit_animation_payload,
    selected_pair_distance,
)
from complex_problems.gravitational_n_body.solver import solve_n_body
from complex_problems.gravitational_n_body.ui import parse_n_body_state_text


def test_two_body_force_is_equal_and_opposite_and_translation_invariant() -> None:
    masses = np.array([2.0, 3.0])
    positions = np.array([[-1.0, 0.5], [2.0, -0.5]])
    acceleration = gravitational_accelerations(positions, masses)
    assert np.allclose(masses[:, None] * acceleration, -(masses[::-1, None] * acceleration[::-1]))
    assert np.allclose(acceleration, gravitational_accelerations(positions + [11.0, -4.0], masses))
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert np.allclose(
        gravitational_accelerations(positions @ rotation.T, masses), acceleration @ rotation.T
    )


def test_softening_is_finite_at_coincidence_and_unsoftened_state_rejects_it() -> None:
    positions = np.zeros((2, 2))
    assert np.all(np.isfinite(gravitational_accelerations(positions, np.ones(2), epsilon=0.1)))
    with pytest.raises(ValueError, match="Coincident"):
        validate_n_body_state([1.0, 1.0], positions, positions, epsilon=0.0)
    with pytest.raises(ValueError, match="positive"):
        validate_n_body_state([1.0, 0.0], [[0.0, 0.0], [1.0, 0.0]], positions)
    with pytest.raises(ValueError, match="finite"):
        validate_n_body_state([1.0, np.nan], [[0.0, 0.0], [1.0, 0.0]], positions)
    with pytest.raises(ValueError, match="shape"):
        validate_n_body_state([1.0, 1.0], [[0.0, 0.0]], positions)


def test_softened_potential_matches_finite_difference_force() -> None:
    masses = np.array([2.0, 3.0])
    positions = np.array([[0.0, 0.0], [1.2, -0.4]])
    epsilon = 0.2
    step = 1e-6
    shifted = positions.copy()
    shifted[0, 0] += step
    force_x = (
        -(
            gravitational_potential_energy(shifted, masses, epsilon=epsilon)
            - gravitational_potential_energy(positions, masses, epsilon=epsilon)
        )
        / step
    )
    acceleration = gravitational_accelerations(positions, masses, epsilon=epsilon)
    assert force_x == pytest.approx(masses[0] * acceleration[0, 0], rel=1e-5)


def test_angular_momentum_and_com_diagnostics_in_2d_and_3d() -> None:
    result = solve_n_body(
        masses=[1.0, 1.0],
        positions=[[-0.5, 0.0], [0.5, 0.0]],
        velocities=[[0.0, -0.5], [0.0, 0.5]],
        t_max=0.2,
        n_points=24,
    )
    assert result.angular_momentum.shape == (24, 3)
    assert np.allclose(result.center_of_mass, 0.0, atol=1e-8)
    assert result.angular_momentum[0, 2] == pytest.approx(0.5)
    three_d = solve_n_body(
        masses=[1.0, 1.0],
        positions=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
        velocities=[[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]],
        t_max=0.1,
        n_points=8,
    )
    assert three_d.angular_momentum.shape == (8, 3)


def test_curated_presets_and_generators_solve_finitely() -> None:
    for state, duration in (
        (figure_eight_state(), 1.0),
        (lagrange_equilateral_state(), 1.0),
        (pythagorean_state(), 0.2),
    ):
        result = solve_n_body(
            masses=state.masses,
            positions=state.positions,
            velocities=state.velocities,
            epsilon=state.recommended_epsilon,
            t_max=duration,
            n_points=48,
        )
        assert np.all(np.isfinite(result.positions))
        assert np.isfinite(result.magnitudes["max_relative_total_energy_drift"])
    first = random_bound_cluster_state(seed=7, n_bodies=6)
    second = random_bound_cluster_state(seed=7, n_bodies=6)
    assert np.allclose(first.positions, second.positions)
    assert np.allclose(first.velocities, second.velocities)
    assert ring_state(n_bodies=6, dimension=3).positions.shape == (6, 3)


def test_figure_eight_returns_near_start_after_one_period() -> None:
    state = figure_eight_state()
    result = solve_n_body(
        masses=state.masses,
        positions=state.positions,
        velocities=state.velocities,
        t_max=state.recommended_t_max,
        n_points=300,
    )
    assert np.max(np.linalg.norm(result.positions[-1] - state.positions, axis=1)) < 3e-3


def test_lagrange_equilateral_side_lengths_remain_equal() -> None:
    state = lagrange_equilateral_state()
    result = solve_n_body(
        masses=state.masses,
        positions=state.positions,
        velocities=state.velocities,
        t_max=state.recommended_t_max / 2.0,
        n_points=100,
    )
    side_lengths = np.stack(
        (
            selected_pair_distance(result, 0, 1),
            selected_pair_distance(result, 0, 2),
            selected_pair_distance(result, 1, 2),
        )
    )
    assert np.max(np.ptp(side_lengths, axis=0)) < 2e-5


def test_orbit_animation_uses_current_reference_frame_data() -> None:
    state = ring_state(n_bodies=4)
    result = solve_n_body(
        masses=state.masses,
        positions=state.positions,
        velocities=state.velocities,
        t_max=0.2,
        n_points=12,
    )
    payload = orbit_animation_payload(result, "Center of mass")
    assert np.allclose(np.average(payload.positions, axis=1, weights=result.masses), 0.0, atol=1e-8)
    figure = create_orbit_animation_figure(payload)
    assert callable(getattr(figure, "_animation_update", None))
    figure._animation_update(3)  # type: ignore[attr-defined]
    assert selected_pair_distance(result, 0, 1).shape == result.t.shape


def test_orbit_animation_tracks_nonzero_moving_com_in_2d_and_com_frame() -> None:
    result = solve_n_body(
        masses=[1.0, 2.0],
        positions=[[1.0, 0.0], [2.0, 0.0]],
        velocities=[[0.0, 1.0], [0.0, 2.0]],
        t_max=0.2,
        n_points=8,
    )
    inertial_payload = orbit_animation_payload(result, "Inertial")
    assert np.allclose(inertial_payload.center_of_mass, result.center_of_mass)
    assert not np.allclose(inertial_payload.center_of_mass, 0.0)
    inertial_figure = create_orbit_animation_figure(inertial_payload)
    try:
        inertial_figure._animation_update(4)  # type: ignore[attr-defined]
        marker = inertial_figure.axes[0].collections[-1]
        np.testing.assert_allclose(marker.get_offsets()[0], result.center_of_mass[4])
    finally:
        import matplotlib.pyplot as plt

        plt.close(inertial_figure)

    com_payload = orbit_animation_payload(result, "Center of mass")
    assert np.allclose(
        np.average(com_payload.positions, axis=1, weights=result.masses), 0.0, atol=1e-8
    )
    assert np.allclose(com_payload.center_of_mass, 0.0)
    com_figure = create_orbit_animation_figure(com_payload)
    try:
        com_figure._animation_update(4)  # type: ignore[attr-defined]
        np.testing.assert_allclose(com_figure.axes[0].collections[-1].get_offsets()[0], [0.0, 0.0])
    finally:
        plt.close(com_figure)


def test_orbit_animation_tracks_com_marker_in_3d() -> None:
    result = solve_n_body(
        masses=[1.0, 2.0],
        positions=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        velocities=[[0.0, 1.0, 0.5], [0.0, 2.0, 1.0]],
        t_max=0.1,
        n_points=5,
    )
    figure = create_orbit_animation_figure(orbit_animation_payload(result))
    try:
        figure._animation_update(3)  # type: ignore[attr-defined]
        marker = figure.axes[0].collections[-1]
        offsets = marker._offsets3d  # type: ignore[attr-defined]
        np.testing.assert_allclose([values[0] for values in offsets], result.center_of_mass[3])
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_custom_state_parser_uses_configured_gravity_and_softening() -> None:
    common = ("1, 1", "0, 0\n0, 0", "0, 0\n0, 0")
    with pytest.raises(ValueError, match="Coincident"):
        parse_n_body_state_text(
            *common, n_bodies=2, dimension=2, gravitational_constant=2.5, epsilon=0.0
        )
    masses, positions, velocities = parse_n_body_state_text(
        *common, n_bodies=2, dimension=2, gravitational_constant=2.5, epsilon=0.25
    )
    assert masses.tolist() == [1.0, 1.0]
    assert positions.shape == velocities.shape == (2, 2)
    with pytest.raises(ValueError, match="positive"):
        parse_n_body_state_text(
            *common, n_bodies=2, dimension=2, gravitational_constant=0.0, epsilon=0.25
        )
    with pytest.raises(ValueError, match="non-negative"):
        parse_n_body_state_text(
            *common, n_bodies=2, dimension=2, gravitational_constant=2.5, epsilon=-0.1
        )
