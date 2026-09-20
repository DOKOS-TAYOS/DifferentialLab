"""Physics and deterministic integration checks for the dedicated FPUT workflow."""

from __future__ import annotations

import numpy as np
import pytest

from complex_problems.fput_experiment.model import (
    bond_strain,
    energy_components,
    fput_force,
    modal_coordinates,
    normal_mode_basis,
    particle_coordinates,
    single_mode_initial_state,
    two_adjacent_modes_initial_state,
)
from complex_problems.fput_experiment.result_dialog import (
    create_fput_animation_figure,
    prepare_fput_animation_payload,
)
from complex_problems.fput_experiment.solver import (
    detect_recurrence_peaks,
    recurrence_fidelity,
    solve_fput,
)


def test_alpha_force_matches_hand_computation() -> None:
    """The alpha force uses both fixed-end bonds with the stated signs."""
    x = np.array([1.0, -2.0])
    force = fput_force(x, model="alpha", coefficient=0.5)
    np.testing.assert_allclose(force, [0.0, 2.5])


def test_beta_force_matches_hand_computation() -> None:
    """The beta force uses cubic bond forces and fixed endpoints."""
    x = np.array([1.0, -2.0])
    force = fput_force(x, model="beta", coefficient=0.5)
    np.testing.assert_allclose(force, [-18.0, 22.5])


@pytest.mark.parametrize("model, coefficient", [("alpha", 0.25), ("beta", 1.5)])
def test_force_is_negative_potential_gradient(model: str, coefficient: float) -> None:
    """Finite differences verify the nonlinear force is -dV/dx."""
    x = np.array([0.3, -0.2, 0.5])
    expected = fput_force(x, model=model, coefficient=coefficient)  # type: ignore[arg-type]
    epsilon = 1.0e-7
    numerical = np.empty_like(x)
    for idx in range(x.size):
        plus, minus = x.copy(), x.copy()
        plus[idx] += epsilon
        minus[idx] -= epsilon
        numerical[idx] = -(
            energy_components(plus, np.zeros_like(x), model=model, coefficient=coefficient)[3]
            - energy_components(minus, np.zeros_like(x), model=model, coefficient=coefficient)[3]
        ) / (2.0 * epsilon)
    np.testing.assert_allclose(expected, numerical, atol=1.0e-7)


def test_boundary_bonds_and_hamiltonian_decomposition() -> None:
    """Potential includes the left and right fixed-end spring bonds."""
    values = energy_components(
        np.array([1.0, 0.0]), np.array([2.0, 0.0]), model="alpha", coefficient=0.5
    )
    kinetic, harmonic, nonlinear, potential, total = values
    assert kinetic == pytest.approx(2.0)
    assert harmonic == pytest.approx(1.0)
    assert nonlinear == pytest.approx(0.0)
    assert potential == pytest.approx(harmonic + nonlinear)
    assert total == pytest.approx(kinetic + potential)


def test_normal_modes_are_orthonormal_and_invertible() -> None:
    """The fixed-end sine transform round-trips Q/P state vectors."""
    basis, _ = normal_mode_basis(7)
    np.testing.assert_allclose(basis.T @ basis, np.eye(7), atol=1.0e-14)
    x, v = np.arange(7, dtype=float), -np.arange(7, dtype=float)
    q, p = modal_coordinates(x, v)
    back_x, back_v = particle_coordinates(q, p)
    np.testing.assert_allclose(back_x, x, atol=1.0e-13)
    np.testing.assert_allclose(back_v, v, atol=1.0e-13)


def test_harmonic_single_mode_has_no_modal_transfer() -> None:
    """A linear reference evolves only its initially excited normal mode."""
    result = solve_fput(n_particles=8, coefficient=0.0, t_end=20.0, dt=0.01, sample_every=5)
    assert np.max(result.modal_energy[:, 1:]) < 1.0e-12


def test_entropy_participation_and_zero_energy_are_safe() -> None:
    """Modal-spreading diagnostics honor their physical bounds and zero-energy convention."""
    result = solve_fput(n_particles=5, x0=np.zeros(5), v0=np.zeros(5), t_end=1.0, dt=0.1)
    assert np.all(result.spectral_entropy == 0.0)
    assert np.all(result.participation_number == 0.0)
    nonzero = solve_fput(n_particles=5, t_end=1.0, dt=0.1)
    assert np.all((0.0 <= nonzero.spectral_entropy) & (nonzero.spectral_entropy <= 1.0))
    assert np.all((1.0 <= nonzero.participation_number) & (nonzero.participation_number <= 5.0))


def test_strain_and_historical_initial_mode_formulas() -> None:
    """Strain has N+1 entries and historical state formulas remain direct."""
    np.testing.assert_allclose(bond_strain(np.array([2.0, 3.0])), [2.0, 1.0, -3.0])
    x, v = single_mode_initial_state(4, 2.0, 1)
    np.testing.assert_allclose(x, 2.0 * np.sin(np.pi * np.arange(1, 5) / 5.0))
    assert np.all(v == 0.0)
    two_x, _ = two_adjacent_modes_initial_state(4, 2.0, 1)
    expected = np.sin(np.pi * np.arange(1, 5) / 5.0) + np.sin(2.0 * np.pi * np.arange(1, 5) / 5.0)
    np.testing.assert_allclose(two_x, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_particles": 1},
        {"n_particles": 4, "model": "beta", "coefficient": -1.0},
        {"n_particles": 4, "coefficient": np.nan},
        {"n_particles": 4, "dt": 0.0},
        {"n_particles": 4, "x0": [0.0, 0.0]},
    ],
)
def test_invalid_fput_inputs_are_rejected(kwargs: dict[str, object]) -> None:
    """Invalid dimensions, coefficients, steps, and states fail explicitly."""
    with pytest.raises(ValueError):
        solve_fput(**kwargs)  # type: ignore[arg-type]


def test_final_partial_step_and_rk4_verlet_short_agreement() -> None:
    """Both steppers end exactly on the requested endpoint and agree when resolved."""
    verlet = solve_fput(n_particles=5, t_end=0.105, dt=0.01, sample_every=3)
    rk4 = solve_fput(n_particles=5, t_end=0.105, dt=0.01, sample_every=3, integrator="rk4")
    assert verlet.t[-1] == pytest.approx(0.105)
    assert verlet.metadata["final_partial_step"] == pytest.approx(0.005)
    np.testing.assert_allclose(verlet.displacement[-1], rk4.displacement[-1], atol=2.0e-5)
    assert verlet.summary["maximum_relative_hamiltonian_drift"] < 1.0e-3


def test_synthetic_recurrence_detector_excludes_initial_time() -> None:
    """Only post-departure high peaks are classified as recurrence peaks."""
    t = np.arange(8, dtype=float)
    fidelity = np.array([1.0, 0.9, 0.6, 0.91, 0.7, 0.95, 0.7, 0.79])
    indices, times, values = detect_recurrence_peaks(t, fidelity)
    np.testing.assert_array_equal(indices, [3, 5])
    np.testing.assert_array_equal(times, [3.0, 5.0])
    np.testing.assert_allclose(values, [0.91, 0.95])
    fractions = np.array([[0.5, 0.5], [0.25, 0.75]])
    np.testing.assert_allclose(recurrence_fidelity(fractions), [1.0, 0.75])


def test_displacement_and_strain_animation_payloads_have_shared_metadata() -> None:
    """Both selected representations make valid cached animation figures."""
    result = solve_fput(n_particles=4, t_end=0.1, dt=0.01, sample_every=2)
    for representation, width in (("displacement", 6), ("strain", 5)):
        payload = prepare_fput_animation_payload(result, representation)  # type: ignore[arg-type]
        figure = create_fput_animation_figure(payload)
        assert payload.frames.shape == (result.t.size, width)
        assert callable(figure._animation_update)  # type: ignore[attr-defined]
        assert figure._animation_n_points == result.t.size  # type: ignore[attr-defined]
