"""Physics and deterministic integration checks for the dedicated FPUT workflow."""

from __future__ import annotations

import numpy as np
import pytest

from complex_problems.fput_experiment.model import (
    FPUT_PRESETS,
    bond_strain,
    energy_components,
    fput_force,
    legacy_kink_pair_initial_state,
    modal_coordinates,
    normal_mode_basis,
    particle_coordinates,
    single_mode_initial_state,
    two_adjacent_modes_initial_state,
)
from complex_problems.fput_experiment.result_dialog import (
    create_fput_animation_figure,
    create_hamiltonian_figure,
    create_modal_energy_figure,
    format_fput_summary,
    fundamental_angular_frequency,
    fundamental_cycles_to_time,
    prepare_fput_animation_payload,
    time_to_fundamental_cycles,
)
from complex_problems.fput_experiment.solver import (
    detect_recurrence_peaks,
    recurrence_fidelity,
    solve_fput,
    solve_fput_recurrence_scaling,
    summarize_recurrence_peaks,
)
from complex_problems.fput_experiment.ui import preset_study_transition


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
    with np.errstate(all="raise"):
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


def test_recurrence_peak_summary_reserves_late_for_peaks_after_first() -> None:
    """A first recurrence alone is never described as a late recurrence."""
    one_peak = summarize_recurrence_peaks(np.array([10.0]), np.array([0.91]))
    assert one_peak["first_recurrence_time"] == 10.0
    assert one_peak["highest_late_recurrence_time"] is None
    assert one_peak["highest_late_recurrence_fidelity"] is None
    multiple = summarize_recurrence_peaks(
        np.array([10.0, 20.0, 30.0]), np.array([0.99, 0.91, 0.95])
    )
    assert multiple["first_recurrence_time"] == 10.0
    assert multiple["highest_late_recurrence_time"] == 30.0
    assert multiple["highest_late_recurrence_fidelity"] == 0.95


def test_recurrence_scaling_preset_transition_never_reinterprets_beta_as_alpha() -> None:
    """Selecting a beta preset while scaling returns to ordinary beta simulation."""
    assert preset_study_transition("Recurrence scaling", "beta") == ("Single simulation", "beta")
    assert preset_study_transition("Recurrence scaling", "alpha") == ("Recurrence scaling", "alpha")
    assert preset_study_transition("Single simulation", "beta") == ("Single simulation", "beta")


def test_fundamental_cycle_transform_round_trips_and_uses_fixed_end_frequency() -> None:
    """The secondary recurrence axis uses the stated fundamental-mode transform."""
    assert fundamental_angular_frequency(3) == pytest.approx(2.0 * np.sin(np.pi / 8.0))
    time = np.array([0.0, 1.5, 7.0])
    cycles = time_to_fundamental_cycles(time, 3)
    np.testing.assert_allclose(fundamental_cycles_to_time(cycles, 3), time)


@pytest.mark.parametrize(
    ("fidelity", "expected"),
    [
        (np.ones(6), []),
        (np.array([1.0, 0.7, 0.79, 0.7, 0.75, 0.7]), []),
        (np.array([1.0, 0.7, 0.91, 0.7, 0.95, 0.7]), [2, 4]),
        (np.array([1.0, 0.7, 0.79, 0.78, 0.7]), []),
    ],
)
def test_recurrence_detector_synthetic_acceptance_cases(
    fidelity: np.ndarray, expected: list[int]
) -> None:
    """Detector requires departure, rejects weak maxima, and keeps accepted peaks."""
    indices, _, _ = detect_recurrence_peaks(np.arange(fidelity.size, dtype=float), fidelity)
    np.testing.assert_array_equal(indices, expected)


def test_alpha_recurrence_and_scaling_are_numerically_detected() -> None:
    """Cheap alpha cases depart and recover with finite recurrence measurements."""
    result = solve_fput(
        n_particles=8, model="alpha", coefficient=0.25, t_end=500.0, dt=0.02, sample_every=20
    )
    assert np.min(result.recurrence_fidelity) < 0.8
    assert result.summary["first_recurrence_time"] is not None
    assert result.summary["first_recurrence_fidelity"] > 0.9  # type: ignore[operator]
    sweep = solve_fput_recurrence_scaling(
        sweep_variable="N", parameter_values=[8, 12, 16], t_end=2200.0, dt=0.02, sample_every=20
    )
    assert np.all(sweep.detected)
    assert np.all(np.isfinite(sweep.first_recurrence_times))
    assert np.all((sweep.recurrence_fidelities > 0.8) & (sweep.recurrence_fidelities <= 1.0))
    assert all(
        value is not None and np.isfinite(value)
        for value in (sweep.slope, sweep.intercept, sweep.r_squared)
    )


def test_short_scaling_horizon_marks_undetected_runs_and_excludes_fit() -> None:
    """Too-short recurrence horizons retain NaN measurements outside the fit."""
    sweep = solve_fput_recurrence_scaling(
        sweep_variable="N", parameter_values=[8, 12, 16], t_end=1.0, dt=0.02, sample_every=10
    )
    assert np.any(~sweep.detected)
    assert np.any(np.isnan(sweep.first_recurrence_times))
    assert sweep.slope is None and sweep.intercept is None and sweep.r_squared is None


def test_result_summary_and_energy_figures_use_cached_data() -> None:
    """No-recurrence summaries and diagnostic layouts remain clear and separated."""
    result = solve_fput(n_particles=4, t_end=0.1, dt=0.01, sample_every=2)
    summary = format_fput_summary(result)
    assert "not detected" in summary
    identity = result.total_energy - np.sum(result.modal_energy, axis=1)
    assert np.max(np.abs(identity - result.nonlinear_potential_energy)) < 1.0e-12
    hamiltonian = create_hamiltonian_figure(result)
    assert len(hamiltonian.axes) == 2
    assert hamiltonian.axes[0].get_ylabel() == "energy"
    assert hamiltonian.axes[1].get_ylabel() == "relative H error"
    modal = create_modal_energy_figure(result, [1, 2], "Linear")
    curve_axis, heatmap_axis = modal.axes[:2]
    assert curve_axis is not heatmap_axis
    assert curve_axis.get_position().y0 >= heatmap_axis.get_position().y1


def test_legacy_kink_pair_formula_and_alpha_validation() -> None:
    """Legacy CC0 state is finite, parameterized, and rejects zero alpha."""
    x, v = legacy_kink_pair_initial_state(4, 1.0, 0.25)
    edited_x, edited_v = legacy_kink_pair_initial_state(
        4, 1.0, 0.25, width=0.7, first_center=2.0, second_center=5.0
    )
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(v))
    assert not np.allclose(x, edited_x) and not np.allclose(v, edited_v)
    with pytest.raises(ValueError, match="non-zero alpha"):
        legacy_kink_pair_initial_state(4, 1.0, 0.0)


def test_historical_preset_constants_are_preserved() -> None:
    """Historical source-derived configurations remain explicit regressions."""
    expected = {
        "Historical alpha recurrence": (32, "alpha", 0.25, 1, 11500.0, 0.01, 100),
        "Historical alpha superrecurrence": (32, "alpha", 0.5, 1, 240000.0, 0.02, 200),
        "Historical phase-space study": (3, "alpha", 0.25, 1, 15000.0, 0.01, 10),
        "Historical beta recurrence": (32, "beta", 10.0, 2, 600.0, 1.0e-4, 400),
    }
    for name, values in expected.items():
        preset = FPUT_PRESETS[name]
        assert (
            preset.n_particles,
            preset.model,
            preset.coefficient,
            preset.mode,
            preset.t_end,
            preset.dt,
            preset.sample_every,
        ) == values


def test_displacement_and_strain_animation_payloads_have_shared_metadata() -> None:
    """Both selected representations make valid cached animation figures."""
    result = solve_fput(n_particles=4, t_end=0.1, dt=0.01, sample_every=2)
    for representation, width in (("displacement", 6), ("strain", 5)):
        payload = prepare_fput_animation_payload(result, representation)  # type: ignore[arg-type]
        figure = create_fput_animation_figure(payload)
        assert payload.frames.shape == (result.t.size, width)
        assert callable(figure._animation_update)  # type: ignore[attr-defined]
        assert figure._animation_n_points == result.t.size  # type: ignore[attr-defined]
