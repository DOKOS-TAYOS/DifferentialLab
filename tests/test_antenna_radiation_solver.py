"""Tests for antenna radiation solver."""

from __future__ import annotations

import numpy as np
import pytest

from complex_problems.antenna_radiation.solver import solve_antenna_radiation


def test_dipole_case_returns_finite_pattern_and_reasonable_gain() -> None:
    result = solve_antenna_radiation(
        antenna_type="dipole",
        frequency_hz=1.0e9,
        transmit_power_w=10.0,
        efficiency=0.9,
        observation_distance_m=50.0,
        n_theta=121,
        n_phi=180,
        length_lambda=0.5,
    )

    assert result.gain_db.shape == (121, 180)
    assert np.all(np.isfinite(result.gain_db))
    assert -1.0 < result.magnitudes["gain_max_db"] < 4.0
    assert result.magnitudes["directivity_max_db"] > 0.0


def test_array_gain_exceeds_dipole_for_same_power_and_efficiency() -> None:
    dipole = solve_antenna_radiation(
        antenna_type="dipole",
        frequency_hz=1.0e9,
        transmit_power_w=10.0,
        efficiency=0.9,
        observation_distance_m=100.0,
        n_theta=91,
        n_phi=120,
        length_lambda=0.5,
    )
    array = solve_antenna_radiation(
        antenna_type="array",
        frequency_hz=1.0e9,
        transmit_power_w=10.0,
        efficiency=0.9,
        observation_distance_m=100.0,
        n_theta=91,
        n_phi=120,
        array_elements=8,
        array_spacing_lambda=0.5,
        array_steer_theta_deg=90.0,
    )
    assert array.magnitudes["gain_max_db"] > dipole.magnitudes["gain_max_db"]


def test_antenna_solver_rejects_invalid_efficiency() -> None:
    with pytest.raises(ValueError):
        solve_antenna_radiation(
            antenna_type="dipole",
            frequency_hz=1.0e9,
            efficiency=1.2,
        )
