"""Tests for solver.statistics."""

from __future__ import annotations

import numpy as np
import pytest

from solver.statistics import compute_statistics


def test_compute_statistics_mean_rms_std() -> None:
    x = np.linspace(0, 10, 100)
    y = np.ones((1, 100)) * 3.0
    stats = compute_statistics(x, y, selected={"mean", "rms", "std"})
    assert "mean" in stats
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["rms"] == pytest.approx(3.0)
    assert stats["std"] == pytest.approx(0.0)


def test_compute_statistics_max_min() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([[1.0, 5.0, -1.0, 3.0]])
    stats = compute_statistics(x, y, selected={"max", "min"})
    assert stats["max"]["value"] == 5.0
    assert stats["max"]["x"] == 1.0
    assert stats["min"]["value"] == -1.0
    assert stats["min"]["x"] == 2.0


def test_compute_statistics_integral() -> None:
    x = np.linspace(0, 10, 100)
    y = np.ones((1, 100))  # constant 1
    stats = compute_statistics(x, y, selected={"integral"})
    assert stats["integral"] == pytest.approx(10.0, rel=1e-4)  # integral of 1 from 0 to 10


def test_compute_statistics_zero_crossings() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([[1.0, -1.0, 1.0, -1.0, 1.0]])
    stats = compute_statistics(x, y, selected={"zero_crossings"})
    assert stats["zero_crossings"] == 4


def test_compute_statistics_amplitude() -> None:
    x = np.linspace(0, 1, 10)
    y = np.array([np.linspace(-2, 4, 10)])  # range 6, amplitude 3
    stats = compute_statistics(x, y, selected={"amplitude"})
    assert stats["amplitude"] == pytest.approx(3.0)


def test_compute_statistics_energy_requires_two_components() -> None:
    x = np.linspace(0, 10, 50)
    y_2d = np.vstack([np.sin(x), np.cos(x)])  # position, velocity
    stats = compute_statistics(x, y_2d, selected={"energy"})
    assert "energy" in stats
    assert "kinetic_initial" in stats["energy"]
    assert "potential_initial" in stats["energy"]
    assert "total_initial" in stats["energy"]


def test_compute_statistics_energy_not_computed_for_single_component() -> None:
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    stats = compute_statistics(x, y, selected={"energy", "mean"})
    assert "mean" in stats
    assert "energy" not in stats


def test_compute_statistics_none_selected_uses_all() -> None:
    x = np.linspace(0, 10, 100)
    y = np.sin(x).reshape(1, -1)
    stats = compute_statistics(x, y, selected=None)
    assert "mean" in stats
    assert "rms" in stats
    assert "std" in stats
    assert "max" in stats
    assert "min" in stats
    assert "integral" in stats
    assert "zero_crossings" in stats


def test_compute_statistics_1d_y_accepted() -> None:
    x = np.linspace(0, 2 * np.pi, 50)
    y_1d = np.sin(x)
    stats = compute_statistics(x, y_1d, selected={"mean"})
    assert "mean" in stats
    assert stats["mean"] == pytest.approx(0.0, abs=0.1)
