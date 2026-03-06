"""Tests for complex problems plugin registry."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from complex_problems.problem_registry import get_problem_descriptors, open_problem_dialog


def test_registry_contains_expected_plugins() -> None:
    descriptors = get_problem_descriptors()
    assert "coupled_oscillators" in descriptors
    assert "membrane_2d" in descriptors
    assert "nonlinear_waves" in descriptors
    assert "schrodinger_td" in descriptors
    assert "antenna_radiation" in descriptors
    assert "aerodynamics_2d" in descriptors
    assert "pipe_flow" in descriptors

    desc = descriptors["coupled_oscillators"]
    assert desc.id == "coupled_oscillators"
    assert "Coupled" in desc.name


def test_open_problem_dialog_dispatches_to_coupled_oscillators() -> None:
    parent = object()
    with patch("complex_problems.coupled_oscillators.problem.PROBLEM.open_dialog") as mock_open:
        open_problem_dialog("coupled_oscillators", parent)  # type: ignore[arg-type]
    mock_open.assert_called_once_with(parent)


def test_open_problem_dialog_dispatches_to_membrane_2d() -> None:
    parent = object()
    with patch("complex_problems.membrane_2d.problem.PROBLEM.open_dialog") as mock_open:
        open_problem_dialog("membrane_2d", parent)  # type: ignore[arg-type]
    mock_open.assert_called_once_with(parent)


def test_open_problem_dialog_dispatches_to_nonlinear_waves() -> None:
    parent = object()
    with patch("complex_problems.nonlinear_waves.problem.PROBLEM.open_dialog") as mock_open:
        open_problem_dialog("nonlinear_waves", parent)  # type: ignore[arg-type]
    mock_open.assert_called_once_with(parent)


def test_open_problem_dialog_dispatches_to_schrodinger_td() -> None:
    parent = object()
    with patch("complex_problems.schrodinger_td.problem.PROBLEM.open_dialog") as mock_open:
        open_problem_dialog("schrodinger_td", parent)  # type: ignore[arg-type]
    mock_open.assert_called_once_with(parent)


def test_open_problem_dialog_dispatches_to_antenna_radiation() -> None:
    parent = object()
    with patch("complex_problems.antenna_radiation.problem.PROBLEM.open_dialog") as mock_open:
        open_problem_dialog("antenna_radiation", parent)  # type: ignore[arg-type]
    mock_open.assert_called_once_with(parent)


def test_open_problem_dialog_dispatches_to_aerodynamics_2d() -> None:
    parent = object()
    with patch("complex_problems.aerodynamics_2d.problem.PROBLEM.open_dialog") as mock_open:
        open_problem_dialog("aerodynamics_2d", parent)  # type: ignore[arg-type]
    mock_open.assert_called_once_with(parent)


def test_open_problem_dialog_dispatches_to_pipe_flow() -> None:
    parent = object()
    with patch("complex_problems.pipe_flow.problem.PROBLEM.open_dialog") as mock_open:
        open_problem_dialog("pipe_flow", parent)  # type: ignore[arg-type]
    mock_open.assert_called_once_with(parent)


def test_open_problem_dialog_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        open_problem_dialog("missing_problem", object())  # type: ignore[arg-type]
