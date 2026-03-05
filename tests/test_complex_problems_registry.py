"""Tests for complex problems plugin registry."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from complex_problems.problem_registry import get_problem_descriptors, open_problem_dialog


def test_registry_contains_coupled_oscillators() -> None:
    descriptors = get_problem_descriptors()
    assert "coupled_oscillators" in descriptors
    desc = descriptors["coupled_oscillators"]
    assert desc.id == "coupled_oscillators"
    assert "Coupled" in desc.name


def test_open_problem_dialog_dispatches_to_plugin() -> None:
    parent = object()
    with patch("complex_problems.coupled_oscillators.problem.PROBLEM.open_dialog") as mock_open:
        open_problem_dialog("coupled_oscillators", parent)  # type: ignore[arg-type]
    mock_open.assert_called_once_with(parent)


def test_open_problem_dialog_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        open_problem_dialog("missing_problem", object())  # type: ignore[arg-type]

