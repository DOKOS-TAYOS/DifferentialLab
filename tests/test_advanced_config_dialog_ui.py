"""Deterministic contracts for the Advanced configuration presentation layer."""

from __future__ import annotations

from complex_problems.common.dialog_ui import ADVANCED_ACTION_STYLES
from complex_problems.fput_experiment.ui import fput_study_visibility
from complex_problems.gravitational_n_body.ui import n_body_control_visibility
from complex_problems.schrodinger_td.ui import schrodinger_visible_groups


def test_advanced_footer_styles_keep_execution_primary_and_close_neutral() -> None:
    assert ADVANCED_ACTION_STYLES == {
        "primary": "Primary.TButton",
        "secondary": "Secondary.TButton",
    }


def test_fput_study_mode_exposes_only_relevant_control_group() -> None:
    assert fput_study_visibility("Single simulation") == (True, False)
    assert fput_study_visibility("Recurrence scaling") == (False, True)


def test_n_body_mode_and_preset_visibility() -> None:
    assert n_body_control_visibility("Three-body", "Custom") == (False, False, False)
    assert n_body_control_visibility("General N-body", "Custom") == (True, False, False)
    assert n_body_control_visibility("General N-body", "Random bound cluster") == (
        True,
        True,
        False,
    )
    assert n_body_control_visibility("General N-body", "Rotating ring") == (
        True,
        False,
        True,
    )


def test_schrodinger_visibility_tracks_each_independent_choice() -> None:
    groups = schrodinger_visible_groups(
        potential="barrier",
        packet="custom",
        dimension="2D",
        boundary="absorbing",
    )
    assert groups == {
        "potential:barrier",
        "packet:custom",
        "dimension:2d",
        "boundary:absorbing",
    }

    simple_groups = schrodinger_visible_groups(
        potential="free",
        packet="gaussian",
        dimension="1D",
        boundary="periodic",
    )
    assert simple_groups == {"potential:free", "packet:gaussian"}
