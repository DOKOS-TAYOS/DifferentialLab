"""Pure presentation helpers for the standard Results workspace."""

from __future__ import annotations

import numpy as np

from frontend.ui_dialogs.result_dialog import (
    _format_display_value,
    computed_metric_items,
    diagnostic_items,
    event_summary_items,
    humanize_metric_label,
    modify_setup_available,
    run_summary_items,
)


def test_run_summary_uses_only_available_factual_fields() -> None:
    metadata = {
        "solver_success": True,
        "method": "BDF",
        "num_points": 1001,
        "n_evaluations": 245,
        "rtol": 1.0e-6,
        "atol": None,
    }

    assert run_summary_items(metadata) == [
        ("Status", "Completed successfully"),
        ("Method", "BDF"),
        ("Points", 1001),
        ("Function evaluations", 245),
        ("Relative tolerance", 1.0e-6),
    ]
    assert run_summary_items({"solver_success": False}) == [("Status", "Solver reported failure")]
    assert run_summary_items({}) == []


def test_diagnostics_extract_known_values_without_quality_verdicts() -> None:
    metadata = {
        "solver_message": "The solver reached the end of the integration interval.",
        "relative_residual_l2": 2.5e-8,
        "n_jacobian_evals": 4,
        "condition_estimate": 1234.0,
        "pde_warnings": (),
    }

    assert diagnostic_items(metadata) == [
        ("Solver message", metadata["solver_message"]),
        ("Relative residual", 2.5e-8),
        ("Jacobian evaluations", 4),
        ("Condition estimate", 1234.0),
    ]


def test_event_summary_counts_groups_and_omits_state_arrays() -> None:
    metadata = {
        "event_times": [np.array([0.5, 1.5]), np.array([2.0])],
        "event_states": [np.ones((2, 3))],
    }

    assert event_summary_items(metadata) == [
        ("Detected events", 3),
        ("Event times", {"Event 1": [0.5, 1.5], "Event 2": [2.0]}),
    ]
    assert event_summary_items({"event_times": [], "event_states": []}) == [("Detected events", 0)]
    assert event_summary_items({}) == []


def test_metric_labels_and_nested_values_are_preserved_for_display() -> None:
    nested = {
        "magnitude": {"max": {"value": 3.0, "x": 0.5, "y": 0.25}},
        "component_0": {"gradient_norm": 1.25},
    }
    items = computed_metric_items({"std": 0.5, "vector_summary": nested})

    assert humanize_metric_label("rms") == "RMS"
    assert humanize_metric_label("std") == "Standard deviation"
    assert humanize_metric_label("some_metric_name") == "Some metric name"
    assert items[0][:2] == ("Standard deviation", 0.5)
    assert items[1][:2] == ("Vector summary", nested)
    assert computed_metric_items({}) == []


def test_unavailable_display_values_render_as_na() -> None:
    assert _format_display_value(None) == "N/A"
    assert _format_display_value([1.0, None]) == "1, N/A"


def test_modify_setup_requires_both_workflow_objects() -> None:
    session = object()
    selection = object()

    assert modify_setup_available(session, selection) is True
    assert modify_setup_available(session, None) is False
    assert modify_setup_available(None, selection) is False
    assert modify_setup_available(None, None) is False
