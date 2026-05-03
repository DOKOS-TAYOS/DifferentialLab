"""Tests for performance guardrails shown before expensive runs."""

from __future__ import annotations

from unittest.mock import patch

from frontend.performance_guard import (
    PerformanceAdvisory,
    assess_parameters_dialog_request,
    assess_time_history_request,
    confirm_performance_advisory,
)


def test_assess_parameters_dialog_request_warns_for_dense_output_grid() -> None:
    advisory = assess_parameters_dialog_request(
        equation_type="ode",
        n_points=100_000,
        state_size=1,
    )

    assert advisory is not None
    assert advisory.severity == "warn"
    assert "100,000" in advisory.message


def test_assess_time_history_request_requires_confirmation_for_large_history() -> None:
    advisory = assess_time_history_request(
        label="2D membrane",
        frames=2_001,
        points_per_frame=256 * 256,
        array_count=2,
        bytes_per_value=8,
    )

    assert advisory is not None
    assert advisory.severity == "confirm"
    assert "2D membrane" in advisory.title


def test_confirm_performance_advisory_uses_warning_and_confirmation_dialogs() -> None:
    warn = PerformanceAdvisory(
        severity="warn",
        title="Large run",
        message="This may take a while.",
    )
    confirm = PerformanceAdvisory(
        severity="confirm",
        title="Very large run",
        message="Proceed?",
    )

    with patch("frontend.performance_guard.messagebox.showwarning") as showwarning:
        assert confirm_performance_advisory(parent=object(), advisory=warn) is True
    showwarning.assert_called_once()

    with patch("frontend.performance_guard.messagebox.askyesno", return_value=False) as askyesno:
        assert confirm_performance_advisory(parent=object(), advisory=confirm) is False
    askyesno.assert_called_once()
