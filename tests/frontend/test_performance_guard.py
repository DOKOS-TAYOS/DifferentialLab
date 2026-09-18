"""Tests for performance guardrails shown before expensive runs."""

from __future__ import annotations

from unittest.mock import patch

from frontend.performance_guard import (
    PerformanceAdvisory,
    assess_parameters_dialog_request,
    assess_pde_3d_request,
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
    assert advisory.title == "Dense output request"
    assert "100,000" in advisory.message
    assert "plotting and exporting" in advisory.message


def test_vector_pde_components_are_included_in_grid_cost() -> None:
    advisory = assess_parameters_dialog_request(
        equation_type="pde",
        n_points=200,
        n_points_y=200,
        state_size=4,
    )

    assert advisory is not None
    assert advisory.severity == "warn"
    assert "4 components" in advisory.message


def test_large_vector_pde_request_still_requires_advisory() -> None:
    advisory = assess_parameters_dialog_request(
        equation_type="pde",
        n_points=1000,
        n_points_y=1000,
        state_size=2,
    )

    assert advisory is not None
    assert advisory.severity == "confirm"
    assert "2 components" in advisory.message


def test_pde_3d_advisory_estimates_sparse_system_and_memory() -> None:
    advisory = assess_pde_3d_request(nx=50, ny=50, nz=50)

    assert advisory is not None
    assert advisory.severity == "warn"
    assert advisory.title == "Large PDE 3D grid request"
    assert "125,000 points" in advisory.message
    assert "unknowns" in advisory.message
    assert "stencil entries" in advisory.message


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
    assert advisory.title == "Large 2D membrane history"
    assert "Continue only if this is intentional." in advisory.message


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
