"""Tests for the shared complex-problem dialog runner."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast
from unittest.mock import MagicMock, patch

from complex_problems.common.background import run_solver_with_loading
from complex_problems.common.dialog_ui import run_solver_dialog
from frontend.ui_dialogs.background_task import BackgroundTaskFailure


class _FakeWindow:
    def __init__(self) -> None:
        self.destroy_calls = 0

    def destroy(self) -> None:
        self.destroy_calls += 1


def test_run_solver_dialog_stops_when_confirmation_is_declined() -> None:
    params = {"nx": 4096}
    collect_inputs = MagicMock(return_value=params)
    confirm_run = MagicMock(return_value=False)
    solver = MagicMock()
    window = _FakeWindow()

    with patch("complex_problems.common.dialog_ui.run_solver_with_loading") as run_solver:
        run_solver_dialog(
            parent=object(),
            window=window,
            collect_inputs=collect_inputs,
            solver=solver,
            message="Solving...",
            result_parent=object(),
            result_dialog_factory=MagicMock(),
            confirm_run=confirm_run,
        )

    confirm_run.assert_called_once_with(params, window)
    run_solver.assert_not_called()
    assert window.destroy_calls == 0


def test_complex_problem_background_failure_uses_solver_error_path() -> None:
    """Complex-problem worker exceptions retain their user-facing error title."""
    forwarded_kwargs: dict[str, object] = {}

    def capture_runner(**kwargs: object) -> None:
        forwarded_kwargs.update(kwargs)

    with patch(
        "complex_problems.common.background.run_task_with_loading", side_effect=capture_runner
    ):
        run_solver_with_loading(
            parent=MagicMock(),
            message="Solving...",
            task=lambda: None,
            on_success=MagicMock(),
            error_title="Membrane solver failed",
        )

    formatter = cast(
        Callable[[BaseException], BackgroundTaskFailure],
        forwarded_kwargs["format_error"],
    )
    failure = formatter(RuntimeError("unstable grid"))
    assert failure.title == "Membrane solver failed"
    assert failure.message == "unstable grid"
