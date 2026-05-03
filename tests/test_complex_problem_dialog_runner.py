"""Tests for the shared complex-problem dialog runner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from complex_problems.common.dialog_ui import run_solver_dialog


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
