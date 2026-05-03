"""Background execution helper for long-running complex-problem solvers."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from typing import Any

from frontend.ui_dialogs.background_task import BackgroundTaskFailure, run_task_with_loading
from utils import get_logger

logger = get_logger(__name__)


def run_solver_with_loading(
    *,
    parent: tk.Tk | tk.Toplevel,
    message: str,
    task: Callable[[], Any],
    on_success: Callable[[Any], None],
    error_title: str = "Solver error",
    poll_ms: int = 100,
) -> None:
    """Run a blocking solver task on a daemon thread with a loading dialog."""

    def _format_error(exc: BaseException) -> BackgroundTaskFailure:
        logger.exception("Background solver task failed")
        return BackgroundTaskFailure(error_title, str(exc))

    run_task_with_loading(
        parent=parent,
        message=message,
        task=task,
        on_success=on_success,
        error_title=error_title,
        poll_ms=poll_ms,
        format_error=_format_error,
    )
