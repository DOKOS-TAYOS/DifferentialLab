"""Background execution helper for long-running complex-problem solvers."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox
from typing import Any

from frontend.ui_dialogs.loading_dialog import LoadingDialog
from utils import get_logger

logger = get_logger(__name__)


def run_solver_with_loading(
    *,
    parent: tk.Tk | tk.Toplevel,
    message: str,
    task: Callable[[], Any],
    on_success: Callable[[Any], None],
    error_title: str = "Solver Error",
    poll_ms: int = 100,
) -> None:
    """Run a blocking solver task on a daemon thread with a loading dialog."""
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def _worker() -> None:
        try:
            result = task()
            result_queue.put(("success", result))
        except Exception as exc:  # pragma: no cover - GUI path
            logger.exception("Background solver task failed")
            result_queue.put(("error", str(exc)))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    loading = LoadingDialog(parent, message=message)

    def _poll() -> None:
        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty:
            parent.after(poll_ms, _poll)
            return

        loading.destroy()
        if status == "success":
            on_success(payload)
            return

        messagebox.showerror(error_title, payload, parent=parent)

    parent.after(poll_ms, _poll)

