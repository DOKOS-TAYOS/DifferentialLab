"""Shared helpers for running GUI tasks behind a loading dialog."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import messagebox
from typing import Any, TypeVar

from frontend.ui_dialogs.loading_dialog import LoadingDialog

_TResult = TypeVar("_TResult")


@dataclass(frozen=True, slots=True)
class BackgroundTaskFailure:
    """User-facing failure details for a background task."""

    title: str
    message: str


def _default_format_error(error_title: str) -> Callable[[BaseException], BackgroundTaskFailure]:
    """Build the default exception-to-dialog formatter."""

    def _format_error(exc: BaseException) -> BackgroundTaskFailure:
        return BackgroundTaskFailure(error_title, str(exc))

    return _format_error


def _widget_exists(widget: tk.Tk | tk.Toplevel) -> bool:
    """Return whether a Tk widget still exists, tolerating test doubles."""
    try:
        return bool(widget.winfo_exists())
    except (AttributeError, tk.TclError):
        return True


def run_task_with_loading(
    *,
    parent: tk.Tk | tk.Toplevel,
    message: str,
    task: Callable[[], _TResult],
    on_success: Callable[[_TResult], None],
    error_title: str = "Error",
    poll_ms: int = 100,
    format_error: Callable[[BaseException], BackgroundTaskFailure] | None = None,
    on_complete: Callable[[], None] | None = None,
) -> None:
    """Run a blocking task on a daemon thread and report the result on the Tk thread."""
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
    format_failure = format_error or _default_format_error(error_title)

    def _worker() -> None:
        try:
            result_queue.put(("success", task()))
        except Exception as exc:  # pragma: no cover - exercised through injected task
            result_queue.put(("error", format_failure(exc)))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    loading = LoadingDialog(parent, message=message)

    def _poll() -> None:
        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty:
            parent.after(poll_ms, _poll)
            return

        try:
            loading.destroy()
        except tk.TclError:
            pass

        if on_complete is not None:
            on_complete()

        if not _widget_exists(parent):
            return

        if status == "success":
            on_success(payload)
            return

        failure: BackgroundTaskFailure = payload
        messagebox.showerror(failure.title, failure.message, parent=parent)

    parent.after(poll_ms, _poll)
