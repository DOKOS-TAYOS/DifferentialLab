"""Shared helpers for running GUI tasks behind a loading dialog."""

from __future__ import annotations

import gc
import queue
import threading
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import messagebox
from typing import Any, TypeVar

from frontend.ui_dialogs.loading_dialog import LoadingDialog

_TResult = TypeVar("_TResult")


_gc_guard_lock = threading.Lock()
_active_gc_guard_count = 0
_gc_was_enabled: bool | None = None


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


def _pause_cyclic_gc_for_worker() -> None:
    """Prevent a worker allocation from finalizing an unreachable Tk cycle.

    Cyclic GC is process-wide and runs in whichever thread crosses its allocation
    threshold.  A solver thread must not be that thread while a Tk UI is active:
    an old widget or image cycle could otherwise run its Tk finalizer there.
    The matching restore always happens from the Tk callback below.
    """
    global _active_gc_guard_count, _gc_was_enabled
    with _gc_guard_lock:
        if _active_gc_guard_count == 0:
            _gc_was_enabled = gc.isenabled()
            if _gc_was_enabled:
                gc.disable()
        _active_gc_guard_count += 1


def _restore_cyclic_gc_on_tk_thread() -> None:
    """Restore cyclic GC after a worker has finished on the Tk thread."""
    global _active_gc_guard_count, _gc_was_enabled
    with _gc_guard_lock:
        _active_gc_guard_count -= 1
        if _active_gc_guard_count != 0:
            return
        if _gc_was_enabled:
            gc.enable()
        _gc_was_enabled = None


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

    loading = LoadingDialog(parent, message=message)
    _pause_cyclic_gc_for_worker()
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    def _poll() -> None:
        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty:
            parent.after(poll_ms, _poll)
            return

        _restore_cyclic_gc_on_tk_thread()
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
