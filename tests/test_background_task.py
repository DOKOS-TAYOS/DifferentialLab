"""Tests for shared background-task UI helpers."""

from __future__ import annotations

import gc
import threading
from typing import Any
from unittest.mock import MagicMock, patch

from frontend.ui_dialogs.background_task import BackgroundTaskFailure, run_task_with_loading


class _FakeParent:
    def __init__(self) -> None:
        self.after_calls: list[tuple[int, Any]] = []

    def after(self, delay_ms: int, callback: Any) -> None:
        self.after_calls.append((delay_ms, callback))


class _FakeLoadingDialog:
    def __init__(self, _parent: object, *, message: str) -> None:
        self.message = message
        self.destroy_calls = 0

    def destroy(self) -> None:
        self.destroy_calls += 1


class _ImmediateThread:
    def __init__(self, *, target: Any, daemon: bool) -> None:
        self._target = target
        self.daemon = daemon

    def start(self) -> None:
        self._target()


class _JoiningThread:
    """Run the target on a real worker, then make the test deterministic."""

    def __init__(self, *, target: Any, daemon: bool) -> None:
        self._thread = _REAL_THREAD(target=target, daemon=daemon)

    def start(self) -> None:
        self._thread.start()
        self._thread.join()


class _ThreadTrackingCycle:
    def __init__(self, finalized_on: list[int]) -> None:
        self.cycle: _ThreadTrackingCycle | None = self
        self._finalized_on = finalized_on

    def __del__(self) -> None:
        self._finalized_on.append(threading.get_ident())


_REAL_THREAD = threading.Thread


def test_run_task_with_loading_delivers_success_on_parent_after() -> None:
    parent = _FakeParent()
    on_success = MagicMock()
    loading_dialogs: list[_FakeLoadingDialog] = []

    def loading_factory(parent_arg: object, *, message: str) -> _FakeLoadingDialog:
        loading = _FakeLoadingDialog(parent_arg, message=message)
        loading_dialogs.append(loading)
        return loading

    with (
        patch("frontend.ui_dialogs.background_task.LoadingDialog", side_effect=loading_factory),
        patch("frontend.ui_dialogs.background_task.threading.Thread", _ImmediateThread),
    ):
        run_task_with_loading(
            parent=parent,
            message="Solving...",
            task=lambda: "result",
            on_success=on_success,
        )

    assert len(parent.after_calls) == 1
    delay_ms, callback = parent.after_calls[0]
    assert delay_ms == 100

    callback()

    assert loading_dialogs[0].message == "Solving..."
    assert loading_dialogs[0].destroy_calls == 1
    on_success.assert_called_once_with("result")


def test_run_task_with_loading_formats_error_dialog() -> None:
    parent = _FakeParent()
    loading_dialogs: list[_FakeLoadingDialog] = []

    def loading_factory(parent_arg: object, *, message: str) -> _FakeLoadingDialog:
        loading = _FakeLoadingDialog(parent_arg, message=message)
        loading_dialogs.append(loading)
        return loading

    def task() -> str:
        raise MemoryError("out of room")

    def format_error(exc: BaseException) -> BackgroundTaskFailure:
        return BackgroundTaskFailure("Memory Error", f"Formatted: {exc}")

    with (
        patch("frontend.ui_dialogs.background_task.LoadingDialog", side_effect=loading_factory),
        patch("frontend.ui_dialogs.background_task.threading.Thread", _ImmediateThread),
        patch("frontend.ui_dialogs.background_task.messagebox.showerror") as showerror,
    ):
        run_task_with_loading(
            parent=parent,
            message="Solving...",
            task=task,
            on_success=MagicMock(),
            format_error=format_error,
        )
        parent.after_calls[0][1]()

    assert loading_dialogs[0].destroy_calls == 1
    showerror.assert_called_once_with("Memory Error", "Formatted: out of room", parent=parent)


def test_run_task_with_loading_defers_cyclic_finalizers_to_tk_thread() -> None:
    """Solver allocations cannot finalize an orphaned Tk-like cycle off-thread."""
    parent = _FakeParent()
    loading_dialogs: list[_FakeLoadingDialog] = []
    finalized_on: list[int] = []
    main_thread = threading.get_ident()
    gc_was_enabled = gc.isenabled()
    gc.enable()

    def loading_factory(parent_arg: object, *, message: str) -> _FakeLoadingDialog:
        loading = _FakeLoadingDialog(parent_arg, message=message)
        loading_dialogs.append(loading)
        return loading

    def task() -> str:
        _ThreadTrackingCycle(finalized_on)
        return "result"

    try:
        with (
            patch("frontend.ui_dialogs.background_task.LoadingDialog", side_effect=loading_factory),
            patch("frontend.ui_dialogs.background_task.threading.Thread", _JoiningThread),
        ):
            run_task_with_loading(
                parent=parent,
                message="Solving...",
                task=task,
                on_success=MagicMock(),
            )

        assert loading_dialogs[0].destroy_calls == 0
        assert finalized_on == []

        parent.after_calls[0][1]()
        gc.collect()
    finally:
        if not gc_was_enabled:
            gc.disable()

    assert gc.isenabled() is gc_was_enabled
    assert finalized_on == [main_thread]
