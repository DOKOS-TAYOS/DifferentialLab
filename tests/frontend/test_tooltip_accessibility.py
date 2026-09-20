"""Unit tests for non-visual tooltip accessibility behavior."""

from __future__ import annotations

from typing import Any

from frontend.ui_dialogs.tooltip import ToolTip


class _FakeWidget:
    def __init__(self) -> None:
        self.bindings: dict[str, tuple[Any, str | None]] = {}
        self.after_calls: list[tuple[int, Any]] = []
        self.cancelled: list[str] = []

    def bind(self, sequence: str, callback: Any, add: str | None = None) -> None:
        self.bindings[sequence] = (callback, add)

    def after(self, delay: int, callback: Any) -> str:
        self.after_calls.append((delay, callback))
        return f"after-{len(self.after_calls)}"

    def after_cancel(self, identifier: str) -> None:
        self.cancelled.append(identifier)


def test_tooltip_binds_pointer_and_keyboard_events_without_replacing_existing_bindings() -> None:
    widget = _FakeWidget()

    ToolTip(widget, "Help", delay=25)  # type: ignore[arg-type]

    assert set(widget.bindings) == {"<Enter>", "<Leave>", "<FocusIn>", "<FocusOut>", "<Destroy>"}
    assert all(add == "+" for _, add in widget.bindings.values())


def test_tooltip_remains_active_while_pointer_or_focus_still_owns_it() -> None:
    widget = _FakeWidget()
    tooltip = ToolTip(widget, "Help", delay=25)  # type: ignore[arg-type]

    tooltip._on_enter(None)  # type: ignore[arg-type]
    first_after = tooltip._id_after
    tooltip._on_focus_in(None)  # type: ignore[arg-type]

    assert first_after in widget.cancelled
    assert tooltip._has_focus is True
    assert tooltip._pointer_inside is True

    tooltip._on_leave(None)  # type: ignore[arg-type]
    assert tooltip._id_after is not None

    tooltip._on_focus_out(None)  # type: ignore[arg-type]
    assert tooltip._id_after is None
