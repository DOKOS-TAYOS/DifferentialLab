"""Unit tests for non-visual tooltip accessibility behavior."""

from __future__ import annotations

from typing import Any

from frontend.ui_dialogs.tooltip import ToolTip


class _FakeWidget:
    def __init__(self, toplevel: _FakeToplevel | None = None) -> None:
        self.bindings: dict[str, tuple[Any, str | None]] = {}
        self.after_calls: list[tuple[int, Any]] = []
        self.cancelled: list[str] = []
        self.toplevel = toplevel or _FakeToplevel()

    def bind(self, sequence: str, callback: Any, add: str | None = None) -> None:
        self.bindings[sequence] = (callback, add)

    def after(self, delay: int, callback: Any) -> str:
        self.after_calls.append((delay, callback))
        return f"after-{len(self.after_calls)}"

    def after_cancel(self, identifier: str) -> None:
        self.cancelled.append(identifier)

    def winfo_toplevel(self) -> _FakeToplevel:
        return self.toplevel


class _FakeToplevel:
    def __init__(self) -> None:
        self.bindings: dict[str, tuple[Any, str | None]] = {}

    def bind(self, sequence: str, callback: Any, add: str | None = None) -> None:
        self.bindings[sequence] = (callback, add)


def test_tooltip_binds_pointer_and_keyboard_events_without_replacing_existing_bindings() -> None:
    widget = _FakeWidget()

    ToolTip(widget, "Help", delay=25)  # type: ignore[arg-type]

    assert set(widget.bindings) == {
        "<Enter>",
        "<Leave>",
        "<ButtonPress>",
        "<FocusIn>",
        "<FocusOut>",
        "<Destroy>",
    }
    assert all(add == "+" for _, add in widget.bindings.values())


def test_tooltip_remains_active_while_pointer_or_focus_still_owns_it() -> None:
    widget = _FakeWidget()
    tooltip = ToolTip(widget, "Help", delay=25)  # type: ignore[arg-type]

    tooltip._on_enter(None)  # type: ignore[arg-type]
    first_after = tooltip._id_after
    tooltip._mark_keyboard_modality(None)  # type: ignore[arg-type]
    tooltip._on_focus_in(None)  # type: ignore[arg-type]

    assert first_after in widget.cancelled
    assert tooltip._has_focus is True
    assert tooltip._pointer_inside is True

    tooltip._on_leave(None)  # type: ignore[arg-type]
    assert tooltip._id_after is not None

    tooltip._on_focus_out(None)  # type: ignore[arg-type]
    assert tooltip._id_after is None


def test_programmatic_focus_does_not_schedule_tooltip_at_startup() -> None:
    widget = _FakeWidget()
    tooltip = ToolTip(widget, "Help", delay=25)  # type: ignore[arg-type]

    tooltip._on_focus_in(None)  # type: ignore[arg-type]

    assert widget.after_calls == []
    assert tooltip._has_focus is False


def test_keyboard_focus_schedules_and_focus_out_hides_without_pointer() -> None:
    widget = _FakeWidget()
    tooltip = ToolTip(widget, "Help", delay=25)  # type: ignore[arg-type]

    tooltip._mark_keyboard_modality(None)  # type: ignore[arg-type]
    tooltip._on_focus_in(None)  # type: ignore[arg-type]
    assert tooltip._has_focus is True
    assert len(widget.after_calls) == 1

    tooltip._on_focus_out(None)  # type: ignore[arg-type]
    assert tooltip._has_focus is False
    assert tooltip._id_after is None


def test_pointer_click_clears_keyboard_focus_ownership() -> None:
    widget = _FakeWidget()
    tooltip = ToolTip(widget, "Help", delay=25)  # type: ignore[arg-type]

    tooltip._mark_keyboard_modality(None)  # type: ignore[arg-type]
    tooltip._on_focus_in(None)  # type: ignore[arg-type]
    tooltip._on_button_press(None)  # type: ignore[arg-type]
    tooltip._on_leave(None)  # type: ignore[arg-type]

    assert tooltip._has_focus is False
    assert tooltip._id_after is None


class _FakeTipWindow:
    def __init__(self) -> None:
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


def test_pointer_entry_dismisses_keyboard_tooltip_from_another_control() -> None:
    toplevel = _FakeToplevel()
    keyboard_widget = _FakeWidget(toplevel)
    pointer_widget = _FakeWidget(toplevel)
    keyboard_tooltip = ToolTip(keyboard_widget, "Keyboard help", delay=25)  # type: ignore[arg-type]
    pointer_tooltip = ToolTip(pointer_widget, "Pointer help", delay=25)  # type: ignore[arg-type]
    tipwindow = _FakeTipWindow()
    keyboard_tooltip._tipwindow = tipwindow  # type: ignore[assignment]
    keyboard_tooltip._has_focus = True
    setattr(toplevel, "_active_tooltip", keyboard_tooltip)

    pointer_tooltip._on_enter(None)  # type: ignore[arg-type]

    assert tipwindow.destroyed is True
    assert keyboard_tooltip._tipwindow is None
    assert getattr(toplevel, "_active_tooltip", None) is None
    assert pointer_tooltip._has_focus is False
    assert pointer_tooltip._pointer_inside is True
    assert len(pointer_widget.after_calls) == 1
