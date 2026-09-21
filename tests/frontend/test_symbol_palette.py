"""Headless tests for literal symbol insertion and target tracking."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable

from frontend.ui_dialogs.symbol_palette import SYMBOLS, SymbolTargetTracker


class _FakeEditor:
    def __init__(self, value: str = "", cursor: int = 0) -> None:
        self.value = value
        self.cursor = cursor
        self.exists = True
        self.focus_calls = 0
        self.bindings: dict[str, list[Callable[[object], object]]] = {}

    def bind(
        self,
        sequence: str,
        func: Callable[[object], object],
        add: str | bool | None = None,
    ) -> str:
        if add == "+":
            self.bindings.setdefault(sequence, []).append(func)
        else:
            self.bindings[sequence] = [func]
        return "binding-id"

    def insert(self, _index: str, text: str) -> None:
        if not self.exists:
            raise RuntimeError("destroyed")
        self.value = self.value[: self.cursor] + text + self.value[self.cursor :]
        self.cursor += len(text)

    def focus_set(self) -> None:
        self.focus_calls += 1

    def winfo_exists(self) -> int:
        return int(self.exists)

    def focus(self) -> None:
        for callback in self.bindings.get("<FocusIn>", []):
            callback(object())


def test_text_like_target_inserts_at_cursor_and_restores_focus() -> None:
    tracker = SymbolTargetTracker()
    editor = _FakeEditor("alpha + beta", cursor=6)
    tracker.register_target(editor)
    editor.focus()

    assert tracker.insert("ω") is True
    assert editor.value == "alpha ω+ beta"
    assert editor.focus_calls == 1


def test_entry_like_target_preserves_surrounding_text() -> None:
    tracker = SymbolTargetTracker()
    entry = _FakeEditor("=2.0", cursor=0)
    tracker.register_target(entry)
    entry.focus()

    assert tracker.insert("π") is True
    assert entry.value == "π=2.0"


def test_no_target_and_destroyed_target_are_safe() -> None:
    tracker = SymbolTargetTracker()
    assert tracker.insert("α") is False

    editor = _FakeEditor()
    tracker.register_target(editor)
    editor.focus()
    editor.exists = False
    assert tracker.insert("β") is False


def test_last_focused_registered_target_wins() -> None:
    tracker = SymbolTargetTracker()
    first = _FakeEditor("first", cursor=5)
    second = _FakeEditor("second", cursor=6)
    tracker.register_target(first)
    tracker.register_target(second)
    first.focus()
    second.focus()

    assert tracker.insert("γ") is True
    assert first.value == "first"
    assert second.value == "secondγ"


def test_registration_preserves_existing_focus_binding() -> None:
    tracker = SymbolTargetTracker()
    editor = _FakeEditor()
    calls: list[str] = []
    editor.bind("<FocusIn>", lambda _event: calls.append("existing"))

    tracker.register_target(editor)
    editor.focus()

    assert calls == ["existing"]
    assert len(editor.bindings["<FocusIn>"]) == 2


def test_dynamic_replacement_does_not_keep_old_target_usable() -> None:
    tracker = SymbolTargetTracker()
    old = _FakeEditor("old", cursor=3)
    tracker.register_target(old)
    old.focus()
    old.exists = False
    assert tracker.insert("δ") is False

    replacement = _FakeEditor("new", cursor=3)
    tracker.register_target(replacement)
    replacement.focus()
    assert tracker.insert("ε") is True
    assert replacement.value == "newε"

    replacement_ref = weakref.ref(replacement)
    del replacement
    gc.collect()
    assert replacement_ref() is None
    assert tracker.insert("ζ") is False


def test_symbol_set_preserves_literal_omega_and_pi() -> None:
    assert "ω" in SYMBOLS
    assert "π" in SYMBOLS
    assert "pi" not in SYMBOLS
