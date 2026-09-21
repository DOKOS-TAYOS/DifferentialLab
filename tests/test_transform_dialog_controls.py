"""Tests for transform-specific UI control visibility."""

from __future__ import annotations

from frontend.ui_dialogs.symbol_palette import SymbolTargetTracker
from frontend.ui_dialogs.transform_dialog import TransformDialog
from transforms import TransformKind


class _FakeVar:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


class _FakeFrame:
    def __init__(self, manager: str = "") -> None:
        self.manager = manager
        self.pack_calls: list[dict[str, object]] = []

    def winfo_manager(self) -> str:
        return self.manager

    def pack(self, **kwargs: object) -> None:
        self.manager = "pack"
        self.pack_calls.append(kwargs)

    def pack_forget(self) -> None:
        self.manager = ""


class _FakeTransformEditor:
    def __init__(self, value: str, cursor: int) -> None:
        self.value = value
        self.cursor = cursor
        self.focused = False

    def insert(self, _index: str, text: str) -> None:
        self.value = self.value[: self.cursor] + text + self.value[self.cursor :]

    def focus_set(self) -> None:
        self.focused = True

    def winfo_exists(self) -> int:
        return 1


def test_taylor_options_are_hidden_for_non_taylor_transforms() -> None:
    dialog = TransformDialog.__new__(TransformDialog)
    dialog._transform_var = _FakeVar(TransformKind.ORIGINAL.value)  # type: ignore[assignment]
    dialog._taylor_frame = _FakeFrame("pack")  # type: ignore[assignment]

    dialog._update_transform_options()

    assert dialog._taylor_frame.winfo_manager() == ""


def test_taylor_options_are_shown_for_taylor_transform(monkeypatch) -> None:
    dialog = TransformDialog.__new__(TransformDialog)
    dialog._transform_var = _FakeVar(TransformKind.TAYLOR.value)  # type: ignore[assignment]
    frame = _FakeFrame()
    dialog._taylor_frame = frame  # type: ignore[assignment]
    monkeypatch.setattr(
        "frontend.ui_dialogs.transform_dialog.get_env_from_schema",
        lambda _key: 8,
    )

    dialog._update_transform_options()

    assert frame.winfo_manager() == "pack"
    assert frame.pack_calls == [{"fill": "x", "pady": (0, 8)}]


def test_transform_change_updates_conditional_controls_without_recomputing() -> None:
    dialog = TransformDialog.__new__(TransformDialog)
    calls: list[str] = []
    dialog._update_transform_options = lambda: calls.append("controls")  # type: ignore[method-assign]
    dialog._on_apply = lambda: calls.append("apply")  # type: ignore[method-assign]

    dialog._on_transform_change(object())

    assert calls == ["controls"]


def test_display_change_does_not_recompute() -> None:
    dialog = TransformDialog.__new__(TransformDialog)
    calls: list[str] = []
    dialog._on_apply = lambda: calls.append("apply")  # type: ignore[method-assign]

    dialog._on_display_change(object())

    assert calls == []


def test_update_runs_the_existing_calculation_path() -> None:
    dialog = TransformDialog.__new__(TransformDialog)
    calls: list[str] = []
    dialog._on_apply = lambda: calls.append("apply")  # type: ignore[method-assign]

    dialog._on_update()

    assert calls == ["apply"]


def test_transform_symbol_insertion_keeps_update_explicit() -> None:
    dialog = TransformDialog.__new__(TransformDialog)
    calls: list[str] = []
    dialog._on_apply = lambda: calls.append("apply")  # type: ignore[method-assign]
    dialog._symbol_tracker = SymbolTargetTracker()
    function_editor = _FakeTransformEditor("sin(x)", cursor=0)
    parameter_editor = _FakeTransformEditor("=2", cursor=0)

    dialog._symbol_tracker.remember_target(function_editor)
    assert dialog._symbol_tracker.insert("ω") is True
    dialog._symbol_tracker.remember_target(parameter_editor)
    assert dialog._symbol_tracker.insert("π") is True

    assert function_editor.value == "ωsin(x)"
    assert parameter_editor.value == "π=2"
    assert function_editor.focused is True
    assert parameter_editor.focused is True
    assert calls == []
