"""Tests for shared UI helpers used by complex problem dialogs."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _load_dialog_ui_module() -> Any:
    try:
        return importlib.import_module("complex_problems.common.dialog_ui")
    except ModuleNotFoundError as exc:
        pytest.fail(f"dialog_ui module not yet implemented: {exc}")


class _FakeWidget:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.pack_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def pack(self, *args: Any, **kwargs: Any) -> None:
        self.pack_calls.append((args, kwargs))


class _FakeWindow:
    def __init__(self) -> None:
        self.destroy_calls = 0

    def destroy(self) -> None:
        self.destroy_calls += 1


def _recording_widget_factory(store: list[_FakeWidget]) -> Any:
    def _factory(*args: Any, **kwargs: Any) -> _FakeWidget:
        widget = _FakeWidget(*args, **kwargs)
        store.append(widget)
        return widget

    return _factory


def test_make_labeled_entry_builds_label_and_entry() -> None:
    dialog_ui = _load_dialog_ui_module()
    parent = object()
    variable = object()
    labels: list[_FakeWidget] = []
    entries: list[_FakeWidget] = []

    with (
        patch.object(dialog_ui, "get_font", return_value=("TestFont", 11)),
        patch.object(
            dialog_ui.ttk,
            "Label",
            side_effect=_recording_widget_factory(labels),
        ) as label_ctor,
        patch.object(
            dialog_ui.ttk,
            "Entry",
            side_effect=_recording_widget_factory(entries),
        ) as entry_ctor,
    ):
        entry = dialog_ui.make_labeled_entry(parent, "Frequency", variable, width=9)

    label = labels[0]
    assert label_ctor.call_args.kwargs["text"] == "Frequency:"
    assert label.pack_calls == [((), {"side": "left", "padx": (0, 4)})]

    assert entry is entries[0]
    assert entry_ctor.call_args.kwargs["textvariable"] is variable
    assert entry_ctor.call_args.kwargs["width"] == 9
    assert entry_ctor.call_args.kwargs["font"] == ("TestFont", 11)
    assert entry.pack_calls == [((), {"side": "left", "padx": (0, 12)})]


def test_make_labeled_spinbox_builds_label_and_spinbox() -> None:
    dialog_ui = _load_dialog_ui_module()
    parent = object()
    variable = object()
    labels: list[_FakeWidget] = []
    spinboxes: list[_FakeWidget] = []

    with (
        patch.object(dialog_ui, "get_font", return_value=("TestFont", 11)),
        patch.object(
            dialog_ui.ttk,
            "Label",
            side_effect=_recording_widget_factory(labels),
        ) as label_ctor,
        patch.object(
            dialog_ui.ttk,
            "Spinbox",
            side_effect=_recording_widget_factory(spinboxes),
        ) as spinbox_ctor,
    ):
        spinbox = dialog_ui.make_labeled_spinbox(
            parent,
            "N_x",
            variable,
            from_=8,
            to=64,
            width=7,
        )

    label = labels[0]
    assert label_ctor.call_args.kwargs["text"] == "N_x:"
    assert label.pack_calls == [((), {"side": "left", "padx": (0, 4)})]

    assert spinbox is spinboxes[0]
    assert spinbox_ctor.call_args.kwargs["textvariable"] is variable
    assert spinbox_ctor.call_args.kwargs["from_"] == 8
    assert spinbox_ctor.call_args.kwargs["to"] == 64
    assert spinbox_ctor.call_args.kwargs["width"] == 7
    assert spinbox_ctor.call_args.kwargs["font"] == ("TestFont", 11)
    assert spinbox.pack_calls == [((), {"side": "left", "padx": (0, 12)})]


def test_make_labeled_combo_builds_readonly_combobox() -> None:
    dialog_ui = _load_dialog_ui_module()
    parent = object()
    variable = object()
    labels: list[_FakeWidget] = []
    combos: list[_FakeWidget] = []

    with (
        patch.object(dialog_ui, "get_font", return_value=("TestFont", 11)),
        patch.object(
            dialog_ui.ttk,
            "Label",
            side_effect=_recording_widget_factory(labels),
        ) as label_ctor,
        patch.object(
            dialog_ui.ttk,
            "Combobox",
            side_effect=_recording_widget_factory(combos),
        ) as combo_ctor,
    ):
        combo = dialog_ui.make_labeled_combo(
            parent,
            "Model",
            variable,
            ("steady", "transient"),
            width=13,
        )

    label = labels[0]
    assert label_ctor.call_args.kwargs["text"] == "Model:"
    assert label.pack_calls == [((), {"side": "left", "padx": (0, 4)})]

    assert combo is combos[0]
    assert combo_ctor.call_args.kwargs["textvariable"] is variable
    assert combo_ctor.call_args.kwargs["values"] == ["steady", "transient"]
    assert combo_ctor.call_args.kwargs["state"] == "readonly"
    assert combo_ctor.call_args.kwargs["width"] == 13
    assert combo_ctor.call_args.kwargs["font"] == ("TestFont", 11)
    assert combo.pack_calls == [((), {"side": "left", "padx": (0, 12)})]


def test_run_solver_dialog_shows_invalid_input_error() -> None:
    dialog_ui = _load_dialog_ui_module()
    parent = object()
    window = _FakeWindow()
    solver = MagicMock()
    result_dialog_factory = MagicMock()

    def collect_inputs() -> dict[str, object]:
        raise ValueError("bad input")

    with (
        patch.object(dialog_ui.messagebox, "showerror") as showerror,
        patch.object(dialog_ui, "run_solver_with_loading") as run_loading,
    ):
        dialog_ui.run_solver_dialog(
            parent=parent,
            window=window,
            collect_inputs=collect_inputs,
            solver=solver,
            message="Solving...",
            result_parent=parent,
            result_dialog_factory=result_dialog_factory,
        )

    showerror.assert_called_once_with("Invalid input", "bad input", parent=window)
    run_loading.assert_not_called()
    solver.assert_not_called()
    result_dialog_factory.assert_not_called()
    assert window.destroy_calls == 0


def test_run_solver_dialog_dispatches_solver_and_result_dialog() -> None:
    dialog_ui = _load_dialog_ui_module()
    parent = object()
    window = _FakeWindow()
    solver = MagicMock(return_value="solved")
    result_dialog_factory = MagicMock()

    with patch.object(dialog_ui, "run_solver_with_loading") as run_loading:
        dialog_ui.run_solver_dialog(
            parent=parent,
            window=window,
            collect_inputs=lambda: {"frequency_hz": 1.0e9, "n_theta": 181},
            solver=solver,
            message="Solving antenna radiation...",
            result_parent=parent,
            result_dialog_factory=result_dialog_factory,
        )

    assert window.destroy_calls == 1
    run_loading.assert_called_once()
    kwargs = run_loading.call_args.kwargs
    assert kwargs["parent"] is parent
    assert kwargs["message"] == "Solving antenna radiation..."

    result = kwargs["task"]()
    solver.assert_called_once_with(frequency_hz=1.0e9, n_theta=181)
    assert result == "solved"

    kwargs["on_success"](result)
    result_dialog_factory.assert_called_once_with(parent, result=result)
