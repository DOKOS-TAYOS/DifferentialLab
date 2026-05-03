"""Tests for shared result-dialog helpers in complex problems."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import patch

import pytest


def _load_result_dialog_ui_module() -> Any:
    try:
        return importlib.import_module("complex_problems.common.result_dialog_ui")
    except ModuleNotFoundError as exc:
        pytest.fail(f"result_dialog_ui module not yet implemented: {exc}")


class _FakeFigure:
    pass


class _FakeCanvas:
    def __init__(self, *, figure: object | None = None) -> None:
        self.figure = figure


class _FakeChild:
    def __init__(self) -> None:
        self.destroy_calls = 0

    def destroy(self) -> None:
        self.destroy_calls += 1


class _FakeFrame:
    def __init__(self, children: list[_FakeChild]) -> None:
        self._children = children

    def winfo_children(self) -> list[_FakeChild]:
        return list(self._children)


def test_close_embedded_figure_closes_canvas_figure() -> None:
    result_dialog_ui = _load_result_dialog_ui_module()
    figure = _FakeFigure()
    canvas = _FakeCanvas(figure=figure)

    with patch("matplotlib.pyplot.close") as close_figure:
        result_dialog_ui.close_embedded_figure(canvas)

    close_figure.assert_called_once_with(figure)


def test_close_embedded_figure_ignores_missing_or_failing_figures() -> None:
    result_dialog_ui = _load_result_dialog_ui_module()

    with patch("matplotlib.pyplot.close", side_effect=RuntimeError("boom")) as close_figure:
        result_dialog_ui.close_embedded_figure(_FakeCanvas(figure=_FakeFigure()))
        result_dialog_ui.close_embedded_figure(None)
        result_dialog_ui.close_embedded_figure(object())

    close_figure.assert_called_once()


def test_close_embedded_figures_reads_multiple_canvas_attrs() -> None:
    result_dialog_ui = _load_result_dialog_ui_module()

    class _Owner:
        def __init__(self) -> None:
            self._anim_canvas = _FakeCanvas(figure=_FakeFigure())
            self._spec_canvas = _FakeCanvas(figure=_FakeFigure())

    owner = _Owner()

    with patch.object(result_dialog_ui, "close_embedded_figure") as close_figure:
        result_dialog_ui.close_embedded_figures(
            owner,
            ("_anim_canvas", "_missing_canvas", "_spec_canvas"),
        )

    assert close_figure.call_count == 3
    assert close_figure.call_args_list[0].args == (owner._anim_canvas,)
    assert close_figure.call_args_list[1].args == (None,)
    assert close_figure.call_args_list[2].args == (owner._spec_canvas,)


def test_reset_embedded_animation_closes_canvas_and_destroys_children() -> None:
    result_dialog_ui = _load_result_dialog_ui_module()
    children = [_FakeChild(), _FakeChild(), _FakeChild()]
    frame = _FakeFrame(children)
    canvas = _FakeCanvas(figure=_FakeFigure())

    with patch.object(result_dialog_ui, "close_embedded_figure") as close_figure:
        result_dialog_ui.reset_embedded_animation(frame, canvas)

    close_figure.assert_called_once_with(canvas)
    assert [child.destroy_calls for child in children] == [1, 1, 1]


@pytest.mark.parametrize(
    "module_name",
    (
        "complex_problems.aerodynamics_2d.result_dialog",
        "complex_problems.antenna_radiation.result_dialog",
        "complex_problems.coupled_oscillators.result_dialog",
        "complex_problems.membrane_2d.result_dialog",
        "complex_problems.nonlinear_waves.result_dialog",
        "complex_problems.pipe_flow.result_dialog",
        "complex_problems.schrodinger_td.result_dialog",
    ),
)
def test_result_dialog_modules_import(module_name: str) -> None:
    importlib.import_module(module_name)
