"""Tests for shared result-dialog helpers in complex problems."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _load_result_dialog_ui_module() -> Any:
    try:
        return importlib.import_module("complex_problems.common.result_dialog_ui")
    except ModuleNotFoundError as exc:
        pytest.fail(f"result_dialog_ui module not yet implemented: {exc}")


def test_responsive_view_control_rows_preserve_groups() -> None:
    ui = _load_result_dialog_ui_module()
    assert ui.layout_view_control_groups(600, (120, 160, 180)) == ((0, 1, 2),)
    assert ui.layout_view_control_groups(300, (120, 160, 180)) == ((0, 1), (2,))
    assert ui.layout_view_control_groups(300, (420, 80)) == ((0,), (1,))


def test_multi_selector_selection_rules_are_deterministic() -> None:
    ui = _load_result_dialog_ui_module()
    assert ui.layout_view_control_groups(300, (180, 80, 180)) == ((0, 1), (2,))
    assert ui.normalize_multi_selector_indexes(5, (3, 1, 3), allow_empty=True) == (1, 3)
    assert ui.normalize_multi_selector_indexes(5, (), allow_empty=True) == ()
    assert ui.normalize_multi_selector_indexes(5, (), allow_empty=False) == (0,)


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


class _FakeWidget:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.pack_calls: list[dict[str, object]] = []
        self.traversal_enabled = False

    def pack(self, **kwargs: object) -> None:
        self.pack_calls.append(kwargs)

    def enable_traversal(self) -> None:
        self.traversal_enabled = True


class _FakeWindow:
    def __init__(self, *, width: int = 1920, height: int = 1080) -> None:
        self.width = width
        self.height = height
        self.protocols: dict[str, object] = {}
        self.minimum_size: tuple[int, int] | None = None
        self.destroy_calls = 0

    def configure(self, **_kwargs: object) -> None:
        return

    def protocol(self, name: str, callback: object) -> None:
        self.protocols[name] = callback

    def winfo_screenwidth(self) -> int:
        return self.width

    def winfo_screenheight(self) -> int:
        return self.height

    def minsize(self, width: int, height: int) -> None:
        self.minimum_size = (width, height)

    def destroy(self) -> None:
        self.destroy_calls += 1


def test_close_embedded_figure_closes_canvas_figure() -> None:
    result_dialog_ui = _load_result_dialog_ui_module()
    figure = _FakeFigure()
    canvas = _FakeCanvas(figure=figure)

    with patch("matplotlib.pyplot.close") as close_figure:
        result_dialog_ui.close_embedded_figure(canvas)

    close_figure.assert_called_once_with(figure)


def test_close_embedded_figure_stops_pending_animation_before_closing() -> None:
    result_dialog_ui = _load_result_dialog_ui_module()
    canvas = _FakeCanvas(figure=_FakeFigure())
    stop_animation = MagicMock()
    canvas._stop_animation = stop_animation  # type: ignore[attr-defined]

    with patch("matplotlib.pyplot.close"):
        result_dialog_ui.close_embedded_figure(canvas)

    stop_animation.assert_called_once_with()


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


def test_advanced_result_close_is_secondary_and_uses_same_cleanup_callback() -> None:
    result_dialog_ui = _load_result_dialog_ui_module()
    window = _FakeWindow()
    close_callback = MagicMock()
    created_buttons: list[_FakeWidget] = []

    def make_button(*args: object, **kwargs: object) -> _FakeWidget:
        button = _FakeWidget(*args, **kwargs)
        created_buttons.append(button)
        return button

    with (
        patch.object(result_dialog_ui.ttk, "Frame", _FakeWidget),
        patch.object(result_dialog_ui.ttk, "Label", _FakeWidget),
        patch.object(result_dialog_ui.ttk, "Notebook", _FakeWidget),
        patch.object(result_dialog_ui.ttk, "Separator", _FakeWidget),
        patch.object(result_dialog_ui.ttk, "Button", side_effect=make_button),
        patch.object(result_dialog_ui, "bind_wraplength"),
    ):
        shell = result_dialog_ui.AdvancedResultShell(
            window,
            title="Test Results",
            summary="Factual value: 1",
            close_command=close_callback,
            pad=8,
        )

    assert result_dialog_ui.ADVANCED_RESULT_CLOSE_STYLE == "Secondary.TButton"
    assert shell.close_button is created_buttons[0]
    assert created_buttons[0].kwargs["style"] == "Secondary.TButton"
    assert created_buttons[0].kwargs["command"] is close_callback
    assert window.protocols["WM_DELETE_WINDOW"] is close_callback
    assert shell.notebook.traversal_enabled is True


def test_advanced_result_minimum_size_is_screen_aware() -> None:
    result_dialog_ui = _load_result_dialog_ui_module()
    size = result_dialog_ui.AdvancedResultSize(1400, 900, 1000, 650)

    assert result_dialog_ui.calculate_advanced_result_minsize(1920, 1080, size) == (1000, 650)
    assert result_dialog_ui.calculate_advanced_result_minsize(900, 600, size) == (810, 540)


@pytest.mark.parametrize(
    ("summary", "expected"),
    ((None, None), ("", None), ("   ", None), ("  value  ", "value")),
)
def test_optional_result_summary_is_normalized(summary: str | None, expected: str | None) -> None:
    result_dialog_ui = _load_result_dialog_ui_module()

    assert result_dialog_ui.normalize_result_summary(summary) == expected


def test_fput_close_cleans_every_tracked_canvas() -> None:
    module = importlib.import_module("complex_problems.fput_experiment.result_dialog")
    dialog = module.FPUTResultDialog.__new__(module.FPUTResultDialog)
    dialog._canvases = [object(), object(), object()]
    dialog.win = _FakeWindow()

    with patch.object(module, "close_embedded_figure") as close_figure:
        dialog._on_close()

    assert [call.args[0] for call in close_figure.call_args_list] == dialog._canvases
    assert dialog.win.destroy_calls == 1


def test_n_body_close_cleans_every_canvas_attribute() -> None:
    module = importlib.import_module("complex_problems.gravitational_n_body.result_dialog")
    dialog = module.GravitationalNBodyResultDialog.__new__(module.GravitationalNBodyResultDialog)
    dialog.win = _FakeWindow()

    with patch.object(module, "close_embedded_figures") as close_figures:
        dialog._on_close()

    close_figures.assert_called_once_with(
        dialog,
        (
            "_anim_canvas",
            "_trajectory_canvas",
            "_phase_canvas",
            "_energy_canvas",
            "_diagnostics_canvas",
            "_separation_canvas",
        ),
    )
    assert dialog.win.destroy_calls == 1


@pytest.mark.parametrize(
    "module_name",
    (
        "complex_problems.aerodynamics_2d.result_dialog",
        "complex_problems.antenna_radiation.result_dialog",
        "complex_problems.coupled_oscillators.result_dialog",
        "complex_problems.fput_experiment.result_dialog",
        "complex_problems.gravitational_n_body.result_dialog",
        "complex_problems.membrane_2d.result_dialog",
        "complex_problems.nonlinear_waves.result_dialog",
        "complex_problems.pipe_flow.result_dialog",
        "complex_problems.schrodinger_td.result_dialog",
    ),
)
def test_result_dialog_modules_import(module_name: str) -> None:
    importlib.import_module(module_name)
