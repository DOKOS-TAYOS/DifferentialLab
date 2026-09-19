"""Lifecycle regressions for the initial ResultDialog plot render."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

import frontend.ui_dialogs.result_dialog as result_dialog_ui
from transforms import TransformKind


class _FakeVariable:
    def get(self) -> str:
        return "0"


class _FakeChild:
    def __init__(self) -> None:
        self.destroy_calls = 0

    def destroy(self) -> None:
        self.destroy_calls += 1


class _FakeFrame:
    def __init__(self, children: list[_FakeChild]) -> None:
        self.children = children

    def winfo_children(self) -> list[_FakeChild]:
        return self.children


class _FakeCanvas:
    def __init__(self, figure: Figure) -> None:
        self.figure = figure
        self.stop_animation = MagicMock()
        self._stop_animation = self.stop_animation


class _FakeWidget:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def pack(self, **_kwargs: object) -> None:
        pass

    def grid(self, **_kwargs: object) -> None:
        pass

    def grid_propagate(self, _value: bool) -> None:
        pass

    def columnconfigure(self, *_args: object, **_kwargs: object) -> None:
        pass

    def rowconfigure(self, *_args: object, **_kwargs: object) -> None:
        pass

    def configure(self, **_kwargs: object) -> None:
        pass


class _FakeButton(_FakeWidget):
    def __init__(self, *_args: object, **kwargs: object) -> None:
        super().__init__(*_args, **kwargs)
        self.command = kwargs["command"]

    def pack(self, **_kwargs: object) -> None:
        pass

    def focus_set(self) -> None:
        pass


class _FakeScrollableFrame:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.inner = _FakeWidget()

    def apply_bg(self, _background: str) -> None:
        pass

    def pack(self, **_kwargs: object) -> None:
        pass

    def bind_new_children(self) -> None:
        pass


class _FakeWindow:
    def __init__(self) -> None:
        self.destroy_calls = 0

    def destroy(self) -> None:
        self.destroy_calls += 1


def test_result_dialog_sets_geometry_and_materializes_layout_before_first_plot() -> None:
    """Initial plot callbacks run only after the constructed frames are laid out."""
    events: list[str] = []
    state: dict[str, object] = {}
    window = MagicMock()
    window.winfo_screenwidth.return_value = 1920
    window.winfo_screenheight.return_value = 1080
    window.minsize.side_effect = lambda *_: events.append("minsize")

    def build_ui(dialog: result_dialog_ui.ResultDialog) -> None:
        events.append("controls")

    def build_plot_tabs(dialog: result_dialog_ui.ResultDialog) -> None:
        events.append("tabs")
        dialog.plot_frame = object()
        state["dialog"] = dialog

        def render_initial_plot() -> None:
            assert events[-1] == "layout"
            assert dialog.plot_frame is not None
            events.append("render")

        dialog._queue_initial_plot(render_initial_plot)

    def materialize_layout() -> None:
        dialog = state["dialog"]
        assert isinstance(dialog, result_dialog_ui.ResultDialog)
        assert hasattr(dialog, "plot_frame")
        assert events == ["geometry", "minsize", "controls", "tabs"]
        events.append("layout")

    window.update_idletasks.side_effect = materialize_layout
    result = SimpleNamespace(
        metadata={"equation_name": "Simple Harmonic Oscillator"},
        notation=None,
        vector_order=1,
    )

    with (
        patch.object(result_dialog_ui.tk, "Toplevel", return_value=window),
        patch.object(
            result_dialog_ui,
            "center_window",
            side_effect=lambda *_args, **_kwargs: events.append("geometry"),
        ),
        patch.object(result_dialog_ui.ResultDialog, "_build_ui", build_ui),
        patch.object(result_dialog_ui.ResultDialog, "_build_plot_tabs", build_plot_tabs),
        patch.object(result_dialog_ui, "make_modal", side_effect=lambda *_: events.append("modal")),
    ):
        dialog = result_dialog_ui.ResultDialog(MagicMock(), result=result)

    assert events == ["geometry", "minsize", "controls", "tabs", "layout", "render", "modal"]
    assert dialog._initial_plot_callbacks == []


def test_animation_replacement_stops_and_closes_previous_owned_figure() -> None:
    """Repeated animation rebuilds keep only the current figure registered."""
    result = SimpleNamespace(
        x=np.linspace(0.0, 1.0, 8),
        y=np.vstack((np.linspace(0.0, 1.0, 8), np.linspace(1.0, 0.0, 8))),
        vector_order=2,
        vector_components=1,
        metadata={"equation_name": "reproduction"},
    )
    child = _FakeChild()
    dialog = object.__new__(result_dialog_ui.ResultDialog)
    dialog._result = result
    dialog._anim_order_var = _FakeVariable()
    dialog._anim_plot_frame = _FakeFrame([child])
    dialog._get_transform_kind = MagicMock(return_value=TransformKind.ORIGINAL)
    dialog._canvases = []
    dialog._closed = False

    baseline_fignums = set(plt.get_fignums())
    old_figure = plt.figure()
    old_canvas = _FakeCanvas(old_figure)
    dialog._anim_canvas = old_canvas
    dialog._canvases.append(old_canvas)
    created_figures = [old_figure]
    created_canvases: list[_FakeCanvas] = []

    def create_figure(*_args: object, **_kwargs: object) -> Figure:
        figure = plt.figure()
        created_figures.append(figure)
        return figure

    def embed_figure(figure: Figure, _parent: object, **_kwargs: object) -> _FakeCanvas:
        canvas = _FakeCanvas(figure)
        created_canvases.append(canvas)
        return canvas

    try:
        with (
            patch("plotting.create_vector_animation_plot", side_effect=create_figure),
            patch.object(result_dialog_ui, "embed_animation_plot_in_tk", side_effect=embed_figure),
        ):
            dialog._update_animation()
            first_canvas = dialog._anim_canvas
            dialog._update_animation()
            second_canvas = dialog._anim_canvas

        assert old_canvas.stop_animation.call_count == 1
        assert child.destroy_calls == 2
        assert first_canvas is created_canvases[0]
        assert second_canvas is created_canvases[1]
        assert first_canvas is not second_canvas
        assert dialog._canvases == [second_canvas]
        assert second_canvas is not None
        assert set(plt.get_fignums()) == baseline_fignums | {second_canvas.figure.number}
        assert second_canvas.figure.number in plt.get_fignums()
    finally:
        for figure in created_figures:
            plt.close(figure)


def test_result_dialog_close_releases_owned_figures_and_is_idempotent() -> None:
    """Closing a dialog stops animations and closes every current figure once."""
    window = _FakeWindow()
    dialog = object.__new__(result_dialog_ui.ResultDialog)
    dialog.win = window
    dialog._closed = False
    canvases = [_FakeCanvas(plt.figure()), _FakeCanvas(plt.figure()), _FakeCanvas(plt.figure())]
    dialog._canvases = canvases
    figure_numbers = {canvas.figure.number for canvas in canvases}

    dialog._close()
    dialog._close()

    assert figure_numbers.isdisjoint(plt.get_fignums())
    assert all(canvas.stop_animation.call_count == 1 for canvas in canvases)
    assert dialog._canvases == []
    assert window.destroy_calls == 1
    for canvas in canvases:
        plt.close(canvas.figure)


def test_close_button_and_window_manager_use_the_same_cleanup_callback() -> None:
    """The explicit button and title-bar close share ResultDialog._close."""
    buttons: list[_FakeButton] = []

    def make_button(*args: object, **kwargs: object) -> _FakeButton:
        button = _FakeButton(*args, **kwargs)
        buttons.append(button)
        return button

    window = MagicMock()
    window.winfo_screenwidth.return_value = 1920
    window.winfo_screenheight.return_value = 1080
    result = SimpleNamespace(
        metadata={"equation_name": "Simple Harmonic Oscillator"},
        statistics={},
        notation=None,
        vector_order=1,
    )

    with (
        patch.object(result_dialog_ui.tk, "Toplevel", return_value=window),
        patch.object(result_dialog_ui.ttk, "Frame", _FakeWidget),
        patch.object(result_dialog_ui.ttk, "Button", side_effect=make_button),
        patch.object(result_dialog_ui.ttk, "Notebook", _FakeWidget),
        patch.object(result_dialog_ui, "ScrollableFrame", _FakeScrollableFrame),
        patch.object(result_dialog_ui.ResultDialog, "_set_window_geometry"),
        patch.object(result_dialog_ui.ResultDialog, "_build_left_panel"),
        patch.object(result_dialog_ui.ResultDialog, "_build_plot_tabs"),
        patch.object(result_dialog_ui, "setup_arrow_enter_navigation"),
        patch.object(result_dialog_ui, "make_modal"),
    ):
        dialog = result_dialog_ui.ResultDialog(MagicMock(), result=result)

    protocol_callback = window.protocol.call_args.args[1]
    button_callback = buttons[0].command
    assert protocol_callback.__self__ is dialog
    assert button_callback.__self__ is dialog
    assert protocol_callback.__func__ is result_dialog_ui.ResultDialog._close
    assert button_callback.__func__ is result_dialog_ui.ResultDialog._close
