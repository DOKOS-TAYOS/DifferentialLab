"""Tests for Tk/matplotlib embedding helpers."""

from __future__ import annotations

from unittest.mock import patch

from matplotlib.figure import Figure

from frontend import plot_embed


class _FakeWidget:
    def __init__(self) -> None:
        self.after_calls: list[tuple[int, object]] = []
        self.cancelled: list[str] = []
        self.exists = True

    def after(self, delay_ms: int, callback: object) -> str:
        job_id = f"job-{len(self.after_calls) + 1}"
        self.after_calls.append((delay_ms, callback))
        return job_id

    def after_cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)

    def winfo_exists(self) -> bool:
        return self.exists


class _FakeToolbar:
    def __init__(self) -> None:
        self.canvas: object | None = None
        self.update_calls = 0

    def update(self) -> None:
        self.update_calls += 1


class _FakeCanvas:
    def __init__(self, widget: _FakeWidget, figure: Figure) -> None:
        self._widget = widget
        self.figure = figure
        self.toolbar = _FakeToolbar()
        self._connections: dict[int, tuple[str, object]] = {}
        self.disconnected: list[int] = []
        self.draw_calls = 0

    def get_tk_widget(self) -> _FakeWidget:
        return self._widget

    def mpl_connect(self, event: str, callback: object) -> int:
        handler_id = len(self._connections) + 1
        self._connections[handler_id] = (event, callback)
        return handler_id

    def mpl_disconnect(self, handler_id: int) -> None:
        self.disconnected.append(handler_id)
        self._connections.pop(handler_id, None)

    def draw(self) -> None:
        self.draw_calls += 1

    def draw_idle(self) -> None:
        self.draw_calls += 1


def test_bind_resize_handler_debounces_and_rebinds() -> None:
    widget = _FakeWidget()
    canvas = _FakeCanvas(widget, Figure())

    plot_embed._bind_resize_handler(canvas, canvas.figure)
    first_handler_id = getattr(canvas, "_resize_handler_id")
    first_callback = canvas._connections[first_handler_id][1]

    first_callback(object())
    first_callback(object())

    assert len(widget.after_calls) == 2
    assert widget.cancelled == ["job-1"]

    widget.after_calls[-1][1]()
    assert canvas.draw_calls == 1

    plot_embed._bind_resize_handler(canvas, Figure())
    assert first_handler_id in canvas.disconnected
    assert widget.cancelled == ["job-1"]


def test_replace_plot_in_tk_reuses_existing_canvas() -> None:
    widget = _FakeWidget()
    old_fig = Figure()
    new_fig = Figure()
    canvas = _FakeCanvas(widget, old_fig)

    with (
        patch.object(plot_embed, "_bind_resize_handler") as bind_resize_handler,
        patch("frontend.plot_embed.embed_plot_in_tk") as embed_plot,
        patch("matplotlib.pyplot.close") as close_figure,
    ):
        reused = plot_embed.replace_plot_in_tk(
            new_fig,
            parent=object(),
            current_canvas=canvas,
        )

    assert reused is canvas
    assert canvas.figure is new_fig
    assert canvas.toolbar.canvas is canvas
    assert canvas.toolbar.update_calls == 1
    assert canvas.draw_calls == 1
    bind_resize_handler.assert_called_once_with(canvas, new_fig)
    close_figure.assert_called_once_with(old_fig)
    embed_plot.assert_not_called()
