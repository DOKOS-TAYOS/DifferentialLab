"""Tests for Tk/matplotlib embedding helpers."""

from __future__ import annotations

from unittest.mock import patch

from matplotlib.figure import Figure

from frontend import plot_embed


class _FakeWidget:
    def __init__(self, width: int = 1, height: int = 1) -> None:
        self.after_calls: list[tuple[int, object]] = []
        self.cancelled: list[str] = []
        self.exists = True
        self.width = width
        self.height = height

    def after(self, delay_ms: int, callback: object) -> str:
        job_id = f"job-{len(self.after_calls) + 1}"
        self.after_calls.append((delay_ms, callback))
        return job_id

    def after_cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)

    def winfo_exists(self) -> bool:
        return self.exists

    def winfo_width(self) -> int:
        return self.width

    def winfo_height(self) -> int:
        return self.height


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
        self.draw_sizes: list[tuple[float, float]] = []

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
        width, height = self.figure.get_size_inches()
        self.draw_sizes.append((float(width), float(height)))

    def draw_idle(self) -> None:
        self.draw_calls += 1


class _FakePackedWidget:
    def __init__(self) -> None:
        self.config_calls: list[dict[str, object]] = []
        self.pack_calls: list[dict[str, object]] = []

    def config(self, **kwargs: object) -> None:
        self.config_calls.append(kwargs)

    def pack(self, **kwargs: object) -> None:
        self.pack_calls.append(kwargs)


class _FakeEmbeddedCanvas:
    def __init__(self, figure: Figure, master: object) -> None:
        self.figure = figure
        self.master = master
        self._widget = _FakePackedWidget()
        self.draw_calls = 0

    def get_tk_widget(self) -> _FakePackedWidget:
        return self._widget

    def draw(self) -> None:
        self.draw_calls += 1


class _FakeNavigationToolbar:
    def __init__(self, canvas: object, parent: object) -> None:
        self.canvas = canvas
        self.parent = parent
        self.update_calls = 0
        self.pack_calls: list[dict[str, object]] = []

    def update(self) -> None:
        self.update_calls += 1

    def pack(self, **kwargs: object) -> None:
        self.pack_calls.append(kwargs)


class _FakeFrame:
    def __init__(self, parent: object) -> None:
        self.parent = parent
        self.pack_calls: list[dict[str, object]] = []

    def pack(self, **kwargs: object) -> None:
        self.pack_calls.append(kwargs)


class _FakeVariable:
    def __init__(self, value: object = None) -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class _FakeParent:
    def winfo_toplevel(self) -> _FakeParent:
        return self


class _FakeControl:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs

    def pack(self, **kwargs: object) -> None:
        pass


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


def test_embed_plot_in_tk_accepts_runtime_figure() -> None:
    figure = Figure()
    parent = object()

    with (
        patch(
            "matplotlib.backends.backend_tkagg.FigureCanvasTkAgg",
            _FakeEmbeddedCanvas,
        ),
        patch.object(plot_embed, "_bind_resize_handler") as bind_resize_handler,
    ):
        canvas = plot_embed.embed_plot_in_tk(figure, parent, toolbar=False)

    assert isinstance(canvas, _FakeEmbeddedCanvas)
    assert canvas.figure is figure
    assert canvas.master is parent
    assert canvas.draw_calls == 1
    bind_resize_handler.assert_called_once_with(canvas, figure)


def test_embed_animation_plot_in_tk_accepts_runtime_figure() -> None:
    figure = Figure()
    parent = _FakeParent()

    with (
        patch(
            "matplotlib.backends.backend_tkagg.FigureCanvasTkAgg",
            _FakeEmbeddedCanvas,
        ),
        patch(
            "matplotlib.backends._backend_tk.NavigationToolbar2Tk",
            _FakeNavigationToolbar,
        ),
        patch.object(plot_embed.ttk, "Frame", _FakeFrame),
        patch.object(plot_embed.tk, "IntVar", _FakeVariable),
        patch.object(plot_embed.tk, "StringVar", _FakeVariable),
        patch.object(plot_embed, "_bind_resize_handler") as bind_resize_handler,
    ):
        canvas = plot_embed.embed_animation_plot_in_tk(figure, parent)

    assert isinstance(canvas, _FakeEmbeddedCanvas)
    assert canvas.figure is figure
    assert canvas.draw_calls == 1
    assert callable(getattr(canvas, "_stop_animation"))
    bind_resize_handler.assert_called_once_with(canvas, figure)


def test_embed_animation_plot_uses_frame_label_with_legacy_fallback() -> None:
    figure = Figure()
    figure._animation_update = lambda _index: None  # type: ignore[attr-defined]
    figure._animation_n_points = 2  # type: ignore[attr-defined]
    figure._animation_frame_label = "z"  # type: ignore[attr-defined]
    parent = _FakeParent()
    labels: list[str] = []

    class _Label(_FakeControl):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            labels.append(str(kwargs.get("text", "")))

    with (
        patch(
            "matplotlib.backends.backend_tkagg.FigureCanvasTkAgg",
            _FakeEmbeddedCanvas,
        ),
        patch(
            "matplotlib.backends._backend_tk.NavigationToolbar2Tk",
            _FakeNavigationToolbar,
        ),
        patch.object(plot_embed.ttk, "Frame", _FakeFrame),
        patch.object(plot_embed.ttk, "Label", _Label),
        patch.object(plot_embed.ttk, "Scale", _FakeControl),
        patch.object(plot_embed.ttk, "Entry", _FakeControl),
        patch.object(plot_embed.ttk, "Button", _FakeControl),
        patch.object(plot_embed.tk, "IntVar", _FakeVariable),
        patch.object(plot_embed.tk, "StringVar", _FakeVariable),
        patch.object(plot_embed, "_bind_resize_handler"),
    ):
        embed_animation_plot_in_tk = plot_embed.embed_animation_plot_in_tk
        embed_animation_plot_in_tk(figure, parent)

    assert "z:" in labels

    fallback = Figure()
    fallback._animation_update = lambda _index: None  # type: ignore[attr-defined]
    fallback._animation_n_points = 1  # type: ignore[attr-defined]
    labels.clear()
    with (
        patch(
            "matplotlib.backends.backend_tkagg.FigureCanvasTkAgg",
            _FakeEmbeddedCanvas,
        ),
        patch(
            "matplotlib.backends._backend_tk.NavigationToolbar2Tk",
            _FakeNavigationToolbar,
        ),
        patch.object(plot_embed.ttk, "Frame", _FakeFrame),
        patch.object(plot_embed.ttk, "Label", _Label),
        patch.object(plot_embed.ttk, "Scale", _FakeControl),
        patch.object(plot_embed.ttk, "Entry", _FakeControl),
        patch.object(plot_embed.ttk, "Button", _FakeControl),
        patch.object(plot_embed.tk, "IntVar", _FakeVariable),
        patch.object(plot_embed.tk, "StringVar", _FakeVariable),
        patch.object(plot_embed, "_bind_resize_handler"),
    ):
        embed_animation_plot_in_tk(fallback, parent)

    assert "x:" in labels


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


def test_replace_plot_in_tk_matches_existing_canvas_dimensions() -> None:
    widget = _FakeWidget(width=1200, height=600)
    old_fig = Figure(figsize=(12, 6), dpi=100)
    new_fig = Figure(figsize=(6, 4), dpi=100)
    canvas = _FakeCanvas(widget, old_fig)

    with (
        patch.object(plot_embed, "_bind_resize_handler"),
        patch("frontend.plot_embed.embed_plot_in_tk") as embed_plot,
        patch("matplotlib.pyplot.close"),
    ):
        reused = plot_embed.replace_plot_in_tk(
            new_fig,
            parent=object(),
            current_canvas=canvas,
        )

    assert reused is canvas
    assert new_fig.get_size_inches().tolist() == [12.0, 6.0]
    assert canvas.draw_sizes == [(12.0, 6.0)]
    assert canvas.toolbar.canvas is canvas
    assert canvas.toolbar.update_calls == 1
    embed_plot.assert_not_called()
