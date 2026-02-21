"""Tkinter embedding utilities for matplotlib figures."""

from __future__ import annotations

import tkinter as tk
import warnings
from tkinter import ttk
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure


def embed_animation_plot_in_tk(
    fig: Figure,
    parent: tk.Widget,
    *,
    on_export_mp4: Callable[[float], None] | None = None,
) -> FigureCanvasTkAgg:
    """Embed an animation figure with Scale, Play button, duration entry, and Export MP4.

    The figure must have _animation_update(idx) and _animation_n_points attributes.
    Duration (seconds) controls both playback speed and MP4 export length.

    Args:
        fig: Matplotlib figure from create_vector_animation_plot.
        parent: Tkinter parent widget.
        on_export_mp4: Optional callback(duration_seconds) when user clicks Export MP4.

    Returns:
        The canvas object.
    """
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    update_fn = getattr(fig, "_animation_update", None)
    n_points = getattr(fig, "_animation_n_points", 0)
    initial_idx = getattr(fig, "_animation_initial_index", 0)

    top_frame = ttk.Frame(parent)
    top_frame.pack(fill=tk.BOTH, expand=True)

    canvas = FigureCanvasTkAgg(fig, master=top_frame)
    tb = NavigationToolbar2Tk(canvas, top_frame)
    tb.update()
    tb.pack(side=tk.BOTTOM, fill=tk.X)

    widget = canvas.get_tk_widget()
    widget.config(width=1, height=1)
    widget.pack(fill=tk.BOTH, expand=True)

    def _on_resize(_event: object, _fig: object = fig, _canvas: object = canvas) -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                _fig.tight_layout()  # type: ignore[union-attr]
            _canvas.draw_idle()  # type: ignore[union-attr]
        except Exception:
            pass

    canvas.mpl_connect("resize_event", _on_resize)
    canvas.draw()

    ctrl_frame = ttk.Frame(parent)
    ctrl_frame.pack(fill=tk.X, pady=(4, 0))

    _play_job: str | None = None
    scale_var = tk.IntVar(value=initial_idx)
    duration_var = tk.StringVar(value="10")
    root = parent.winfo_toplevel()

    def _get_duration_sec() -> float:
        try:
            d = float(duration_var.get())
            return max(0.5, d)
        except (ValueError, TypeError):
            return 10.0

    def _on_scale_change(v: str) -> None:
        idx = int(float(v))
        if update_fn is not None:
            update_fn(idx)

    _MAX_FPS = 30

    def _play_tick() -> None:
        nonlocal _play_job
        try:
            idx = scale_var.get()
        except tk.TclError:
            _play_job = None
            return
        if idx >= n_points - 1:
            _play_job = None
            return
        duration_sec = _get_duration_sec()
        ticks_total = max(1, int(_MAX_FPS * duration_sec))
        step = max(1, n_points // ticks_total)
        new_idx = min(idx + step, n_points - 1)
        scale_var.set(new_idx)
        if update_fn is not None:
            update_fn(new_idx)
        interval_ms = max(34, int(1000.0 / _MAX_FPS))
        _play_job = root.after(interval_ms, _play_tick)

    def _on_play() -> None:
        nonlocal _play_job
        if _play_job is not None:
            try:
                root.after_cancel(_play_job)
            except tk.TclError:
                pass
            _play_job = None
        if scale_var.get() >= n_points - 1:
            scale_var.set(0)
            if update_fn is not None:
                update_fn(0)
        _play_tick()

    def _on_stop() -> None:
        nonlocal _play_job
        if _play_job is not None:
            try:
                root.after_cancel(_play_job)
            except tk.TclError:
                pass
            _play_job = None

    if update_fn is not None and n_points > 0:
        scale = ttk.Scale(
            ctrl_frame,
            from_=0,
            to=max(1, n_points - 1),
            variable=scale_var,
            orient=tk.HORIZONTAL,
            command=_on_scale_change,
        )
        ttk.Label(ctrl_frame, text="x:").pack(side=tk.LEFT, padx=(0, 4))
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        ttk.Label(ctrl_frame, text="Duración (s):").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(ctrl_frame, textvariable=duration_var, width=14).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            ctrl_frame, text="▶", width=3, style="Small.TButton",
            command=_on_play,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            ctrl_frame, text="■", width=3, style="Small.TButton",
            command=_on_stop,
        ).pack(side=tk.LEFT, padx=2)

    if on_export_mp4 is not None:
        btn = ttk.Button(
            ctrl_frame,
            text="Export MP4",
            command=lambda: on_export_mp4(_get_duration_sec()),
        )
        btn.pack(side=tk.RIGHT, padx=4)

    return canvas


def embed_plot_in_tk(
    fig: Figure,
    parent: tk.Widget,
    toolbar: bool = True,
) -> FigureCanvasTkAgg:
    """Embed a matplotlib figure in a Tkinter parent widget.

    Args:
        fig: Figure to embed.
        parent: Tkinter parent (Frame, Toplevel, etc.).
        toolbar: Whether to add the navigation toolbar.

    Returns:
        The canvas object.
    """
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    canvas = FigureCanvasTkAgg(fig, master=parent)

    if toolbar:
        tb = NavigationToolbar2Tk(canvas, parent)
        tb.update()
        tb.pack(side=tk.BOTTOM, fill=tk.X)

    widget = canvas.get_tk_widget()
    # Override the natural size so tkinter can freely size the widget from
    # available space; FigureCanvasTkAgg redraws at the actual allocated size.
    widget.config(width=1, height=1)
    widget.pack(fill=tk.BOTH, expand=True)

    def _on_resize(_event: object, _fig: object = fig, _canvas: object = canvas) -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                _fig.tight_layout()  # type: ignore[union-attr]
            _canvas.draw_idle()  # type: ignore[union-attr]
        except Exception:
            pass

    canvas.mpl_connect("resize_event", _on_resize)
    canvas.draw()

    return canvas
