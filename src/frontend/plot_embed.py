"""Tkinter embedding utilities for matplotlib figures."""

from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure


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
            _fig.tight_layout()  # type: ignore[union-attr]
            _canvas.draw_idle()  # type: ignore[union-attr]
        except Exception:
            pass

    canvas.mpl_connect("resize_event", _on_resize)
    canvas.draw()

    return canvas
