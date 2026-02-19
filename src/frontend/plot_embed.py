"""Tkinter embedding utilities for matplotlib figures."""

from __future__ import annotations

import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
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
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    if toolbar:
        tb = NavigationToolbar2Tk(canvas, parent)
        tb.update()
        tb.pack(fill=tk.X)

    return canvas
