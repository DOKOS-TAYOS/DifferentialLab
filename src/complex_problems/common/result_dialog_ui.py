"""Shared UI helpers for complex-problem result dialogs."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Sequence


def close_embedded_figure(canvas: object | None) -> None:
    """Close a matplotlib figure referenced by an embedded Tk canvas."""
    if canvas is None:
        return

    figure = getattr(canvas, "figure", None)
    if figure is None:
        return

    import matplotlib.pyplot as plt

    try:
        plt.close(figure)
    except Exception:
        return


def close_embedded_figures(owner: object, canvas_attrs: Sequence[str]) -> None:
    """Close all embedded figures referenced by the given owner attributes."""
    for canvas_attr in canvas_attrs:
        close_embedded_figure(getattr(owner, canvas_attr, None))


def reset_embedded_animation(frame: tk.Misc, canvas: object | None) -> None:
    """Close the current animation figure and clear the target frame."""
    close_embedded_figure(canvas)
    for child in frame.winfo_children():
        child.destroy()
