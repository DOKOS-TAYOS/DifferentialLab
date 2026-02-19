"""Window management utilities for Tkinter."""

from __future__ import annotations

import tkinter as tk


def center_window(
    window: tk.Tk | tk.Toplevel,
    width: int | None = None,
    height: int | None = None,
    *,
    preserve_size: bool = False,
    max_width_ratio: float = 0.85,
    max_height_ratio: float = 0.85,
) -> None:
    """Center *window* on the screen.

    When *preserve_size* is ``True`` the window is sized to its requested
    (content-driven) dimensions instead of the supplied *width*/*height*.
    Maximum dimensions are clamped to *max_width_ratio* / *max_height_ratio*
    of the screen.

    Args:
        window: The Tk or Toplevel window.
        width: Desired minimum width (ignored when *preserve_size* is set).
        height: Desired minimum height (ignored when *preserve_size* is set).
        preserve_size: Use the widget-requested size instead of explicit dims.
        max_width_ratio: Max fraction of screen width.
        max_height_ratio: Max fraction of screen height.
    """
    window.update_idletasks()
    screen_w = window.winfo_screenwidth()
    screen_h = window.winfo_screenheight()

    max_w = int(screen_w * max_width_ratio)
    max_h = int(screen_h * max_height_ratio)

    if preserve_size:
        w = max(1, window.winfo_reqwidth())
        h = max(1, window.winfo_reqheight())
    else:
        w = width or max(1, window.winfo_reqwidth())
        h = height or max(1, window.winfo_reqheight())

    w = min(w, max_w)
    h = min(h, max_h)

    x = max(0, (screen_w - w) // 2)
    y = max(0, (screen_h - h) // 2)
    window.geometry(f"{w}x{h}+{x}+{y}")
    window.resizable(False, False)


def make_modal(dialog: tk.Toplevel, parent: tk.Tk | tk.Toplevel) -> None:
    """Make a Toplevel dialog modal relative to *parent*.

    Args:
        dialog: The dialog window.
        parent: The parent window to block.
    """
    dialog.transient(parent)
    dialog.grab_set()
    dialog.focus_force()


def on_close_destroy(window: tk.Tk | tk.Toplevel) -> None:
    """Bind the window-close event to destroy the window.

    Args:
        window: The window to configure.
    """
    window.protocol("WM_DELETE_WINDOW", window.destroy)
