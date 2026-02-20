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
    resizable: bool = False,
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
        resizable: Whether the window can be resized by the user.
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
    window.resizable(resizable, resizable)


def fit_and_center(
    window: tk.Tk | tk.Toplevel,
    min_width: int = 400,
    min_height: int = 300,
    padding: int = 40,
    *,
    max_ratio: float = 0.9,
    **center_kwargs: object,
) -> None:
    """Size *window* to fit its content (with minimums) and center it.

    Computes dimensions from the window's requested size, clamps to
    ``[min_width, screen * max_ratio]``, then delegates to
    :func:`center_window`.

    Args:
        window: The Tk or Toplevel window.
        min_width: Minimum window width in pixels.
        min_height: Minimum window height in pixels.
        padding: Extra pixels added to the requested size.
        max_ratio: Maximum fraction of the screen for each dimension.
        **center_kwargs: Forwarded to :func:`center_window`.
    """
    window.update_idletasks()
    req_w = window.winfo_reqwidth() + padding
    req_h = window.winfo_reqheight() + padding

    screen_w = window.winfo_screenwidth()
    screen_h = window.winfo_screenheight()

    w = min(max(req_w, min_width), int(screen_w * max_ratio))
    h = min(max(req_h, min_height), int(screen_h * max_ratio))

    center_window(
        window, w, h,
        max_width_ratio=max_ratio,
        max_height_ratio=max_ratio,
        **center_kwargs,
    )


def make_modal(dialog: tk.Toplevel, parent: tk.Tk | tk.Toplevel) -> None:
    """Make a Toplevel dialog modal relative to *parent*.

    Args:
        dialog: The dialog window.
        parent: The parent window to block.
    """
    dialog.transient(parent)
    dialog.grab_set()
    dialog.focus_force()
