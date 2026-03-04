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
    y_offset_up: int = 40,
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
        y_offset_up: Pixels to shift the window up from center (avoids bottom overflow).
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
    y = max(0, (screen_h - h) // 2 - y_offset_up)
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
        window,
        w,
        h,
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


def bind_wraplength(
    frame: tk.Widget,
    label_or_labels: tk.Widget | list[tk.Widget],
    pad: int = 20,
    min_wrap: int = 200,
    debounce_ms: int = 50,
) -> None:
    """Bind label(s) wraplength to the width of a frame.

    Automatically adjusts wraplength when the frame is resized, ensuring text
    wraps nicely within the available space. Supports debouncing to avoid
    excessive updates during rapid resize.

    Args:
        frame: The frame whose width determines the wraplength.
        label_or_labels: Single label widget or list of labels to update.
        pad: Padding in pixels to subtract from frame width.
        min_wrap: Minimum wraplength in pixels.
        debounce_ms: Debounce delay for Configure events (0 = no debounce).
    """
    labels = [label_or_labels] if isinstance(label_or_labels, tk.Widget) else list(label_or_labels)

    def _update(event: object | None = None) -> None:
        w = frame.winfo_width()
        if w > 100:
            wrap = max(min_wrap, w - pad)
            for lbl in labels:
                if lbl.winfo_exists():
                    lbl.configure(wraplength=wrap)

    if debounce_ms > 0:
        _job: str | None = None

        def _debounced(event: object | None = None) -> None:
            nonlocal _job
            if _job is not None:
                try:
                    frame.after_cancel(_job)
                except tk.TclError:
                    pass

            def _run() -> None:
                nonlocal _job
                _job = None
                _update(event)

            _job = frame.after(debounce_ms, _run)

        frame.bind("<Configure>", _debounced)
    else:
        frame.bind("<Configure>", _update)

    frame.after(100, _update)
