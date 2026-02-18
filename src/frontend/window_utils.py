"""Window management utilities for Tkinter."""

from __future__ import annotations

import tkinter as tk


def center_window(window: tk.Tk | tk.Toplevel, width: int, height: int) -> None:
    """Center a window on the screen.

    Args:
        window: The Tk or Toplevel window.
        width: Desired window width.
        height: Desired window height.
    """
    window.update_idletasks()
    screen_w = window.winfo_screenwidth()
    screen_h = window.winfo_screenheight()
    x = (screen_w - width) // 2
    y = (screen_h - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")


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
