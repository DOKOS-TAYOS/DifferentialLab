"""Loading dialog shown while the solver runs."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from config import get_env_from_schema
from frontend.theme import get_font
from frontend.window_utils import center_window


class LoadingDialog:
    """Modal loading overlay shown during solver execution.

    Args:
        parent: Parent window.
        message: Text to display (e.g. "Solving...").
    """

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        *,
        message: str = "Solving...",
    ) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("")
        self.win.resizable(False, False)

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)

        # Remove window decorations for a cleaner overlay look
        self.win.overrideredirect(False)
        self.win.transient(parent)
        self.win.grab_set()

        pad: int = get_env_from_schema("UI_PADDING")
        font_family, font_size = get_font()

        main_frame = ttk.Frame(self.win, padding=pad * 2)
        main_frame.pack(fill=tk.BOTH, expand=True)

        lbl = ttk.Label(
            main_frame,
            text=message,
            font=(font_family, font_size + 2),
        )
        lbl.pack(pady=(0, pad))

        self._progress = ttk.Progressbar(
            main_frame,
            mode="indeterminate",
            length=280,
        )
        self._progress.pack(pady=pad)
        self._progress.start(10)

        self.win.update_idletasks()
        center_window(
            self.win,
            width=320,
            height=120,
            preserve_size=False,
        )

    def destroy(self) -> None:
        """Stop the progress bar and close the dialog."""
        try:
            self._progress.stop()
        except tk.TclError:
            pass
        self.win.grab_release()
        self.win.destroy()
