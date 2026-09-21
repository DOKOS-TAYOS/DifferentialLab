"""Loading dialog shown while a background task runs."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from config import get_env_from_schema
from frontend.window_utils import center_window


class LoadingDialog:
    """Modal progress dialog shown while a background task runs.

    The shared background-task helper does not currently expose safe
    cancellation or determinate progress, so this dialog deliberately uses an
    indeterminate progress bar and does not provide a cancel/close action.

    Args:
        parent: Parent window.
        message: Short description of the operation in progress.
    """

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        *,
        message: str = "Working...",
    ) -> None:
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Working")
        self.win.resizable(False, False)

        bg: str = get_env_from_schema("UI_BACKGROUND")
        self.win.configure(bg=bg)
        self.win.transient(parent)
        self.win.grab_set()
        self.win.protocol("WM_DELETE_WINDOW", self._ignore_close)

        pad: int = get_env_from_schema("UI_PADDING")

        main_frame = ttk.Frame(self.win, padding=(pad * 2, pad * 2))
        main_frame.pack(fill=tk.BOTH, expand=True)

        screen_width = self.win.winfo_screenwidth()
        max_dialog_width = max(280, min(480, int(screen_width * 0.8)))
        wraplength = max(140, max_dialog_width - 4 * pad)

        ttk.Label(
            main_frame,
            text=message,
            style="Subtitle.TLabel",
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=wraplength,
        ).pack(fill=tk.X, pady=(0, pad // 2))

        ttk.Label(
            main_frame,
            text="Please wait while DifferentialLab finishes this operation.",
            style="Small.TLabel",
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=wraplength,
        ).pack(fill=tk.X, pady=(0, pad))

        self._progress = ttk.Progressbar(
            main_frame,
            mode="indeterminate",
            length=min(320, wraplength),
        )
        self._progress.pack(fill=tk.X, pady=(pad // 2, 0))
        self._progress.start(10)

        self.win.update_idletasks()
        requested_width = min(max_dialog_width, max(280, main_frame.winfo_reqwidth() + 2 * pad))
        requested_height = main_frame.winfo_reqheight() + 2 * pad
        center_window(
            self.win,
            width=requested_width,
            height=requested_height,
            max_width_ratio=0.8,
            preserve_size=False,
        )

    def _ignore_close(self) -> None:
        """Keep the modal visible while the non-cancellable task is running."""

    def destroy(self) -> None:
        """Stop the progress bar and close the dialog."""
        try:
            self._progress.stop()
        except tk.TclError:
            pass
        try:
            self.win.grab_release()
        except tk.TclError:
            pass
        try:
            self.win.destroy()
        except tk.TclError:
            pass
