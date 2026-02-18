"""Tooltip widget for Tkinter/ttk elements."""

from __future__ import annotations

import tkinter as tk


class ToolTip:
    """Hover tooltip for any Tkinter widget.

    Args:
        widget: The widget to attach the tooltip to.
        text: The tooltip text.
        delay: Delay in milliseconds before showing.
    """

    def __init__(self, widget: tk.Widget, text: str, delay: int = 500) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self._tipwindow: tk.Toplevel | None = None
        self._id_after: str | None = None
        widget.bind("<Enter>", self._on_enter)
        widget.bind("<Leave>", self._on_leave)

    def _on_enter(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        self._id_after = self.widget.after(self.delay, self._show)

    def _on_leave(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._id_after:
            self.widget.after_cancel(self._id_after)
            self._id_after = None
        self._hide()

    def _show(self) -> None:
        if self._tipwindow:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#333333",
            foreground="#EEEEEE",
            relief=tk.SOLID,
            borderwidth=1,
            padx=8,
            pady=4,
            font=("Segoe UI", 10),
            wraplength=350,
        )
        label.pack()
        self._tipwindow = tw

    def _hide(self) -> None:
        if self._tipwindow:
            self._tipwindow.destroy()
            self._tipwindow = None
