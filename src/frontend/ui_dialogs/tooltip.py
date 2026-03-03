"""Tooltip widget for Tkinter/ttk elements."""

from __future__ import annotations

import tkinter as tk

from config import get_env_from_schema
from frontend.theme import get_font


class ToolTip:
    """Hover tooltip for any Tkinter widget.

    Args:
        widget: The widget to attach the tooltip to.
        text: The tooltip text.
        delay: Delay in milliseconds before showing.
    """

    def __init__(
        self,
        widget: tk.Widget,
        text: str,
        delay: int | None = None,
    ) -> None:
        self.widget = widget
        self.text = text
        self.delay = int(get_env_from_schema("UI_TOOLTIP_DELAY_MS")) if delay is None else delay
        self._tipwindow: tk.Toplevel | None = None
        self._id_after: str | None = None
        widget.bind("<Enter>", self._on_enter)
        widget.bind("<Leave>", self._on_leave)
        widget.bind("<Destroy>", self._on_destroy)

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
        if not self.text or not self.text.strip():
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tooltip_bg: str = get_env_from_schema("UI_BUTTON_BG")
        tooltip_fg: str = get_env_from_schema("UI_FOREGROUND")
        wraplength: int = get_env_from_schema("UI_TOOLTIP_WRAPLENGTH")
        padx: int = get_env_from_schema("UI_TOOLTIP_PADX")
        pady: int = get_env_from_schema("UI_TOOLTIP_PADY")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background=tooltip_bg,
            foreground=tooltip_fg,
            relief=tk.SOLID,
            borderwidth=1,
            padx=padx,
            pady=pady,
            font=get_font(),
            wraplength=wraplength,
        )
        label.pack()
        self._tipwindow = tw

    def _on_destroy(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._id_after:
            try:
                self.widget.after_cancel(self._id_after)
            except tk.TclError:
                pass
            self._id_after = None
        self._hide()

    def _hide(self) -> None:
        if self._tipwindow:
            self._tipwindow.destroy()
            self._tipwindow = None
