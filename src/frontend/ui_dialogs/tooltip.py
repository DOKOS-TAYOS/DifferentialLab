"""Tooltip widget for Tkinter/ttk elements."""

from __future__ import annotations

import tkinter as tk

from config import get_env_from_schema
from frontend.theme import get_font


class ToolTip:
    """Tooltip for any Tkinter widget, available by pointer or keyboard focus.

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
        self._pointer_inside = False
        self._has_focus = False

        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<FocusIn>", self._on_focus_in, add="+")
        widget.bind("<FocusOut>", self._on_focus_out, add="+")
        widget.bind("<Destroy>", self._on_destroy, add="+")

    def _schedule(self) -> None:
        if self._id_after:
            try:
                self.widget.after_cancel(self._id_after)
            except tk.TclError:
                pass
        self._id_after = self.widget.after(self.delay, self._show)

    def _cancel_pending(self) -> None:
        if not self._id_after:
            return
        try:
            self.widget.after_cancel(self._id_after)
        except tk.TclError:
            pass
        self._id_after = None

    def _on_enter(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        self._pointer_inside = True
        self._schedule()

    def _on_leave(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        self._pointer_inside = False
        if not self._has_focus:
            self._cancel_pending()
            self._hide()

    def _on_focus_in(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        self._has_focus = True
        self._schedule()

    def _on_focus_out(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        self._has_focus = False
        if not self._pointer_inside:
            self._cancel_pending()
            self._hide()

    def _show(self) -> None:
        self._id_after = None
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
        base_font = get_font()
        base_size = (
            int(base_font[1]) if len(base_font) > 1 else int(get_env_from_schema("UI_FONT_SIZE"))
        )
        tooltip_font = (base_font[0], max(10, int(round(base_size * 0.75))))

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
            font=tooltip_font,
            wraplength=wraplength,
        )
        label.pack()
        self._tipwindow = tw

    def _on_destroy(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        self._cancel_pending()
        self._hide()

    def _hide(self) -> None:
        if self._tipwindow:
            try:
                self._tipwindow.destroy()
            except tk.TclError:
                pass
            self._tipwindow = None
