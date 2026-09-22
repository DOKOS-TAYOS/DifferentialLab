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

        self._bind_input_modality()

        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_button_press, add="+")
        widget.bind("<FocusIn>", self._on_focus_in, add="+")
        widget.bind("<FocusOut>", self._on_focus_out, add="+")
        widget.bind("<Destroy>", self._on_destroy, add="+")

    def _bind_input_modality(self) -> None:
        """Track keyboard versus pointer intent once per containing window."""
        try:
            toplevel = self.widget.winfo_toplevel()
        except (AttributeError, tk.TclError):
            return
        if getattr(toplevel, "_tooltip_modality_bound", False):
            return
        setattr(toplevel, "_tooltip_modality", "unknown")
        setattr(toplevel, "_tooltip_modality_bound", True)
        toplevel.bind("<KeyPress>", self._mark_keyboard_modality, add="+")
        toplevel.bind("<ButtonPress>", self._mark_pointer_modality, add="+")

    def _input_modality(self) -> str:
        """Return the current input modality for this tooltip's window."""
        try:
            return str(getattr(self.widget.winfo_toplevel(), "_tooltip_modality", "unknown"))
        except (AttributeError, tk.TclError):
            return "unknown"

    def _dismiss_other_tooltip(self) -> None:
        """Ensure this window never displays more than one contextual tooltip."""
        try:
            toplevel = self.widget.winfo_toplevel()
        except (AttributeError, tk.TclError):
            return
        active = getattr(toplevel, "_active_tooltip", None)
        if isinstance(active, ToolTip) and active is not self:
            active._cancel_pending()
            active._hide()

    def _mark_keyboard_modality(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        """Record keyboard activity without changing the focused tooltip."""
        try:
            setattr(self.widget.winfo_toplevel(), "_tooltip_modality", "keyboard")
        except (AttributeError, tk.TclError):
            pass

    def _mark_pointer_modality(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        """Record pointer activity for subsequent focus changes."""
        try:
            setattr(self.widget.winfo_toplevel(), "_tooltip_modality", "pointer")
        except (AttributeError, tk.TclError):
            pass

    def _on_button_press(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        """Make a click-owned focus transition unable to keep a tooltip alive."""
        self._mark_pointer_modality(_event)
        self._has_focus = False

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
        self._mark_pointer_modality(_event)
        self._has_focus = False
        self._dismiss_other_tooltip()
        self._pointer_inside = True
        self._schedule()

    def _on_leave(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        self._pointer_inside = False
        if not self._has_focus:
            self._cancel_pending()
            self._hide()

    def _on_focus_in(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        self._has_focus = self._input_modality() == "keyboard"
        if self._has_focus:
            self._schedule()
        elif not self._pointer_inside:
            self._cancel_pending()
            self._hide()

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

        self._dismiss_other_tooltip()
        try:
            toplevel = self.widget.winfo_toplevel()
        except (AttributeError, tk.TclError):
            toplevel = None

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
        if toplevel is not None:
            setattr(toplevel, "_active_tooltip", self)

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
        try:
            toplevel = self.widget.winfo_toplevel()
        except (AttributeError, tk.TclError):
            return
        if getattr(toplevel, "_active_tooltip", None) is self:
            setattr(toplevel, "_active_tooltip", None)
