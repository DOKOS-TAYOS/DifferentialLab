"""Reusable collapsible section widget for Tkinter dialogs."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from frontend.ui_dialogs.scrollable_frame import ScrollableFrame

COLLAPSED = "\u25b6"
EXPANDED = "\u25bc"

_REFRESH_DELAY_MS = 50


class CollapsibleSection:
    """A header bar that toggles the visibility of an inner content frame.

    Usage::

        section = CollapsibleSection(parent, scroll, "Title", expanded=True)
        ttk.Label(section.content, text="Hello").pack()

    Args:
        parent: Parent widget to pack into.
        scroll: The :class:`ScrollableFrame` that hosts the section (used
            to refresh the scroll region and rebind mousewheel events).
        title: Section header text.
        expanded: Whether the section starts open.
        pad: Vertical padding above the wrapper.
    """

    def __init__(
        self,
        parent: ttk.Frame,
        scroll: ScrollableFrame,
        title: str,
        *,
        expanded: bool = False,
        pad: int = 6,
    ) -> None:
        self._scroll = scroll
        self._title = title
        self._expanded = expanded

        self._wrapper = ttk.Frame(parent)
        self._wrapper.pack(fill=tk.X, pady=(pad // 2, 0))

        self._header = ttk.Button(
            self._wrapper,
            text=self._header_text(),
            style="SectionHeader.TButton",
            command=self.toggle,
            takefocus=True,
        )
        self._header.pack(fill=tk.X)
        for sequence in ("<Return>", "<KP_Enter>", "<space>"):
            self._header.bind(sequence, self._on_key_toggle, add="+")

        self.content = ttk.Frame(self._wrapper, padding=(16, 4, 4, 8))

        if expanded:
            self.content.pack(fill=tk.X)

    def _header_text(self) -> str:
        """Return the accessible header text for the current expansion state."""
        arrow = EXPANDED if self._expanded else COLLAPSED
        return f"{arrow}  {self._title}"

    def _on_key_toggle(self, _event: tk.Event[tk.Misc]) -> str:
        """Toggle once for an explicit keyboard activation sequence."""
        self.toggle()
        return "break"

    def toggle(self) -> None:
        """Show or hide the section body and refresh its scroll container."""
        if self._expanded:
            self.content.pack_forget()
        else:
            self.content.pack(fill=tk.X)
            self._scroll.bind_new_children()

        self._expanded = not self._expanded
        self._header.configure(text=self._header_text())
        self._wrapper.after(_REFRESH_DELAY_MS, self._scroll.refresh_scroll_region)
