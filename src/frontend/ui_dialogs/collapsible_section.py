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
        arrow_var = tk.StringVar(value=EXPANDED if expanded else COLLAPSED)

        wrapper = ttk.Frame(parent)
        wrapper.pack(fill=tk.X, pady=(pad // 2, 0))

        header = ttk.Frame(wrapper, style="SectionHeader.TFrame")
        header.configure(cursor="hand2")
        header.pack(fill=tk.X)

        arrow_lbl = ttk.Label(
            header, textvariable=arrow_var, style="SectionHeader.TLabel",
        )
        arrow_lbl.pack(side=tk.LEFT, padx=(10, 6), pady=8)

        title_lbl = ttk.Label(
            header, text=title, style="SectionHeader.TLabel",
        )
        title_lbl.pack(side=tk.LEFT, pady=8)

        self.content = ttk.Frame(wrapper, padding=(16, 4, 4, 8))

        if expanded:
            self.content.pack(fill=tk.X)

        def toggle(_e: tk.Event | None = None) -> None:  # type: ignore[type-arg]
            if self.content.winfo_manager():
                self.content.pack_forget()
                arrow_var.set(COLLAPSED)
            else:
                self.content.pack(fill=tk.X)
                arrow_var.set(EXPANDED)
                scroll.bind_new_children()
            wrapper.after(_REFRESH_DELAY_MS, scroll.refresh_scroll_region)

        for w in (header, arrow_lbl, title_lbl):
            w.bind("<Button-1>", toggle)
