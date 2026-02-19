"""Reusable scrollable frame widget with cross-platform mousewheel and keyboard support."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

_REFRESH_DELAY_MS = 50


class ScrollableFrame(ttk.Frame):
    """A frame that wraps a Canvas + Scrollbar + inner Frame.

    Children should be packed/gridded inside ``self.inner``.

    Args:
        parent: Parent widget.
        **kwargs: Extra keyword arguments forwarded to the outer ``ttk.Frame``.
    """

    def __init__(self, parent: tk.Widget, **kwargs) -> None:  # type: ignore[type-arg]
        super().__init__(parent, **kwargs)

        self._canvas = tk.Canvas(self, highlightthickness=0)
        self._scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL,
                                        command=self._canvas.yview)
        self.inner = ttk.Frame(self._canvas)

        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self.inner, anchor=tk.NW,
        )

        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        self._bind_mousewheel_recursive(self)
        self._pending_refresh = False

    def apply_bg(self, bg: str) -> None:
        """Set the canvas background to match the theme."""
        self._canvas.configure(bg=bg)

    def refresh_scroll_region(self) -> None:
        """Force-update the scroll region after dynamic content changes."""
        self._canvas.update_idletasks()
        bbox = self._canvas.bbox("all")
        if bbox:
            self._canvas.configure(scrollregion=bbox)

    def _schedule_refresh(self) -> None:
        """Coalesce multiple layout events into a single deferred update."""
        if not self._pending_refresh:
            self._pending_refresh = True
            self._canvas.after(_REFRESH_DELAY_MS, self._do_refresh)

    def _do_refresh(self) -> None:
        self._pending_refresh = False
        if self._canvas.winfo_exists():
            bbox = self._canvas.bbox("all")
            if bbox:
                self._canvas.configure(scrollregion=bbox)

    def _on_inner_configure(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        self._schedule_refresh()

    def _on_canvas_configure(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        self._canvas.itemconfig(self._canvas_window, width=event.width)
        self._schedule_refresh()

    def _on_mousewheel(self, event: tk.Event) -> str:  # type: ignore[type-arg]
        if self._canvas.winfo_exists():
            if hasattr(event, "delta") and event.delta != 0:
                self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif getattr(event, "num", 0) == 5:
                self._canvas.yview_scroll(1, "units")
            elif getattr(event, "num", 0) == 4:
                self._canvas.yview_scroll(-1, "units")
        return "break"

    def _bind_mousewheel_recursive(self, widget: tk.Widget) -> None:
        widget.bind("<MouseWheel>", self._on_mousewheel)
        widget.bind("<Button-4>", self._on_mousewheel)
        widget.bind("<Button-5>", self._on_mousewheel)
        for child in widget.winfo_children():
            self._bind_mousewheel_recursive(child)

    def bind_new_children(self) -> None:
        """Re-bind mousewheel on all descendants (call after adding widgets)."""
        self._bind_mousewheel_recursive(self)
