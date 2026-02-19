"""Keyboard navigation utilities for Tkinter dialogs."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Sequence
from typing import Any, Optional


def setup_arrow_enter_navigation(
    widgets_grid: Sequence[Sequence[Any]],
    on_enter: Optional[Callable[[Any, tk.Event], bool]] = None,  # type: ignore[type-arg]
) -> None:
    """Set up arrow-key and Enter navigation on a 2-D grid of widgets.

    Args:
        widgets_grid: Rows of widgets (``None`` cells are skipped).
        on_enter: Optional callback ``(widget, event) -> handled``.
            If it returns ``True`` the default invoke is skipped.
    """
    grid: dict[tuple[int, int], Any] = {}
    for r, row in enumerate(widgets_grid):
        for c, w in enumerate(row):
            if w is not None:
                grid[(r, c)] = w

    if not grid:
        return

    max_r = max(r for r, _ in grid)
    max_c = max(c for _, c in grid)

    def _focus_at(nr: int, nc: int) -> None:
        w = grid.get((nr, nc))
        if w is not None:
            w.focus_set()

    def _move(event: tk.Event, dr: int, dc: int) -> str:  # type: ignore[type-arg]
        current = event.widget
        for (r, c), w in grid.items():
            if w is current:
                nr, nc = r + dr, c + dc
                nr = max(0, min(nr, max_r))
                nc = max(0, min(nc, max_c))
                _focus_at(nr, nc)
                return "break"
        return "break"

    def _invoke_focused(event: tk.Event) -> str:  # type: ignore[type-arg]
        w = event.widget
        if on_enter is not None and on_enter(w, event):
            return "break"
        invoke = getattr(w, "invoke", None)
        if callable(invoke):
            try:
                invoke()
            except tk.TclError:
                pass
        return "break"

    for (_r, _c), w in grid.items():
        w.bind("<Return>", _invoke_focused)
        w.bind("<KP_Enter>", _invoke_focused)
        w.bind("<Left>", lambda e, dr=0, dc=-1: _move(e, dr, dc))
        w.bind("<Right>", lambda e, dr=0, dc=1: _move(e, dr, dc))
        w.bind("<Up>", lambda e, dr=-1, dc=0: _move(e, dr, dc))
        w.bind("<Down>", lambda e, dr=1, dc=0: _move(e, dr, dc))
