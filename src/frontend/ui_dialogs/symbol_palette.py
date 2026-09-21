"""Reusable literal-Unicode symbol insertion controls for mathematical editors."""

from __future__ import annotations

import tkinter as tk
import weakref
from tkinter import ttk
from typing import Any

from config import get_env_from_schema
from frontend.theme import get_select_colors
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.window_utils import bind_wraplength

SYMBOLS: tuple[str, ...] = (
    "α",
    "β",
    "γ",
    "δ",
    "ε",
    "ζ",
    "η",
    "θ",
    "λ",
    "μ",
    "ξ",
    "π",
    "ρ",
    "σ",
    "φ",
    "ω",
    "Δ",
    "Σ",
    "Φ",
    "Ω",
)


class SymbolTargetTracker:
    """Track the most recently focused editor without retaining destroyed widgets."""

    def __init__(self) -> None:
        self._last_target: weakref.ReferenceType[Any] | None = None

    def register_target(self, target: Any) -> None:
        """Register an editor through an additive FocusIn binding."""
        target_ref = weakref.ref(target)

        def _remember(_event: object, ref: weakref.ReferenceType[Any] = target_ref) -> None:
            focused = ref()
            if focused is not None:
                self._last_target = ref

        target.bind("<FocusIn>", _remember, add="+")

    def remember_target(self, target: Any) -> None:
        """Mark *target* active; exposed for deterministic event-free tests."""
        self._last_target = weakref.ref(target)

    def insert(self, symbol: str) -> bool:
        """Insert *symbol* literally at the active target cursor and restore focus."""
        target = self._resolve_target()
        if target is None:
            return False
        try:
            target.insert(tk.INSERT, symbol)
            target.focus_set()
        except (AttributeError, RuntimeError, TypeError, tk.TclError):
            self._last_target = None
            return False
        return True

    def _resolve_target(self) -> Any | None:
        """Return the live active target, clearing stale references safely."""
        if self._last_target is None:
            return None
        target = self._last_target()
        if target is None:
            self._last_target = None
            return None
        try:
            if not target.winfo_exists():
                self._last_target = None
                return None
        except (AttributeError, RuntimeError, tk.TclError):
            self._last_target = None
            return None
        return target


class SymbolPalette(ttk.LabelFrame):
    """Compact keyboard-accessible grid that inserts literal Unicode symbols."""

    def __init__(
        self,
        parent: tk.Misc,  # type: ignore[type-arg]
        *,
        tracker: SymbolTargetTracker | None = None,
        columns: int = 7,
        title: str = "Symbols",
    ) -> None:
        super().__init__(parent, text=title, padding=6)
        self.tracker = tracker or SymbolTargetTracker()
        self.buttons: list[ttk.Button] = []
        columns = max(1, columns)
        button_bg: str = get_env_from_schema("UI_BUTTON_BG")
        button_fg: str = get_env_from_schema("UI_BUTTON_FG")
        foreground: str = get_env_from_schema("UI_FOREGROUND")
        focus_bg, _focus_fg = get_select_colors(button_bg, button_fg)
        style = ttk.Style(self)
        style.configure(
            "SymbolFocus.TButton",
            background=focus_bg,
            foreground=foreground,
            padding=(4, 2),
            borderwidth=3,
            relief="sunken",
        )

        grid: list[list[ttk.Button]] = []
        for index, symbol in enumerate(SYMBOLS):
            row_index, column_index = divmod(index, columns)
            if row_index == len(grid):
                grid.append([])
            button = ttk.Button(
                self,
                text=symbol,
                width=2,
                style="Small.TButton",
                takefocus=index == 0,
                command=lambda value=symbol: self.insert_symbol(value),
            )
            button.grid(row=row_index, column=column_index, padx=2, pady=2, sticky="ew")
            button.bind(
                "<FocusIn>",
                lambda _event, target=button: target.configure(style="SymbolFocus.TButton"),
                add="+",
            )
            button.bind(
                "<FocusOut>",
                lambda _event, target=button: target.configure(style="Small.TButton"),
                add="+",
            )
            button.bind("<space>", lambda event, value=symbol: self._insert_from_key(event, value))
            self.buttons.append(button)
            grid[row_index].append(button)
            self.columnconfigure(column_index, weight=1)

        setup_arrow_enter_navigation(grid)
        note_row = len(grid)
        note = ttk.Label(
            self,
            text="Symbols are inserted literally. Use pi for the built-in π constant.",
            style="Small.TLabel",
            justify=tk.LEFT,
        )
        note.grid(row=note_row, column=0, columnspan=columns, sticky="ew", pady=(4, 0))
        bind_wraplength(self, note, pad=12, min_wrap=120)

    def register_target(self, target: Any) -> None:
        """Make a Text/Entry-like editor eligible for palette insertion."""
        self.tracker.register_target(target)

    def insert_symbol(self, symbol: str) -> bool:
        """Insert exactly the displayed symbol into the last active editor."""
        return self.tracker.insert(symbol)

    def _insert_from_key(self, _event: tk.Event, symbol: str) -> str:  # type: ignore[type-arg]
        self.insert_symbol(symbol)
        return "break"
