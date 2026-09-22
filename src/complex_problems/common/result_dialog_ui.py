"""Shared UI helpers for complex-problem result dialogs."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from tkinter import ttk

from config import get_env_from_schema
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.window_utils import (
    bind_wraplength,
    calculate_screen_aware_minsize,
    center_window,
)

ADVANCED_RESULT_CLOSE_STYLE = "Secondary.TButton"


@dataclass(frozen=True)
class AdvancedResultSize:
    """Preferred and minimum dimensions for an Advanced result dialog."""

    preferred_width: int
    preferred_height: int
    minimum_width: int = 1000
    minimum_height: int = 650


def calculate_advanced_result_minsize(
    screen_width: int,
    screen_height: int,
    size: AdvancedResultSize,
) -> tuple[int, int]:
    """Return a result minimum size clamped to the usable screen area."""
    return calculate_screen_aware_minsize(
        screen_width,
        screen_height,
        size.minimum_width,
        size.minimum_height,
    )


def normalize_result_summary(summary: str | None) -> str | None:
    """Strip an optional summary and suppress an empty summary area."""
    if summary is None:
        return None
    normalized = summary.strip()
    return normalized or None


def format_result_summary(
    items: Sequence[tuple[str, object]],
    *,
    separator: str = "   •   ",
) -> str:
    """Format already-selected factual result values with human-readable labels."""
    return separator.join(f"{label}: {value}" for label, value in items)


class AdvancedResultShell:
    """Presentation-only shell shared by Advanced result dialogs.

    Problem modules retain ownership of result data, summaries, tabs, plots,
    exports, animation behavior, and lifecycle attributes.
    """

    def __init__(
        self,
        window: tk.Toplevel,
        *,
        title: str,
        summary: str | None,
        close_command: Callable[[], object],
        pad: int | None = None,
    ) -> None:
        self.window = window
        self.pad = pad if pad is not None else int(get_env_from_schema("UI_PADDING"))
        window.configure(bg=get_env_from_schema("UI_BACKGROUND"))

        self.root = ttk.Frame(window, padding=self.pad)
        self.root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(self.root)
        header.pack(fill=tk.X, pady=(0, self.pad))
        ttk.Label(header, text=title, style="Title.TLabel").pack(anchor=tk.W)

        normalized_summary = normalize_result_summary(summary)
        self.summary_label: ttk.Label | None = None
        if normalized_summary is not None:
            self.summary_label = ttk.Label(
                header,
                text=normalized_summary,
                style="Small.TLabel",
                justify=tk.LEFT,
                wraplength=1100,
            )
            self.summary_label.pack(fill=tk.X, anchor=tk.W, pady=(4, 0))
            bind_wraplength(header, self.summary_label, pad=self.pad, min_wrap=240)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.enable_traversal()

        ttk.Separator(self.root).pack(fill=tk.X, pady=(self.pad, self.pad))
        self.footer = ttk.Frame(self.root)
        self.footer.pack(fill=tk.X)
        self.close_button = ttk.Button(
            self.footer,
            text="Close",
            style=ADVANCED_RESULT_CLOSE_STYLE,
            command=close_command,
        )
        self.close_button.pack(side=tk.RIGHT)
        window.protocol("WM_DELETE_WINDOW", close_command)

    def finish(self, size: AdvancedResultSize) -> None:
        """Apply screen-aware sizing and footer keyboard navigation."""
        min_width, min_height = calculate_advanced_result_minsize(
            self.window.winfo_screenwidth(),
            self.window.winfo_screenheight(),
            size,
        )
        self.window.minsize(min_width, min_height)
        center_window(
            self.window,
            size.preferred_width,
            size.preferred_height,
            max_width_ratio=0.96,
            max_height_ratio=0.92,
            resizable=True,
        )
        setup_arrow_enter_navigation([[self.close_button]])


def make_view_controls(parent: ttk.Frame) -> ttk.LabelFrame:
    """Create a compact, consistently labelled row for visual controls."""
    controls = ttk.LabelFrame(parent, text="View controls", padding=(8, 4))
    controls.pack(fill=tk.X, padx=4, pady=4)
    return controls


def close_embedded_figure(canvas: object | None) -> None:
    """Close a matplotlib figure referenced by an embedded Tk canvas."""
    if canvas is None:
        return

    stop_animation = getattr(canvas, "_stop_animation", None)
    if callable(stop_animation):
        try:
            stop_animation()
        except Exception:
            pass

    figure = getattr(canvas, "figure", None)
    if figure is None:
        return

    import matplotlib.pyplot as plt

    try:
        plt.close(figure)
    except Exception:
        return


def close_embedded_figures(owner: object, canvas_attrs: Sequence[str]) -> None:
    """Close all embedded figures referenced by the given owner attributes."""
    for canvas_attr in canvas_attrs:
        close_embedded_figure(getattr(owner, canvas_attr, None))


def reset_embedded_animation(frame: tk.Misc, canvas: object | None) -> None:
    """Close the current animation figure and clear the target frame."""
    close_embedded_figure(canvas)
    for child in frame.winfo_children():
        child.destroy()
