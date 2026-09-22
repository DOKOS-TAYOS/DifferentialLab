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


def layout_view_control_groups(
    available_width: int,
    requested_group_widths: Sequence[int],
    *,
    gap: int = 12,
) -> tuple[tuple[int, ...], ...]:
    """Return responsive rows while keeping each control group indivisible."""
    if available_width <= 0:
        return (tuple(range(len(requested_group_widths))),) if requested_group_widths else ()

    rows: list[list[int]] = []
    row: list[int] = []
    row_width = 0
    for index, requested_width in enumerate(requested_group_widths):
        width = max(1, requested_width)
        next_width = width if not row else row_width + gap + width
        if row and next_width > available_width:
            rows.append(row)
            row = []
            row_width = 0
        row.append(index)
        row_width = width if len(row) == 1 else row_width + gap + width
    if row:
        rows.append(row)
    return tuple(tuple(current_row) for current_row in rows)


def normalize_multi_selector_indexes(
    item_count: int,
    indexes: Sequence[int],
    *,
    allow_empty: bool,
) -> tuple[int, ...]:
    """Normalize selected indexes while preserving item order and empty policy."""
    valid = {index for index in indexes if 0 <= index < item_count}
    if not allow_empty and not valid and item_count:
        valid.add(0)
    return tuple(index for index in range(item_count) if index in valid)


class AdvancedMultiSelector(ttk.Frame):
    """Compact, keyboard-accessible multi-selection control for result views."""

    def __init__(
        self,
        parent: tk.Misc,
        labels: Sequence[str],
        *,
        selected_indexes: Sequence[int] = (),
        allow_empty: bool = True,
        command: Callable[[], object] | None = None,
        compact_threshold: int = 8,
    ) -> None:
        super().__init__(parent)
        self._labels = tuple(labels)
        self._allow_empty = allow_empty
        self._command = command
        self._compact = len(self._labels) > compact_threshold
        self._variables = [tk.BooleanVar(self, value=False) for _ in self._labels]
        self._menu_button: ttk.Menubutton | None = None
        self._menu: tk.Menu | None = None
        if self._compact:
            self._menu = tk.Menu(self, tearoff=False)
            for label, variable in zip(self._labels, self._variables):
                self._menu.add_checkbutton(
                    label=label,
                    variable=variable,
                    command=self._on_change,
                )
            self._menu_button = ttk.Menubutton(self, text="Select items", menu=self._menu)
            self._menu_button.pack(fill=tk.X)
        else:
            for label, variable in zip(self._labels, self._variables):
                ttk.Checkbutton(
                    self,
                    text=label,
                    variable=variable,
                    command=self._on_change,
                ).pack(anchor=tk.W)
        self.set_selected_indexes(selected_indexes, notify=False)

    @property
    def labels(self) -> tuple[str, ...]:
        """Return labels in their original, deterministic order."""
        return self._labels

    def selected_indexes(self) -> tuple[int, ...]:
        """Return selected item indexes in original order."""
        return tuple(index for index, variable in enumerate(self._variables) if variable.get())

    def selected_labels(self) -> tuple[str, ...]:
        """Return selected labels in original order."""
        return tuple(self._labels[index] for index in self.selected_indexes())

    def set_selected_indexes(self, indexes: Sequence[int], *, notify: bool = True) -> None:
        """Set selected indexes, enforcing the configured empty-selection policy."""
        normalized = normalize_multi_selector_indexes(
            len(self._labels), indexes, allow_empty=self._allow_empty
        )
        for index, variable in enumerate(self._variables):
            variable.set(index in normalized)
        if notify:
            self._notify()

    def _on_change(self) -> None:
        """Keep non-empty selectors valid after a checkbutton toggle."""
        if not self._allow_empty and not self.selected_indexes() and self._variables:
            self._variables[0].set(True)
        self._notify()

    def _notify(self) -> None:
        """Notify the owner after a user-visible selection change."""
        if self._command is not None:
            self._command()


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


class AdvancedViewControls(ttk.LabelFrame):
    """Responsive container for indivisible label/control groups."""

    def __init__(self, parent: ttk.Frame) -> None:
        super().__init__(parent, text="View controls", padding=(8, 4))
        self._groups: list[tuple[ttk.Frame, int]] = []
        self._layout_after_id: str | None = None
        self.bind("<Configure>", self._schedule_layout)
        self.bind("<Destroy>", self._cancel_layout)

    def add_group(self, *, requested_width: int = 0) -> ttk.Frame:
        """Create and register one indivisible control group."""
        group = ttk.Frame(self)
        self._groups.append((group, requested_width))
        self._schedule_layout()
        return group

    def _schedule_layout(self, _event: object | None = None) -> None:
        """Debounce layout work until Tk has updated the container width."""
        if self._layout_after_id is not None:
            self.after_cancel(self._layout_after_id)
        self._layout_after_id = self.after_idle(self._layout_groups)

    def _cancel_layout(self, _event: object | None = None) -> None:
        """Cancel pending callbacks before Tk destroys the widget."""
        if self._layout_after_id is not None:
            try:
                self.after_cancel(self._layout_after_id)
            except tk.TclError:
                pass
            self._layout_after_id = None

    def _layout_groups(self) -> None:
        """Place groups in rows according to their requested widths."""
        self._layout_after_id = None
        widths = [requested or group.winfo_reqwidth() for group, requested in self._groups]
        rows = layout_view_control_groups(max(self.winfo_width() - 16, 1), widths)
        for group, _requested in self._groups:
            group.grid_forget()
        for row_index, row in enumerate(rows):
            for column_index, group_index in enumerate(row):
                group = self._groups[group_index][0]
                group.grid(row=row_index, column=column_index, padx=(0, 12), pady=2, sticky=tk.W)


def make_view_controls(parent: ttk.Frame) -> AdvancedViewControls:
    """Create a responsive, consistently labelled Advanced control container."""
    controls = AdvancedViewControls(parent)
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
