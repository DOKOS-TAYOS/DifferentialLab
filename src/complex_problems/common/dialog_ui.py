"""Shared UI helpers for complex problem configuration dialogs."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Protocol, TypeVar

from complex_problems.common.background import run_solver_with_loading
from config import get_env_from_schema
from frontend.theme import get_font
from frontend.ui_dialogs.keyboard_nav import setup_arrow_enter_navigation
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame
from frontend.window_utils import (
    bind_wraplength,
    calculate_screen_aware_minsize,
    center_window,
)

_TResult_contra = TypeVar("_TResult_contra", contravariant=True)


class ResultDialogFactory(Protocol[_TResult_contra]):
    """Callable that opens a result dialog for a solved problem."""

    def __call__(
        self,
        parent: tk.Tk | tk.Toplevel,
        *,
        result: _TResult_contra,
    ) -> object: ...


@dataclass(frozen=True)
class AdvancedDialogSize:
    """Preferred and minimum dimensions for an Advanced setup dialog."""

    preferred_width: int
    preferred_height: int
    minimum_width: int = 680
    minimum_height: int = 520


ADVANCED_ACTION_STYLES = {
    "primary": "Primary.TButton",
    "secondary": "Secondary.TButton",
}


class AdvancedDialogShell:
    """Small presentation shell shared by Advanced configuration dialogs.

    Problem modules continue to own their controls and scientific behavior. The
    shell only provides the standard header, scrollable body, and fixed footer.
    """

    def __init__(
        self,
        window: tk.Toplevel,
        *,
        title: str,
        description: str,
        pad: int | None = None,
    ) -> None:
        self.window = window
        self.pad = pad if pad is not None else int(get_env_from_schema("UI_PADDING"))
        background = get_env_from_schema("UI_BACKGROUND")
        window.configure(bg=background)

        self.root = ttk.Frame(window, padding=self.pad * 2)
        self.root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(self.root)
        header.pack(fill=tk.X, pady=(0, self.pad))
        ttk.Label(header, text=title, style="Title.TLabel").pack(anchor=tk.W)
        description_label = ttk.Label(
            header,
            text=description,
            style="Small.TLabel",
            justify=tk.LEFT,
            wraplength=760,
        )
        description_label.pack(fill=tk.X, anchor=tk.W, pady=(4, 0))
        bind_wraplength(header, description_label, pad=self.pad, min_wrap=240)

        self.scroll = ScrollableFrame(self.root)
        self.scroll.apply_bg(background)
        self.scroll.pack(fill=tk.BOTH, expand=True)
        self.body = self.scroll.inner
        self.body.configure(padding=(0, self.pad, self.pad, self.pad))

        ttk.Separator(self.root).pack(fill=tk.X, pady=(self.pad, self.pad))
        self.footer = ttk.Frame(self.root)
        self.footer.pack(fill=tk.X)
        self.footer_left = ttk.Frame(self.footer)
        self.footer_left.pack(side=tk.LEFT)
        self.footer_right = ttk.Frame(self.footer)
        self.footer_right.pack(side=tk.RIGHT)
        self._footer_buttons: list[ttk.Button] = []

    def add_footer_button(
        self,
        text: str,
        command: Callable[[], object],
        *,
        primary: bool = False,
        contextual: bool = False,
    ) -> ttk.Button:
        """Add a consistently styled footer action and return the button."""
        button = ttk.Button(
            self.footer_left if contextual else self.footer_right,
            text=text,
            style=ADVANCED_ACTION_STYLES["primary" if primary else "secondary"],
            command=command,
        )
        if contextual:
            button.pack(side=tk.LEFT, padx=(0, self.pad))
        else:
            button.pack(side=tk.LEFT, padx=(self.pad, 0))
        self._footer_buttons.append(button)
        return button

    def refresh(self) -> None:
        """Refresh scrolling and wheel bindings after conditional layout changes."""
        self.scroll.bind_new_children()
        self.scroll.refresh_scroll_region()

    def finish(self, size: AdvancedDialogSize) -> None:
        """Apply screen-aware sizing and finish keyboard/footer setup."""
        self.refresh()
        min_width, min_height = calculate_screen_aware_minsize(
            self.window.winfo_screenwidth(),
            self.window.winfo_screenheight(),
            size.minimum_width,
            size.minimum_height,
        )
        self.window.minsize(min_width, min_height)
        center_window(
            self.window,
            size.preferred_width,
            size.preferred_height,
            max_width_ratio=0.9,
            max_height_ratio=0.9,
            resizable=True,
        )
        if self._footer_buttons:
            setup_arrow_enter_navigation([self._footer_buttons])


def make_labeled_entry(
    parent: ttk.Frame,
    label: str,
    variable: tk.StringVar,
    *,
    width: int = 10,
) -> ttk.Entry:
    """Create a label and entry pair packed left-to-right."""
    ttk.Label(parent, text=f"{label}:").pack(side=tk.LEFT, padx=(0, 4))
    entry = ttk.Entry(parent, textvariable=variable, width=width, font=get_font())
    entry.pack(side=tk.LEFT, padx=(0, 12))
    return entry


def make_labeled_spinbox(
    parent: ttk.Frame,
    label: str,
    variable: tk.StringVar,
    *,
    from_: int,
    to: int,
    width: int = 8,
) -> ttk.Spinbox:
    """Create a label and spinbox pair packed left-to-right."""
    ttk.Label(parent, text=f"{label}:").pack(side=tk.LEFT, padx=(0, 4))
    spinbox = ttk.Spinbox(
        parent,
        textvariable=variable,
        from_=from_,
        to=to,
        width=width,
        font=get_font(),
    )
    spinbox.pack(side=tk.LEFT, padx=(0, 12))
    return spinbox


def make_labeled_combo(
    parent: ttk.Frame,
    label: str,
    variable: tk.StringVar,
    values: Sequence[str],
    *,
    width: int = 12,
) -> ttk.Combobox:
    """Create a label and readonly combobox pair packed left-to-right."""
    ttk.Label(parent, text=f"{label}:").pack(side=tk.LEFT, padx=(0, 4))
    combo = ttk.Combobox(
        parent,
        textvariable=variable,
        values=list(values),
        state="readonly",
        width=width,
        font=get_font(),
    )
    combo.pack(side=tk.LEFT, padx=(0, 12))
    return combo


def run_solver_dialog(
    *,
    parent: tk.Tk | tk.Toplevel,
    window: tk.Toplevel,
    collect_inputs: Callable[[], dict[str, object]],
    solver: Callable[..., _TResult_contra],
    message: str,
    result_parent: tk.Tk | tk.Toplevel,
    result_dialog_factory: ResultDialogFactory[_TResult_contra],
    invalid_input_title: str = "Check the input values",
    error_title: str = "Solver error",
    poll_ms: int = 100,
    confirm_run: Callable[[dict[str, object], tk.Toplevel], bool] | None = None,
) -> None:
    """Validate inputs, start a solver in background, and open its result dialog."""
    try:
        params = collect_inputs()
    except ValueError as exc:
        messagebox.showerror(invalid_input_title, str(exc), parent=window)
        return

    if confirm_run is not None and not confirm_run(params, window):
        return

    window.destroy()

    def _task() -> _TResult_contra:
        return solver(**params)

    def _on_success(result: _TResult_contra) -> None:
        result_dialog_factory(result_parent, result=result)

    run_solver_with_loading(
        parent=parent,
        message=message,
        task=_task,
        on_success=_on_success,
        error_title=error_title,
        poll_ms=poll_ms,
    )
