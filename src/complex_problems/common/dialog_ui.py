"""Shared UI helpers for complex problem configuration dialogs."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Sequence
from tkinter import messagebox, ttk
from typing import Protocol, TypeVar

from complex_problems.common.background import run_solver_with_loading
from frontend.theme import get_font

_TResult_contra = TypeVar("_TResult_contra", contravariant=True)


class ResultDialogFactory(Protocol[_TResult_contra]):
    """Callable that opens a result dialog for a solved problem."""

    def __call__(
        self,
        parent: tk.Tk | tk.Toplevel,
        *,
        result: _TResult_contra,
    ) -> object: ...


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
    invalid_input_title: str = "Invalid input",
    error_title: str = "Solver Error",
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
