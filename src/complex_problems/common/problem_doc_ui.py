"""UI helpers for rendering shared complex-problem documentation blocks."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from complex_problems.problem_docs import get_problem_doc
from frontend.ui_dialogs.collapsible_section import CollapsibleSection
from frontend.ui_dialogs.scrollable_frame import ScrollableFrame


def _build_doc_text(problem_id: str) -> str:
    doc = get_problem_doc(problem_id)
    lines: list[str] = []
    if doc.equation_summary:
        lines.append(f"Equation: {doc.equation_summary}")
        lines.append("")

    lines.append("Physical description:")
    lines.extend(f"• {line}" for line in _split_sentences(doc.extended_description))
    lines.append("")

    lines.append("What each option controls:")
    lines.extend(f"• {line}" for line in doc.config_options_summary)
    lines.append("")

    lines.append("Main visualizations:")
    lines.extend(f"• {line}" for line in doc.visualizations_summary)
    return "\n".join(lines)


def _split_sentences(text: str) -> tuple[str, ...]:
    pieces = [piece.strip() for piece in text.split(".") if piece.strip()]
    return tuple(f"{piece}." for piece in pieces) if pieces else (text,)


def add_how_to_config_section(
    parent: ttk.Frame,
    scroll: ScrollableFrame,
    *,
    problem_id: str,
    pad: int,
    wraplength: int = 760,
) -> None:
    """Add a standard collapsed 'How to configure' block for a problem dialog."""
    section = CollapsibleSection(parent, scroll, "How to configure", expanded=False, pad=pad)
    lbl = ttk.Label(
        section.content,
        text=_build_doc_text(problem_id),
        style="Small.TLabel",
        justify=tk.LEFT,
        wraplength=wraplength,
    )
    lbl.pack(anchor=tk.W)
    scroll.bind_new_children()
