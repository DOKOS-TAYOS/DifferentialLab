"""Shared types for complex problem plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from tkinter import Tk, Toplevel


@dataclass(frozen=True, slots=True)
class ProblemDescriptor:
    """Display metadata for a complex problem."""

    id: str
    name: str
    description: str


class ComplexProblem(Protocol):
    """Protocol that each complex problem plugin must satisfy."""

    descriptor: ProblemDescriptor

    def open_dialog(self, parent: "Tk | Toplevel") -> None:
        """Open the problem-specific configuration dialog."""

