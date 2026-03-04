"""Registry of available complex problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from tkinter import Tk, Toplevel


@dataclass
class ProblemDescriptor:
    """Descriptor for a complex problem.

    Attributes:
        id: Unique identifier for the problem.
        name: Display name for the UI.
        description: Short description of the problem.
        open_dialog: Callable(parent) -> None that opens the problem-specific dialog.
    """

    id: str
    name: str
    description: str
    open_dialog: Callable[["Tk | Toplevel"], None]


def _open_coupled_oscillators(parent: "Tk | Toplevel") -> None:
    """Open the coupled oscillators dialog."""
    from complex_problems.coupled_oscillators.ui import CoupledOscillatorsDialog

    CoupledOscillatorsDialog(parent)


PROBLEM_REGISTRY: dict[str, ProblemDescriptor] = {
    "coupled_oscillators": ProblemDescriptor(
        id="coupled_oscillators",
        name="Coupled Harmonic Oscillators",
        description=(
            "One-dimensional chain of N oscillators with configurable "
            "masses, coupling constants, and coupling types."
        ),
        open_dialog=_open_coupled_oscillators,
    ),
}
