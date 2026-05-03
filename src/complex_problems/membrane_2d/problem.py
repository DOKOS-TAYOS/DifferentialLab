"""Plugin entrypoint for the 2D membrane problem."""

from __future__ import annotations

from typing import TYPE_CHECKING

from complex_problems.base import ProblemDescriptor

if TYPE_CHECKING:
    from tkinter import Tk, Toplevel


class Membrane2DProblem:
    """Complex problem plugin for a 2D membrane of coupled oscillators."""

    descriptor = ProblemDescriptor(
        id="membrane_2d",
        name="2D Nonlinear Membrane",
        description=(
            "Rectangular membrane lattice with optional nonlinear corrections, "
            "energy diagnostics, spectral views, and animations."
        ),
    )

    def open_dialog(self, parent: "Tk | Toplevel") -> None:
        """Open the membrane configuration dialog."""
        from complex_problems.membrane_2d.ui import Membrane2DDialog

        Membrane2DDialog(parent)


PROBLEM = Membrane2DProblem()
