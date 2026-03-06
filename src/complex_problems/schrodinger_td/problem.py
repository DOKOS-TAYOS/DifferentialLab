"""Plugin entrypoint for time-dependent Schrodinger simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from complex_problems.base import ProblemDescriptor

if TYPE_CHECKING:
    from tkinter import Tk, Toplevel


class SchrodingerTDProblem:
    """Complex problem plugin for TDSE in 1D/2D."""

    descriptor = ProblemDescriptor(
        id="schrodinger_td",
        name="Schrodinger TD (1D/2D)",
        description=(
            "Time-dependent Schrodinger equation with split-operator spectral "
            "solver, configurable potentials, and packet initial states."
        ),
    )

    def open_dialog(self, parent: "Tk | Toplevel") -> None:
        """Open the Schrodinger TD configuration dialog."""
        from complex_problems.schrodinger_td.ui import SchrodingerTDDialog

        SchrodingerTDDialog(parent)


PROBLEM = SchrodingerTDProblem()

