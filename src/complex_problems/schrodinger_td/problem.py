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
        name="Schrodinger Time Evolution (1D/2D)",
        description=(
            "Split-operator TDSE solver with configurable potentials, wave packets, "
            "and 1D or 2D domains."
        ),
    )

    def open_dialog(self, parent: "Tk | Toplevel") -> None:
        """Open the Schrodinger TD configuration dialog."""
        from complex_problems.schrodinger_td.ui import SchrodingerTDDialog

        SchrodingerTDDialog(parent)


PROBLEM = SchrodingerTDProblem()
