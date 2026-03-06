"""Plugin entrypoint for 2D aerodynamics simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from complex_problems.base import ProblemDescriptor

if TYPE_CHECKING:
    from tkinter import Tk, Toplevel


class Aerodynamics2DProblem:
    """Complex problem plugin for 2D incompressible flow around an obstacle."""

    descriptor = ProblemDescriptor(
        id="aerodynamics_2d",
        name="Aerodynamics 2D",
        description=(
            "2D incompressible flow around configurable obstacles with "
            "Navier-Stokes projection and Stokes-limit approximations."
        ),
    )

    def open_dialog(self, parent: "Tk | Toplevel") -> None:
        """Open the aerodynamics configuration dialog."""
        from complex_problems.aerodynamics_2d.ui import Aerodynamics2DDialog

        Aerodynamics2DDialog(parent)


PROBLEM = Aerodynamics2DProblem()
