"""Plugin entrypoint for 3D aerodynamics simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from complex_problems.base import ProblemDescriptor

if TYPE_CHECKING:
    from tkinter import Tk, Toplevel


class Aerodynamics3DProblem:
    """Complex problem plugin for lightweight 3D incompressible flow."""

    descriptor = ProblemDescriptor(
        id="aerodynamics_3d",
        name="Aerodynamics 3D",
        description=(
            "Incompressible 3D structured-grid flow in a periodic Cartesian domain "
            "with immersed/penalized obstacles; a lightweight educational/scientific "
            "solver, not an industrial CFD replacement."
        ),
    )

    def open_dialog(self, parent: "Tk | Toplevel") -> None:
        """Open the aerodynamics configuration dialog."""
        from complex_problems.aerodynamics_3d.ui import Aerodynamics3DDialog

        Aerodynamics3DDialog(parent)


PROBLEM = Aerodynamics3DProblem()
