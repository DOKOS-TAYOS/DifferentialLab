"""Plugin entrypoint for antenna radiation simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from complex_problems.base import ProblemDescriptor

if TYPE_CHECKING:
    from tkinter import Tk, Toplevel


class AntennaRadiationProblem:
    """Complex problem plugin for far-field antenna radiation patterns."""

    descriptor = ProblemDescriptor(
        id="antenna_radiation",
        name="Antenna Radiation",
        description=(
            "Far-field radiation patterns and field magnitudes for dipoles, "
            "loops, patch-like apertures, and uniform linear arrays."
        ),
    )

    def open_dialog(self, parent: "Tk | Toplevel") -> None:
        """Open the antenna configuration dialog."""
        from complex_problems.antenna_radiation.ui import AntennaRadiationDialog

        AntennaRadiationDialog(parent)


PROBLEM = AntennaRadiationProblem()
