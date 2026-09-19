"""Plugin entrypoint for the coupled oscillators complex problem."""

from __future__ import annotations

from typing import TYPE_CHECKING

from complex_problems.base import ProblemDescriptor

if TYPE_CHECKING:
    from tkinter import Tk, Toplevel


class CoupledOscillatorsProblem:
    """Complex problem plugin for coupled harmonic oscillators."""

    descriptor = ProblemDescriptor(
        id="coupled_oscillators",
        name="Coupled Harmonic Oscillators",
        description=(
            "Mass-spring chain with configurable masses, boundary conditions, "
            "long-range coupling, nonlinear terms, and forcing."
        ),
    )

    def open_dialog(self, parent: "Tk | Toplevel") -> None:
        """Open the coupled oscillators configuration dialog."""
        from complex_problems.coupled_oscillators.ui import CoupledOscillatorsDialog

        CoupledOscillatorsDialog(parent)


PROBLEM = CoupledOscillatorsProblem()
