"""Plugin entrypoint for pipe flow simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from complex_problems.base import ProblemDescriptor

if TYPE_CHECKING:
    from tkinter import Tk, Toplevel


class PipeFlowProblem:
    """Complex problem plugin for pipe flow approximations."""

    descriptor = ProblemDescriptor(
        id="pipe_flow",
        name="Pipe Flow",
        description=(
            "Steady and transient 1D pipe-flow models with configurable geometry, "
            "fluid properties, friction correlations, and pressure boundary conditions."
        ),
    )

    def open_dialog(self, parent: "Tk | Toplevel") -> None:
        """Open the pipe-flow configuration dialog."""
        from complex_problems.pipe_flow.ui import PipeFlowDialog

        PipeFlowDialog(parent)


PROBLEM = PipeFlowProblem()
