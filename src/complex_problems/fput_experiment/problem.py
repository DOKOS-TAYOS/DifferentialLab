"""Plugin entrypoint for the Fermi--Pasta--Ulam--Tsingou experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from complex_problems.base import ProblemDescriptor

if TYPE_CHECKING:
    from tkinter import Tk, Toplevel


class FPUTExperimentProblem:
    """Dedicated advanced-problem workflow for fixed-end FPUT chains."""

    descriptor = ProblemDescriptor(
        id="fput_experiment",
        name="Fermi-Pasta-Ulam-Tsingou Experiment",
        description=(
            "Fixed-end alpha/beta FPUT chain with recurrence, modal-energy, strain, "
            "and scaling studies."
        ),
    )

    def open_dialog(self, parent: "Tk | Toplevel") -> None:
        """Open the FPUT configuration dialog lazily."""
        from complex_problems.fput_experiment.ui import FPUTExperimentDialog

        FPUTExperimentDialog(parent)


PROBLEM = FPUTExperimentProblem()
