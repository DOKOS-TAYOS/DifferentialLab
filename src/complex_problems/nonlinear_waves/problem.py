"""Plugin entrypoint for nonlinear wave propagation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from complex_problems.base import ProblemDescriptor

if TYPE_CHECKING:
    from tkinter import Tk, Toplevel


class NonlinearWavesProblem:
    """Complex problem plugin for NLSE and KdV simulations."""

    descriptor = ProblemDescriptor(
        id="nonlinear_waves",
        name="Nonlinear Waves (NLSE + KdV)",
        description=(
            "Propagation in nonlinear media with pseudo-spectral solvers for "
            "NLSE (complex envelope) and KdV (real nonlinear dispersive waves)."
        ),
    )

    def open_dialog(self, parent: "Tk | Toplevel") -> None:
        """Open the nonlinear waves configuration dialog."""
        from complex_problems.nonlinear_waves.ui import NonlinearWavesDialog

        NonlinearWavesDialog(parent)


PROBLEM = NonlinearWavesProblem()

