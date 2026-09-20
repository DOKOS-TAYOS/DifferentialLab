"""Plugin entrypoint for Gravitational N-Body Dynamics."""
# ruff: noqa: E501

from __future__ import annotations

from typing import TYPE_CHECKING

from complex_problems.base import ProblemDescriptor

if TYPE_CHECKING:
    from tkinter import Tk, Toplevel


class GravitationalNBodyProblem:
    """Advanced Problem plugin for classical Newtonian point-mass gravity."""

    descriptor = ProblemDescriptor(
        id="gravitational_n_body",
        name="Gravitational N-Body Dynamics",
        description="Interactive 2D/3D Newtonian point-mass gravity for curated three-body and general N-body systems.",
    )

    def open_dialog(self, parent: "Tk | Toplevel") -> None:
        """Open the gravitational N-body configuration dialog."""
        from complex_problems.gravitational_n_body.ui import GravitationalNBodyDialog

        GravitationalNBodyDialog(parent)


PROBLEM = GravitationalNBodyProblem()
