"""Lifecycle regressions for the initial ResultDialog plot render."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import frontend.ui_dialogs.result_dialog as result_dialog_ui


def test_result_dialog_sets_geometry_and_materializes_layout_before_first_plot() -> None:
    """The first canvas is created only after its parent has an allocated size."""
    events: list[str] = []
    window = MagicMock()
    window.winfo_screenwidth.return_value = 1920
    window.winfo_screenheight.return_value = 1080
    window.minsize.side_effect = lambda *_: events.append("minsize")
    window.update_idletasks.side_effect = lambda: events.append("layout")
    result = SimpleNamespace(
        metadata={"equation_name": "Simple Harmonic Oscillator"},
        notation=None,
        vector_order=1,
    )

    with (
        patch.object(result_dialog_ui.tk, "Toplevel", return_value=window),
        patch.object(
            result_dialog_ui,
            "center_window",
            side_effect=lambda *_args, **_kwargs: events.append("geometry"),
        ),
        patch.object(
            result_dialog_ui.ResultDialog,
            "_build_ui",
            side_effect=lambda: events.append("controls"),
        ),
        patch.object(
            result_dialog_ui.ResultDialog,
            "_build_plot_tabs",
            side_effect=lambda: events.append("plots"),
        ),
        patch.object(result_dialog_ui, "make_modal", side_effect=lambda *_: events.append("modal")),
    ):
        result_dialog_ui.ResultDialog(MagicMock(), result=result)

    assert events == ["geometry", "minsize", "controls", "layout", "plots", "modal"]
