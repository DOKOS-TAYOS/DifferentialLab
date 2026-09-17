"""Lifecycle regressions for the initial ResultDialog plot render."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import frontend.ui_dialogs.result_dialog as result_dialog_ui


def test_result_dialog_sets_geometry_and_materializes_layout_before_first_plot() -> None:
    """Initial plot callbacks run only after the constructed frames are laid out."""
    events: list[str] = []
    state: dict[str, object] = {}
    window = MagicMock()
    window.winfo_screenwidth.return_value = 1920
    window.winfo_screenheight.return_value = 1080
    window.minsize.side_effect = lambda *_: events.append("minsize")

    def build_ui(dialog: result_dialog_ui.ResultDialog) -> None:
        events.append("controls")

    def build_plot_tabs(dialog: result_dialog_ui.ResultDialog) -> None:
        events.append("tabs")
        dialog.plot_frame = object()
        state["dialog"] = dialog

        def render_initial_plot() -> None:
            assert events[-1] == "layout"
            assert dialog.plot_frame is not None
            events.append("render")

        dialog._queue_initial_plot(render_initial_plot)

    def materialize_layout() -> None:
        dialog = state["dialog"]
        assert isinstance(dialog, result_dialog_ui.ResultDialog)
        assert hasattr(dialog, "plot_frame")
        assert events == ["geometry", "minsize", "controls", "tabs"]
        events.append("layout")

    window.update_idletasks.side_effect = materialize_layout
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
        patch.object(result_dialog_ui.ResultDialog, "_build_ui", build_ui),
        patch.object(result_dialog_ui.ResultDialog, "_build_plot_tabs", build_plot_tabs),
        patch.object(result_dialog_ui, "make_modal", side_effect=lambda *_: events.append("modal")),
    ):
        dialog = result_dialog_ui.ResultDialog(MagicMock(), result=result)

    assert events == ["geometry", "minsize", "controls", "tabs", "layout", "render", "modal"]
    assert dialog._initial_plot_callbacks == []
