"""Regression coverage for animated 2D aerodynamics result views."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch

import complex_problems.aerodynamics_2d.result_dialog as result_dialog


def _make_result() -> SimpleNamespace:
    t = np.array([0.0, 0.5, 1.0])
    x = np.linspace(-2.0, 2.0, 6)
    y = np.linspace(-1.0, 1.0, 4)
    xx, yy = np.meshgrid(x, y)
    obstacle_mask = (xx * xx + yy * yy) < 0.2
    u = np.stack([np.full((4, 6), 1.0 + i) for i in range(3)])
    v = np.stack([np.full((4, 6), 0.1 * i) for i in range(3)])
    speed = np.sqrt(u * u + v * v)
    vorticity = np.stack([np.full((4, 6), -0.2 + i) for i in range(3)])
    pressure = np.stack([np.full((4, 6), 0.1 * i) for i in range(3)])
    return SimpleNamespace(
        x=x,
        y=y,
        t=t,
        u=u,
        v=v,
        speed=speed,
        vorticity=vorticity,
        pressure=pressure,
        obstacle_mask=obstacle_mask,
        drag_coeff=np.zeros(3),
        lift_coeff=np.zeros(3),
        magnitudes={
            "reynolds": 1.0,
            "mean_cd_tail": 0.0,
            "rms_cl": 0.0,
            "max_divergence_l2": 0.0,
        },
    )


def test_field_map_uses_physical_coordinates_and_all_speed_frames() -> None:
    result = _make_result()
    payload = result_dialog._create_field_payload(result, "speed")
    figure = result_dialog._create_animation_figure(payload)
    try:
        axis = figure.axes[0]
        assert payload.frames is result.speed
        assert payload.t is result.t
        assert figure._animation_n_points == len(result.t)  # type: ignore[attr-defined]
        assert axis.images[0].get_extent() == [
            result.x[0],
            result.x[-1],
            result.y[0],
            result.y[-1],
        ]
        assert axis.images[0].get_clim()[0] >= 0.0
        assert axis.images[0].get_clim()[1] == np.max(result.speed)
        figure._animation_update(2)  # type: ignore[attr-defined]
        assert axis.get_title().endswith("t=1)")
        np.testing.assert_array_equal(axis.images[0].get_array(), result.speed[2])
    finally:
        plt.close(figure)


def test_streamlines_update_without_mutating_result_and_keep_global_normalization() -> None:
    result = _make_result()
    original_u = result.u.copy()
    original_v = result.v.copy()
    payload = result_dialog._create_streamline_payload(result)
    figure = result_dialog._create_animation_figure(payload)
    try:
        axis = figure.axes[0]
        assert payload.u is result.u
        assert payload.v is result.v
        assert payload.speed is result.speed
        assert any(
            collection.get_clim() == (0.0, np.max(result.speed))
            for collection in axis.collections
            if hasattr(collection, "get_clim")
        )
        figure._animation_update(1)  # type: ignore[attr-defined]
        figure._animation_update(2)  # type: ignore[attr-defined]
        assert axis.get_title().endswith("t=1)")
        assert len(axis.collections) >= 2  # obstacle contour plus current streamlines
        np.testing.assert_array_equal(result.u, original_u)
        np.testing.assert_array_equal(result.v, original_v)
    finally:
        plt.close(figure)


def test_streamlines_remove_previous_arrow_patches_on_every_update() -> None:
    result = _make_result()
    payload = result_dialog._create_streamline_payload(result)
    figure = result_dialog._create_animation_figure(payload)
    try:
        axis = figure.axes[0]

        def streamline_arrows() -> tuple[FancyArrowPatch, ...]:
            return tuple(patch for patch in axis.patches if isinstance(patch, FancyArrowPatch))

        previous_arrows = streamline_arrows()
        assert previous_arrows
        initial_arrow_count = len(previous_arrows)

        figure._animation_update(1)  # type: ignore[attr-defined]
        replacement_arrows = streamline_arrows()
        assert replacement_arrows
        assert all(arrow not in axis.patches for arrow in previous_arrows)
        assert len(replacement_arrows) != 0

        update_sequence = [2, 0, 2, 1, 0]
        expected_counts = [len(replacement_arrows)]
        for index in update_sequence:
            previous_arrows = replacement_arrows
            figure._animation_update(index)  # type: ignore[attr-defined]
            replacement_arrows = streamline_arrows()
            assert all(arrow not in axis.patches for arrow in previous_arrows)
            expected_counts.append(len(replacement_arrows))

        for _ in range(20):
            for expected_count, index in zip(expected_counts, [1, *update_sequence], strict=True):
                previous_arrows = replacement_arrows
                figure._animation_update(index)  # type: ignore[attr-defined]
                replacement_arrows = streamline_arrows()
                assert all(arrow not in axis.patches for arrow in previous_arrows)
                assert len(replacement_arrows) == expected_count

        assert len(replacement_arrows) == initial_arrow_count
    finally:
        plt.close(figure)


def test_centerline_animation_updates_two_labeled_lines_with_stable_limits() -> None:
    result = _make_result()
    payload = result_dialog._create_centerline_payload(result)
    figure = result_dialog._create_animation_figure(payload)
    try:
        axis = figure.axes[0]
        assert len(axis.lines) == 2
        assert [line.get_label() for line in axis.lines] == ["u centerline", "ω centerline"]
        initial_limits = axis.get_ylim()
        figure._animation_update(2)  # type: ignore[attr-defined]
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), result.u[2, len(result.y) // 2])
        np.testing.assert_array_equal(
            axis.lines[1].get_ydata(), result.vorticity[2, len(result.y) // 2]
        )
        assert axis.get_ylim() == initial_limits
        assert axis.get_title().endswith("t=1)")
    finally:
        plt.close(figure)


def test_animation_wiring_uses_shared_embedding_and_export_callbacks() -> None:
    result = _make_result()
    dialog = object.__new__(result_dialog.Aerodynamics2DResultDialog)
    dialog._result = result
    dialog.win = object()
    dialog._map_canvas = None
    dialog._stream_canvas = None
    dialog._profile_canvas = None
    embedded: list[dict[str, object]] = []

    def embed(_figure: object, _parent: object, **kwargs: object) -> object:
        embedded.append(kwargs)
        return object()

    with patch.object(result_dialog, "embed_animation_plot_in_tk", side_effect=embed):
        dialog._build_map_tab(object())
        dialog._build_stream_tab(object())
        dialog._build_profile_tab(object())

    assert len(embedded) == 3
    assert all(callable(item["on_export_mp4"]) for item in embedded)


def test_mp4_export_cancel_success_and_ffmpeg_error_are_user_facing(tmp_path: Path) -> None:
    result = _make_result()
    payload = result_dialog._create_field_payload(result, "speed")
    dialog = object.__new__(result_dialog.Aerodynamics2DResultDialog)
    dialog.win = object()
    export = MagicMock()

    with (
        patch.object(result_dialog.filedialog, "asksaveasfilename", return_value=""),
        patch.object(result_dialog, "export_animated_figure_to_mp4", export),
    ):
        dialog._on_export_animation_mp4(payload, 2.0)
    export.assert_not_called()

    output = tmp_path / "aerodynamics.mp4"
    with (
        patch.object(result_dialog.filedialog, "asksaveasfilename", return_value=str(output)),
        patch.object(result_dialog, "export_animated_figure_to_mp4", export),
        patch.object(result_dialog.messagebox, "showinfo") as show_info,
    ):
        dialog._on_export_animation_mp4(payload, 2.0)
    assert export.call_args.kwargs["duration_seconds"] == 2.0
    show_info.assert_called_once()

    export.side_effect = RuntimeError("FFMpeg is not available")
    with (
        patch.object(result_dialog.filedialog, "asksaveasfilename", return_value=str(output)),
        patch.object(result_dialog, "export_animated_figure_to_mp4", export),
        patch.object(result_dialog.messagebox, "showerror") as show_error,
    ):
        dialog._on_export_animation_mp4(payload, 2.0)
    assert "ffmpeg" in show_error.call_args.args[1].lower()
    plt.close("all")
