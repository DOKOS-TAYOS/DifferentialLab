"""Regression tests for vector animation ticks and MP4 export labels."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation_module  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from config import get_env_from_schema as configured_env  # noqa: E402
from plotting import (
    create_image_animation_plot,
    create_line_animation_plot,
    create_surface_animation_plot,
    create_vector_animation_plot,
    export_animated_figure_to_mp4,
    export_animation_to_mp4,
)


def _animation_data(n_components: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, 4)
    y = np.zeros((2 * n_components, len(x)))
    y[::2] = np.arange(n_components)[:, np.newaxis]
    return x, y


def test_vector_animation_thins_component_ticks_and_preserves_custom_labels() -> None:
    x, y = _animation_data(32)
    labels = [f"Mode {index}" for index in range(1, 33)]

    figure = create_vector_animation_plot(
        x,
        y,
        order=2,
        vector_components=32,
        component_labels=labels,
    )
    try:
        axis = figure.axes[0]
        tick_positions = axis.get_xticks()
        tick_labels = [label.get_text() for label in axis.get_xticklabels()]

        assert len(tick_positions) == 10
        assert len(tick_labels) == 10
        assert tick_positions[0] == 0
        assert tick_positions[-1] == 31
        assert tick_labels[0] == "Mode 1"
        assert tick_labels[-1] == "Mode 32"
        assert len(np.unique(tick_positions)) == len(tick_positions)
    finally:
        plt.close(figure)


def test_vector_animation_shows_every_component_for_small_vectors() -> None:
    x, y = _animation_data(10)

    figure = create_vector_animation_plot(x, y, order=2, vector_components=10)
    try:
        axis = figure.axes[0]
        assert list(axis.get_xticks()) == list(range(10))
        assert len(axis.get_xticklabels()) == 10
    finally:
        plt.close(figure)


def test_mp4_export_uses_same_component_tick_policy_and_custom_labels(
    tmp_path: Path,
) -> None:
    x, y = _animation_data(32)
    labels = [f"Mode {index}" for index in range(1, 33)]
    captured: dict[str, list[str]] = {}

    class _FakeAnimation:
        def __init__(self, figure: object, *_args: object, **_kwargs: object) -> None:
            self.figure = figure

        def save(self, _path: str, **_kwargs: object) -> None:
            axis = self.figure.axes[0]  # type: ignore[attr-defined]
            captured["labels"] = [label.get_text() for label in axis.get_xticklabels()]

    with (
        patch.object(animation_module, "FuncAnimation", _FakeAnimation),
        patch.object(animation_module.writers, "is_available", return_value=True),
    ):
        output = export_animation_to_mp4(
            x,
            y,
            order=2,
            vector_components=32,
            filepath=tmp_path / "animation.mp4",
            component_labels=labels,
        )

    assert output == tmp_path / "animation.mp4"
    assert captured["labels"][0] == "Mode 1"
    assert captured["labels"][-1] == "Mode 32"
    assert len(captured["labels"]) == 10


def test_image_and_surface_animation_helpers_render_small_frame_histories() -> None:
    t = np.array([0.0, 0.5, 1.0])
    frames = np.arange(3 * 3 * 4, dtype=float).reshape(3, 3, 4)

    image_figure = create_image_animation_plot(
        t,
        frames,
        title="Synthetic spectrum",
        xlabel="kx",
        ylabel="ky",
    )
    surface_figure = create_surface_animation_plot(
        t,
        np.arange(4),
        np.arange(3),
        frames,
        title="Synthetic membrane",
    )
    try:
        image_figure._animation_update(2)  # type: ignore[attr-defined]
        surface_figure._animation_update(2)  # type: ignore[attr-defined]
        assert image_figure.axes[0].get_title().endswith("t=1)")
        assert surface_figure.axes[0].get_title().endswith("t=1)")
        assert surface_figure.axes[0].get_zlabel() == "u"
    finally:
        plt.close(image_figure)
        plt.close(surface_figure)


def test_line_animation_uses_stable_ranges_and_frame_metadata() -> None:
    coordinates = np.array([0.0, 1.0])
    x = np.array([0.0, 1.0, 2.0])
    frames = np.array([[1.0, -2.0, 0.0], [3.0, 4.0, -1.0]])
    figure = create_line_animation_plot(
        coordinates,
        x,
        frames,
        title="Profile",
        xlabel="x",
        ylabel="u",
        frame_label="y",
    )
    try:
        axis = figure.axes[0]
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), frames[0])
        assert figure._animation_n_points == 2  # type: ignore[attr-defined]
        assert figure._animation_frame_label == "y"  # type: ignore[attr-defined]
        np.testing.assert_array_equal(figure._animation_frame_coordinates, coordinates)  # type: ignore[attr-defined]
        assert axis.get_title().endswith("y=0)")
        assert axis.get_ylim() == (-2.0, 4.0)
        figure._animation_update(1)  # type: ignore[attr-defined]
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), frames[1])
        assert axis.get_title().endswith("y=1)")
    finally:
        plt.close(figure)


def test_line_animation_symmetric_range_is_non_degenerate() -> None:
    figure = create_line_animation_plot(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([[0.0, 2.0], [-3.0, 1.0]]),
        title="Profile",
        xlabel="x",
        ylabel="u",
        symmetric_y_range=True,
    )
    try:
        assert figure.axes[0].get_ylim() == (-3.0, 3.0)
    finally:
        plt.close(figure)


def test_line_animation_hides_title_on_initial_and_updated_frames() -> None:
    def get_plot_config(key: str) -> object:
        if key == "PLOT_SHOW_TITLE":
            return False
        return configured_env(key)

    with patch("plotting.plot_utils.get_env_from_schema", side_effect=get_plot_config):
        figure = create_line_animation_plot(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            title="Hidden profile",
            xlabel="x",
            ylabel="u",
        )
        try:
            axis = figure.axes[0]
            assert axis.get_title() == ""
            figure._animation_update(1)  # type: ignore[attr-defined]
            assert axis.get_title() == ""
        finally:
            plt.close(figure)


def test_line_animation_uses_configured_line_style() -> None:
    configured_style = {
        "PLOT_LINE_COLOR": "darkgreen",
        "PLOT_LINE_WIDTH": 2.75,
        "PLOT_LINE_STYLE": "--",
    }

    def get_plot_config(key: str) -> object:
        return configured_style.get(key, configured_env(key))

    with patch("plotting.plot_utils.get_env_from_schema", side_effect=get_plot_config):
        figure = create_line_animation_plot(
            np.array([0.0]),
            np.array([0.0, 1.0]),
            np.array([[1.0, 2.0]]),
            title="Styled profile",
            xlabel="x",
            ylabel="u",
        )
    try:
        line = figure.axes[0].lines[0]
        assert line.get_color() == "darkgreen"
        assert line.get_linewidth() == 2.75
        assert line.get_linestyle() == "--"
    finally:
        plt.close(figure)


def test_image_and_surface_animation_frame_labels_are_customizable() -> None:
    frames = np.arange(8, dtype=float).reshape(2, 2, 2)
    image_figure = create_image_animation_plot(
        np.array([0.0, 1.0]), frames, title="Image", xlabel="x", ylabel="y", frame_label="z"
    )
    surface_figure = create_surface_animation_plot(
        np.array([0.0, 1.0]), np.arange(2), np.arange(2), frames, title="Surface", frame_label="z"
    )
    try:
        assert image_figure.axes[0].get_title().endswith("z=0)")
        assert surface_figure.axes[0].get_title().endswith("z=0)")
        assert image_figure._animation_frame_label == "z"  # type: ignore[attr-defined]
        assert surface_figure._animation_frame_label == "z"  # type: ignore[attr-defined]
        image_figure._animation_update(1)  # type: ignore[attr-defined]
        surface_figure._animation_update(1)  # type: ignore[attr-defined]
        assert image_figure.axes[0].get_title().endswith("z=1)")
        assert surface_figure.axes[0].get_title().endswith("z=1)")
    finally:
        plt.close(image_figure)
        plt.close(surface_figure)


def test_image_animation_uses_actual_nonzero_symmetric_amplitude() -> None:
    figure = create_image_animation_plot(
        np.array([0.0, 1.0]),
        np.array([[-0.05, 0.0], [0.02, 0.04]], dtype=float)[np.newaxis, ...].repeat(2, axis=0),
        title="Low-amplitude field",
        xlabel="x",
        ylabel="y",
        symmetric_color_range=True,
    )
    try:
        assert figure.axes[0].images[0].get_clim() == (-0.05, 0.05)
    finally:
        plt.close(figure)


def test_surface_animation_uses_full_history_amplitude_before_downsampling() -> None:
    frames = np.zeros((2, 5, 5), dtype=float)
    frames[1, 1, 1] = 0.2

    figure = create_surface_animation_plot(
        np.array([0.0, 1.0]),
        np.arange(5),
        np.arange(5),
        frames,
        title="Low-amplitude membrane",
        max_render_resolution=2,
    )
    try:
        axis = figure.axes[0]
        assert axis.get_zlim() == (-0.2, 0.2)
        assert axis.collections[0].get_clim() == (-0.2, 0.2)
    finally:
        plt.close(figure)


def test_surface_animation_supports_positive_ranges_and_custom_labels() -> None:
    frames = np.zeros((2, 5, 5), dtype=float)
    frames[1, 1, 1] = 0.2

    figure = create_surface_animation_plot(
        np.array([0.0, 1.0]),
        np.arange(5),
        np.arange(5),
        frames,
        title="Density",
        max_render_resolution=2,
        xlabel="x",
        ylabel="y",
        zlabel="|ψ|²",
        colorbar_label="|ψ|²",
        symmetric_z_range=False,
    )
    try:
        axis = figure.axes[0]
        assert axis.get_zlim() == (0.0, 0.2)
        assert axis.collections[0].get_clim() == (0.0, 0.2)
        assert axis.get_xlabel() == "x"
        assert axis.get_ylabel() == "y"
        assert axis.get_zlabel() == "|ψ|²"
        assert figure.axes[1].get_ylabel() == "|ψ|²"
    finally:
        plt.close(figure)


def test_animation_helpers_use_valid_range_for_all_zero_history() -> None:
    t = np.array([0.0, 1.0])
    frames = np.zeros((2, 2, 3), dtype=float)

    image_figure = create_image_animation_plot(
        t,
        frames,
        title="Zero field",
        xlabel="x",
        ylabel="y",
        symmetric_color_range=True,
    )
    surface_figure = create_surface_animation_plot(
        t,
        np.arange(3),
        np.arange(2),
        frames,
        title="Zero membrane",
    )
    try:
        assert image_figure.axes[0].images[0].get_clim() == (-1.0, 1.0)
        assert surface_figure.axes[0].get_zlim() == (-1.0, 1.0)
    finally:
        plt.close(image_figure)
        plt.close(surface_figure)


def test_figure_animation_export_uses_attached_animation_payload(tmp_path: Path) -> None:
    figure = create_image_animation_plot(
        np.array([0.0, 1.0]),
        np.arange(8, dtype=float).reshape(2, 2, 2),
        title="Synthetic spectrum",
        xlabel="kx",
        ylabel="ky",
    )
    captured: dict[str, str] = {}

    class _FakeAnimation:
        def __init__(self, _figure: object, callback: object, **_kwargs: object) -> None:
            callback(1)  # type: ignore[operator]

        def save(self, _path: str, **_kwargs: object) -> None:
            captured["title"] = figure.axes[0].get_title()

    with (
        patch.object(animation_module, "FuncAnimation", _FakeAnimation),
        patch.object(animation_module.writers, "is_available", return_value=True),
    ):
        output = export_animated_figure_to_mp4(figure, tmp_path / "animated.mp4")

    assert output == tmp_path / "animated.mp4"
    assert captured["title"].endswith("t=1)")
