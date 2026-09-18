"""Regression tests for vector animation ticks and MP4 export labels."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation_module  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from plotting import create_vector_animation_plot, export_animation_to_mp4


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
