"""Focused tests for antenna radiation result-dialog plotting helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.projections.polar import PolarAxes

from complex_problems.antenna_radiation.result_dialog import (
    _create_3d_pattern_figure,
    _create_polar_cut_figure,
)


def test_create_polar_cut_figure_builds_polar_plot() -> None:
    figure = _create_polar_cut_figure(
        np.array([0.0, 45.0, 90.0]),
        np.array([1.0, 2.0, 1.5]),
        title="Synthetic theta cut",
    )

    try:
        assert len(figure.axes) == 1
        axes = figure.axes[0]
        assert isinstance(axes, PolarAxes)
        assert len(axes.lines) == 1
        assert axes.lines[0].get_ydata().tolist() == [1.0, 2.0, 1.5]
        assert axes.get_title() == "Synthetic theta cut"
    finally:
        plt.close(figure)


def test_create_3d_pattern_figure_builds_surface_plot() -> None:
    figure = _create_3d_pattern_figure(
        np.array([0.0, 45.0, 90.0]),
        np.array([0.0, 90.0, 180.0, 270.0]),
        np.array(
            [
                [-10.0, -8.0, -10.0, -8.0],
                [-3.0, -1.0, -3.0, -1.0],
                [0.0, -2.0, 0.0, -2.0],
            ]
        ),
    )

    try:
        assert len(figure.axes) == 1
        axes = figure.axes[0]
        assert axes.name == "3d"
        assert len(axes.collections) >= 1
    finally:
        plt.close(figure)
