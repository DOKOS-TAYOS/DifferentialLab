"""Helpers for attaching animation metadata to matplotlib figures."""

from __future__ import annotations

from typing import Any, Callable, TypeAlias, cast

from matplotlib.figure import Figure

AnimationUpdate: TypeAlias = Callable[[int], None]


def attach_animation_metadata(
    fig: Figure,
    *,
    update: AnimationUpdate,
    n_points: int,
    initial_index: int = 0,
    extras: dict[str, object] | None = None,
    frame_label: str | None = None,
    frame_coordinates: object | None = None,
) -> Figure:
    """Attach Tk animation metadata to a figure and return it."""
    dynamic_figure = cast(Any, fig)
    dynamic_figure._animation_update = update
    dynamic_figure._animation_n_points = n_points
    dynamic_figure._animation_initial_index = initial_index

    if frame_label is not None:
        dynamic_figure._animation_frame_label = frame_label
    if frame_coordinates is not None:
        dynamic_figure._animation_frame_coordinates = frame_coordinates

    if extras is not None:
        for attr_name, value in extras.items():
            setattr(dynamic_figure, attr_name, value)

    return fig
