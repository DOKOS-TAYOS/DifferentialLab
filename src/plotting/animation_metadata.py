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
) -> Figure:
    """Attach Tk animation metadata to a figure and return it."""
    dynamic_figure = cast(Any, fig)
    dynamic_figure._animation_update = update
    dynamic_figure._animation_n_points = n_points
    dynamic_figure._animation_initial_index = initial_index

    if extras is not None:
        for attr_name, value in extras.items():
            setattr(dynamic_figure, attr_name, value)

    return fig
