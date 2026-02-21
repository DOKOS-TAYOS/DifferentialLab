"""Solver for difference equations (recurrence relations)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from utils import get_logger

logger = get_logger(__name__)


@dataclass
class DifferenceSolution:
    """Container for difference equation solution data.

    Attributes:
        n: Discrete index values (0, 1, 2, ...).
        y: Solution array — shape ``(n_vars, n_points)``.
        success: Whether the iteration completed without error.
        message: Status message.
    """

    n: np.ndarray
    y: np.ndarray
    success: bool
    message: str


def solve_difference(
    recur_func: Callable[[int, np.ndarray], float],
    n_min: int,
    n_max: int,
    y0: list[float],
    order: int,
) -> DifferenceSolution:
    """Solve a difference equation by iterating the recurrence.

    The recurrence has the form y_{n+order} = f(n, [y_n, y_{n+1}, ..., y_{n+order-1}]).
    State vector at step n: y[0]=y_n, y[1]=y_{n+1}, ..., y[order-1]=y_{n+order-1}.

    Args:
        recur_func: Function (n, y) -> next value y_{n+order}.
        n_min: Start index (inclusive).
        n_max: End index (inclusive).
        y0: Initial conditions [y_0, y_1, ..., y_{order-1}].
        order: Order of the recurrence.

    Returns:
        A :class:`DifferenceSolution` with n and y arrays.
        y has shape (order, n_points) for compatibility with ODE pipeline.
    """
    if n_min >= n_max:
        return DifferenceSolution(
            n=np.array([]),
            y=np.array([]).reshape(0, 0),
            success=False,
            message="n_min must be less than n_max",
        )

    n_points = n_max - n_min + 1
    n_arr = np.arange(n_min, n_max + 1, dtype=float)
    y_arr = np.zeros((order, n_points))
    y_arr[:, 0] = y0[:order]

    state = np.array(y0[:order], dtype=float)
    last_i = 0

    try:
        for i in range(1, n_points):
            last_i = i
            n_curr = n_min + i - 1
            next_val = float(recur_func(n_curr, state))
            state = np.roll(state, -1)
            state[-1] = next_val
            y_arr[:, i] = state

        logger.info(
            "Difference equation solved: %d points from n=%d to n=%d",
            n_points, n_min, n_max,
        )
        return DifferenceSolution(
            n=n_arr,
            y=y_arr,
            success=True,
            message="Solved successfully",
        )
    except Exception as exc:
        logger.error("Difference equation iteration failed: %s", exc)
        return DifferenceSolution(
            n=n_arr[: last_i + 1],
            y=y_arr[:, : last_i + 1],
            success=False,
            message=str(exc),
        )
