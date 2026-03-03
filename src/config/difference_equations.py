"""Difference equation (recurrence) functions.

Functions here are callable as ``f(n, y, **params)`` and return the next value
(scalar). They can be referenced from equations.yaml via function_name for
equation_type: difference.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def geometric_growth(n: int, y: np.ndarray, r: float = 1.5, **kwargs: Any) -> float:
    """y_{n+1} = r * y_n — Geometric growth (discrete exponential).

    Args:
        n: Discrete index.
        y: State vector ``[y_n]``.
        r: Growth rate.
        **kwargs: Ignored.

    Returns:
        Next value y_{n+1}.
    """
    return r * y[0]


def logistic_map(n: int, y: np.ndarray, r: float = 3.5, **kwargs: Any) -> float:
    """y_{n+1} = r * y_n * (1 - y_n) — Logistic map (chaotic dynamics).

    Args:
        n: Discrete index.
        y: State vector ``[y_n]``.
        r: Growth parameter.
        **kwargs: Ignored.

    Returns:
        Next value y_{n+1}.
    """
    return r * y[0] * (1.0 - y[0])


def fibonacci(n: int, y: np.ndarray, **kwargs: Any) -> float:
    """y_{n+2} = y_{n+1} + y_n — Fibonacci recurrence.

    Args:
        n: Discrete index.
        y: State vector ``[y_n, y_{n+1}]``.
        **kwargs: Ignored.

    Returns:
        Next value y_{n+2}.
    """
    return y[1] + y[0]


def linear_recurrence_2(
    n: int, y: np.ndarray, a: float = 1.0, b: float = 1.0, **kwargs: Any
) -> float:
    """y_{n+2} = a*y_{n+1} + b*y_n — Second-order linear recurrence.

    Args:
        n: Discrete index.
        y: State vector ``[y_n, y_{n+1}]``.
        a: Coefficient for y_{n+1}.
        b: Coefficient for y_n.
        **kwargs: Ignored.

    Returns:
        Next value y_{n+2}.
    """
    return a * y[1] + b * y[0]


def cobweb_model(n: int, y: np.ndarray, r: float = 2.5, K: float = 1.0, **kwargs: Any) -> float:
    """y_{n+1} = r * y_n * (1 - y_n/K) — Discrete logistic (Ricker-type).

    Args:
        n: Discrete index.
        y: State vector ``[y_n]``.
        r: Growth rate.
        K: Carrying capacity.
        **kwargs: Ignored.

    Returns:
        Next value y_{n+1}.
    """
    return r * y[0] * (1.0 - y[0] / K)
