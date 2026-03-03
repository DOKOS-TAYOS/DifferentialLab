"""Tests for solver.difference_solver."""

from __future__ import annotations

import numpy as np

from solver.difference_solver import solve_difference


def test_geometric_growth() -> None:
    """Geometric growth y_{n+1} = r * y_n: verify y_n = r^n * y_0."""
    r = 1.5
    y0_val = 2.0

    def recur(n: int, y: np.ndarray) -> float:
        return r * y[0]

    result = solve_difference(recur, n_min=0, n_max=20, y0=[y0_val], order=1)
    assert result.success is True
    assert result.y.shape == (1, 21)
    np.testing.assert_allclose(result.y[0, 0], y0_val)
    np.testing.assert_allclose(result.y[0, -1], y0_val * (r**20))
    np.testing.assert_array_equal(result.n, np.arange(0, 21, dtype=float))


def test_fibonacci() -> None:
    """Fibonacci y_{n+2} = y_{n+1} + y_n with y_0=0, y_1=1."""
    def recur(n: int, y: np.ndarray) -> float:
        return y[0] + y[1]

    result = solve_difference(recur, n_min=0, n_max=10, y0=[0.0, 1.0], order=2)
    assert result.success is True
    assert result.y.shape == (2, 11)
    # Fibonacci: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55
    expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    np.testing.assert_allclose(result.y[0, :], expected)
    np.testing.assert_allclose(result.y[1, :], expected[1:] + [89])


def test_n_min_ge_n_max_returns_failure() -> None:
    """n_min >= n_max should return success=False."""
    def recur(n: int, y: np.ndarray) -> float:
        return y[0]

    result = solve_difference(recur, n_min=5, n_max=5, y0=[1.0], order=1)
    assert result.success is False
    assert "n_min" in result.message.lower() or "n_max" in result.message.lower()
    assert len(result.n) == 0
    assert result.y.size == 0

    result2 = solve_difference(recur, n_min=10, n_max=5, y0=[1.0], order=1)
    assert result2.success is False


def test_exception_in_recur_func_propagates() -> None:
    """Exception in recur_func should set success=False and propagate message."""
    def recur_failing(n: int, y: np.ndarray) -> float:
        raise ValueError("custom error")

    result = solve_difference(recur_failing, n_min=0, n_max=5, y0=[1.0], order=1)
    assert result.success is False
    assert "custom error" in result.message
    assert len(result.n) >= 1  # Partial result up to failure point
