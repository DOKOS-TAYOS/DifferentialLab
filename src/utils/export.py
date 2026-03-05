"""Export utilities for CSV, JSON, and plot files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


def _ensure_parent_dir(filepath: Path) -> None:
    """Create parent directory if it does not exist."""
    filepath.parent.mkdir(parents=True, exist_ok=True)


def _export_csv(
    x: np.ndarray,
    y: np.ndarray,
    filepath: Path,
    headers: list[str] | None = None,
) -> Path:
    """Write solution data to a CSV file.

    Args:
        x: Independent variable values (1-D array).
        y: Solution values. Shape ``(n,)`` for first-order or ``(n_vars, n)``
            for systems.
        filepath: Destination file path.
        headers: Column headers. Auto-generated if ``None``.

    Returns:
        The path that was written.
    """
    y_2d = np.atleast_2d(y)
    # Ensure shape is (n_vars, n_points) where n_points == len(x)
    if y_2d.shape[-1] != len(x) and y_2d.shape[0] == len(x):
        y_2d = y_2d.T

    n_vars = y_2d.shape[0]
    if headers is None:
        headers = ["x"] + [f"f{i}" if n_vars > 1 else "f" for i in range(n_vars)]

    _ensure_parent_dir(filepath)
    data = np.column_stack((x, y_2d.T))
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data.tolist())

    logger.info("CSV exported: %s", filepath)
    return filepath


def _export_json(
    statistics: dict[str, Any],
    metadata: dict[str, Any],
    filepath: Path,
) -> Path:
    """Write statistics and metadata to a JSON file.

    Args:
        statistics: Computed magnitudes/statistics.
        metadata: Equation info, solver parameters, etc.
        filepath: Destination file path.

    Returns:
        The path that was written.
    """
    _ensure_parent_dir(filepath)
    payload = {
        "metadata": _make_serializable(metadata),
        "statistics": _make_serializable(statistics),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("JSON exported: %s", filepath)
    return filepath


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON.

    Args:
        obj: Object to convert.

    Returns:
        JSON-safe equivalent.
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _export_csv_2d(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    u: np.ndarray,
    filepath: Path,
) -> Path:
    """Write 2D solution data to CSV (x, y, u columns).

    Args:
        x_grid: 1D x values.
        y_grid: 1D y values.
        u: 2D array shape (len(y_grid), len(x_grid)).
        filepath: Destination path.

    Returns:
        Path written.
    """
    _ensure_parent_dir(filepath)
    X, Y = np.meshgrid(x_grid, y_grid)
    data = np.column_stack((X.ravel(), Y.ravel(), u.ravel()))
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "f"])
        writer.writerows(data.tolist())
    logger.info("CSV exported (2D): %s", filepath)
    return filepath


def export_csv_to_path(
    x: np.ndarray,
    y: np.ndarray,
    filepath: Path,
    *,
    y_grid: np.ndarray | None = None,
) -> Path:
    """Export solution data to CSV at the given path.

    Args:
        x: Independent variable values (1D) or x grid for 2D.
        y: Solution values. For 2D PDE: shape (ny, nx).
        filepath: Destination path.
        y_grid: For 2D PDE, the y grid. If provided with 2D y, uses 2D CSV format.

    Returns:
        The path that was written.
    """
    if y_grid is not None and y.ndim == 2:
        _export_csv_2d(x, y_grid, y, filepath)
    else:
        _export_csv(x, y, filepath)
    return filepath


def export_json_to_path(
    statistics: dict[str, Any],
    metadata: dict[str, Any],
    filepath: Path,
) -> Path:
    """Export statistics and metadata to JSON at the given path.

    Args:
        statistics: Computed magnitudes/statistics.
        metadata: Equation info, solver parameters, etc.
        filepath: Destination path.

    Returns:
        The path that was written.
    """
    return _export_json(statistics, metadata, filepath)
