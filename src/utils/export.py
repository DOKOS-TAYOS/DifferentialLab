"""Export utilities for CSV, JSON, and plot files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


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
        headers = ["x"] + [f"y{i}" if n_vars > 1 else "y" for i in range(n_vars)]

    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([x] + [y_2d[i] for i in range(n_vars)])
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
    filepath.parent.mkdir(parents=True, exist_ok=True)
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
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "u"])
        for j, yv in enumerate(y_grid):
            for i, xv in enumerate(x_grid):
                writer.writerow([xv, yv, u[j, i]])
    return filepath


def export_all_results(
    x: np.ndarray,
    y: np.ndarray,
    statistics: dict[str, Any],
    metadata: dict[str, Any],
    csv_path: Path,
    json_path: Path,
    *,
    y_grid: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Export both CSV data and JSON statistics.

    Args:
        x: Independent variable values (1D) or x grid for 2D.
        y: Solution values. For 2D PDE: shape (ny, nx).
        statistics: Computed statistics dict.
        metadata: Equation/solver metadata dict.
        csv_path: CSV destination.
        json_path: JSON destination.
        y_grid: For 2D PDE, the y grid. If provided with 2D y, uses 2D CSV.

    Returns:
        Tuple of ``(csv_path, json_path)`` that were written.
    """
    if y_grid is not None and y.ndim == 2:
        _export_csv_2d(x, y_grid, y, csv_path)
    else:
        _export_csv(x, y, csv_path)
    _export_json(statistics, metadata, json_path)
    return csv_path, json_path
