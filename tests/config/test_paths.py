"""Tests for config.paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from config.paths import (
    generate_output_basename,
    get_csv_path,
    get_json_path,
    get_plot_path,
    get_output_dir,
    get_env_path,
)


@patch("config.paths.get_env_from_schema")
def test_generate_output_basename(mock_get_env: object) -> None:
    basename = generate_output_basename(prefix="solution")
    assert basename.startswith("solution_")
    assert "_" in basename
    # Contains something like 20260220_123456
    parts = basename.split("_")
    assert len(parts) >= 3  # prefix, date, time


@patch("config.paths.get_env_from_schema")
@patch("config.paths.get_output_dir")
def test_get_plot_path(mock_output_dir: object, mock_get_env: object, tmp_path: Path) -> None:
    mock_output_dir.return_value = tmp_path
    mock_get_env.side_effect = lambda k: "png" if k == "FILE_PLOT_FORMAT" else str(tmp_path)
    path = get_plot_path("solution_20260220_120000")
    assert path == tmp_path / "solution_20260220_120000.png"


@patch("config.paths.get_output_dir")
def test_get_csv_path(mock_output_dir: object, tmp_path: Path) -> None:
    mock_output_dir.return_value = tmp_path
    path = get_csv_path("solution_20260220_120000")
    assert path == tmp_path / "solution_20260220_120000.csv"


@patch("config.paths.get_output_dir")
def test_get_json_path(mock_output_dir: object, tmp_path: Path) -> None:
    mock_output_dir.return_value = tmp_path
    path = get_json_path("solution_20260220_120000")
    assert path == tmp_path / "solution_20260220_120000.json"


@patch("config.paths.get_env_from_schema")
def test_get_output_dir_creates_dir(mock_get_env: object, tmp_path: Path) -> None:
    sub = tmp_path / "out"
    mock_get_env.return_value = str(sub)
    result = get_output_dir()
    assert result.exists()
    assert result == sub or (result == tmp_path / "out")


def test_get_env_path_returns_path() -> None:
    path = get_env_path()
    assert isinstance(path, Path)
    assert path.name == ".env"
