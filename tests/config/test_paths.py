"""Tests for config.paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from config.paths import (
    generate_output_basename,
    get_csv_path,
    get_env_path,
    get_output_dir,
)


def test_generate_output_basename() -> None:
    basename = generate_output_basename(prefix="solution")
    assert basename.startswith("solution_")
    assert "_" in basename
    # Contains something like 20260220_123456
    parts = basename.split("_")
    assert len(parts) >= 3  # prefix, date, time


@patch("config.paths.get_output_dir")
def test_get_csv_path(mock_output_dir: object, tmp_path: Path) -> None:
    mock_output_dir.return_value = tmp_path
    path = get_csv_path("solution_20260220_120000")
    assert path == tmp_path / "solution_20260220_120000.csv"


def test_get_output_dir_creates_dir(tmp_path: Path) -> None:
    with patch("config.paths._PROJECT_ROOT", tmp_path):
        result = get_output_dir()
    assert result.exists()
    assert result == tmp_path / "output"


def test_get_env_path_returns_path() -> None:
    path = get_env_path()
    assert isinstance(path, Path)
    assert path.name == ".env"
