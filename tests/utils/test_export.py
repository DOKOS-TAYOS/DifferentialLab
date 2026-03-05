"""Tests for utils.export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from utils.export import (
    _export_csv,
    _export_json,
    _make_serializable,
)


class TestMakeSerializable:
    def test_dict_recursive(self) -> None:
        obj = {"a": np.int32(1), "b": [np.float64(2.0)]}
        result = _make_serializable(obj)
        assert result == {"a": 1, "b": [2.0]}

    def test_ndarray_to_list(self) -> None:
        arr = np.array([1.0, 2.0])
        assert _make_serializable(arr) == [1.0, 2.0]

    def test_path_to_str(self) -> None:
        p = Path("/some/file.txt")
        result = _make_serializable(p)
        assert isinstance(result, str)
        assert result.endswith("file.txt")
        assert "some" in result or "file.txt" in result

    def test_numpy_scalars(self) -> None:
        assert _make_serializable(np.int64(42)) == 42
        assert _make_serializable(np.float32(3.14)) == pytest.approx(3.14)


class TestExportCsv:
    def test_writes_headers_and_data(self, tmp_path: Path) -> None:
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([[1.0, 2.0, 3.0]])
        filepath = tmp_path / "out.csv"
        result = _export_csv(x, y, filepath)
        assert result == filepath
        content = filepath.read_text()
        assert "x" in content and "f" in content
        lines = content.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows

    def test_two_variables_two_columns(self, tmp_path: Path) -> None:
        x = np.array([0.0, 1.0])
        y = np.array([[1.0, 2.0], [0.0, 1.0]])
        filepath = tmp_path / "out.csv"
        _export_csv(x, y, filepath)
        content = filepath.read_text()
        assert "f0" in content and "f1" in content


class TestExportJson:
    def test_writes_metadata_and_statistics(self, tmp_path: Path) -> None:
        filepath = tmp_path / "out.json"
        _export_json(
            statistics={"mean": 1.0, "rms": 2.0},
            metadata={"equation_name": "Test", "order": 1},
            filepath=filepath,
        )
        assert filepath.exists()
        import json

        data = json.loads(filepath.read_text())
        assert "metadata" in data and "statistics" in data
        assert data["metadata"]["equation_name"] == "Test"
        assert data["statistics"]["mean"] == 1.0
