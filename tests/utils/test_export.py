"""Tests for utils.export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from utils.export import (
    _export_csv,
    _export_json,
    _make_serializable,
    export_csv_to_path,
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

    def test_vector_pde_writes_components_and_magnitude(self, tmp_path: Path) -> None:
        x = np.array([0.0, 1.0])
        y_grid = np.array([0.0, 1.0])
        fields = np.array(
            [
                [[3.0, 0.0], [0.0, 5.0]],
                [[4.0, 2.0], [1.0, 12.0]],
            ]
        )
        filepath = tmp_path / "vector_pde.csv"

        export_csv_to_path(x, fields, filepath, y_grid=y_grid)

        rows = filepath.read_text().strip().splitlines()
        assert rows[0] == "x,y,f0,f1,magnitude"
        assert rows[1].endswith(",3.0,4.0,5.0")
        assert len(rows) == 5

    def test_scalar_pde_3d_writes_xyz_and_value(self, tmp_path: Path) -> None:
        x = np.array([0.0, 1.0])
        y_grid = np.array([0.0, 2.0])
        z_grid = np.array([0.0, 3.0])
        field = np.arange(8, dtype=float).reshape(2, 2, 2)
        filepath = tmp_path / "pde_3d.csv"

        export_csv_to_path(x, field, filepath, y_grid=y_grid, z_grid=z_grid)

        rows = filepath.read_text().strip().splitlines()
        assert rows[0] == "x,y,z,f"
        assert rows[-1] == "1.0,2.0,3.0,7.0"
        assert len(rows) == 9


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
