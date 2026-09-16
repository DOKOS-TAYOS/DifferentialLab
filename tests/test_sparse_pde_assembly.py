"""Regression tests for fixed-capacity sparse PDE COO workspaces."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from solver.pde_3d_solver import _assemble_pde_3d, _prepare_boundary_3d
from solver.pde_assembly import assemble_vector_pde
from solver.pde_boundary import prepare_vector_boundaries
from solver.pde_types import (
    PDEBoundaryConditions3D,
    PDECoefficients3D,
    VectorPDEBoundaryConditions,
    VectorPDECoefficients,
)


def test_vector_and_3d_assemblers_pass_trimmed_numpy_coo_arrays(monkeypatch: Any) -> None:
    """COO construction receives compact NumPy views rather than Python lists."""
    captured: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    original_coo_matrix = sparse.coo_matrix

    def capture_coo(
        entries: tuple[np.ndarray, tuple[np.ndarray, np.ndarray]],
        *,
        shape: tuple[int, int],
    ) -> Any:
        """Record the COO arrays while preserving SciPy's conversion behavior."""
        data, (rows, cols) = entries
        captured.append((data, rows, cols))
        return original_coo_matrix(entries, shape=shape)

    monkeypatch.setattr(sparse, "coo_matrix", capture_coo)

    vector_points = 4
    vector_x = np.linspace(0.0, 1.0, vector_points, endpoint=False)
    vector_y = np.linspace(0.0, 1.0, vector_points, endpoint=False)
    vector_boundaries = prepare_vector_boundaries(
        VectorPDEBoundaryConditions(periodic_x=True, periodic_y=True),
        components=2,
        mask=None,
        x=vector_x,
        y=vector_y,
    )
    vector_coefficients = VectorPDECoefficients(
        fxx=np.eye(2),
        fxy=np.zeros((2, 2)),
        fyy=np.eye(2),
        fx=np.zeros((2, 2)),
        fy=np.zeros((2, 2)),
        f=np.zeros((2, 2)),
        constant=np.zeros(2),
    )

    def vector_coefficient_at(_x: float, _y: float) -> tuple[VectorPDECoefficients, int]:
        """Return constant vector coefficients with zero-skipped couplings."""
        return vector_coefficients, 1

    vector_assembled = assemble_vector_pde(
        vector_x,
        vector_y,
        hx=1.0 / vector_points,
        hy=1.0 / vector_points,
        boundaries=vector_boundaries,
        coefficient_at=vector_coefficient_at,
    )

    three_dimensional_points = 4
    three_dimensional_x = np.linspace(0.0, 1.0, three_dimensional_points, endpoint=False)
    three_dimensional_y = np.linspace(0.0, 1.0, three_dimensional_points, endpoint=False)
    three_dimensional_z = np.linspace(0.0, 1.0, three_dimensional_points, endpoint=False)
    three_dimensional_boundary = _prepare_boundary_3d(
        PDEBoundaryConditions3D(periodic_x=True, periodic_y=True, periodic_z=True),
        x=three_dimensional_x,
        y=three_dimensional_y,
        z=three_dimensional_z,
        hx=1.0 / three_dimensional_points,
        hy=1.0 / three_dimensional_points,
        hz=1.0 / three_dimensional_points,
    )
    three_dimensional_coefficients: PDECoefficients3D = (
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    def three_dimensional_coefficient_at(
        _x: float,
        _y: float,
        _z: float,
    ) -> tuple[PDECoefficients3D, int]:
        """Return constant scalar coefficients with zero-skipped mixed terms."""
        return three_dimensional_coefficients, 1

    three_dimensional_assembled = _assemble_pde_3d(
        three_dimensional_x,
        three_dimensional_y,
        three_dimensional_z,
        hx=1.0 / three_dimensional_points,
        hy=1.0 / three_dimensional_points,
        hz=1.0 / three_dimensional_points,
        boundary=three_dimensional_boundary,
        coefficient_at=three_dimensional_coefficient_at,
    )

    vector_entries, three_dimensional_entries = captured
    assert all(isinstance(array, np.ndarray) for arrays in captured for array in arrays)
    assert vector_entries[0].size == 5 * vector_assembled.matrix.shape[0]
    assert three_dimensional_entries[0].size == 7 * three_dimensional_assembled.matrix.shape[0]
    assert vector_entries[0].base is not None
    assert three_dimensional_entries[0].base is not None
