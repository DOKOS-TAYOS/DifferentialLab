"""Pure result-helper tests for 3D aerodynamics views."""

from __future__ import annotations

import numpy as np

from complex_problems.aerodynamics_3d.result_dialog import slice_field, trace_streamline_3d
from complex_problems.aerodynamics_3d.solver import Aerodynamics3DResult


def _uniform_result() -> Aerodynamics3DResult:
    x = np.linspace(0.0, 4.0, 16, endpoint=False)
    y = np.linspace(0.0, 2.0, 8, endpoint=False)
    z = np.linspace(0.0, 2.0, 8, endpoint=False)
    shape = (1, len(z), len(y), len(x))
    return Aerodynamics3DResult(
        x=x,
        y=y,
        z=z,
        t=np.array([0.0]),
        u=np.ones(shape),
        v=np.zeros(shape),
        w=np.zeros(shape),
        pressure=np.zeros(shape),
        obstacle_mask=np.zeros(shape[1:], dtype=bool),
        drag_coeff=np.zeros(1),
        lift_coeff=np.zeros(1),
        side_force_coeff=np.zeros(1),
        divergence_l2=np.zeros(1),
        max_speed=np.ones(1),
    )


def test_uniform_positive_x_streamline_is_straight() -> None:
    result = _uniform_result()
    line = trace_streamline_3d((0.2, 0.8, 0.8), result=result, step_size=0.1, max_steps=10)
    assert len(line) > 3
    np.testing.assert_allclose(line[:, 1], line[0, 1], atol=1.0e-12)
    np.testing.assert_allclose(line[:, 2], line[0, 2], atol=1.0e-12)
    assert np.all(np.diff(line[:, 0]) > 0)


def test_slice_helper_returns_requested_physical_coordinate() -> None:
    result = _uniform_result()
    x_axis, y_axis, values, coordinate = slice_field(
        result, frame=0, plane="XY", index=3, field="Speed"
    )
    np.testing.assert_array_equal(x_axis, result.x)
    np.testing.assert_array_equal(y_axis, result.y)
    assert values.shape == (len(result.y), len(result.x))
    assert coordinate.startswith("z =")
    np.testing.assert_allclose(values, 1.0)
