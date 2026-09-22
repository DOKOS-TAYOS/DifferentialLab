"""Pure result-helper tests for 3D aerodynamics views."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np

import complex_problems.aerodynamics_3d.result_dialog as result_dialog
from complex_problems.aerodynamics_3d.result_dialog import (
    _scaled_arrow_lengths,
    build_streamline_cache,
    prepare_slice_history,
    slice_center_index,
    slice_field,
    slice_index_count,
    trace_streamline_3d,
)
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
    slow = replace(result, u=result.u * 0.2)
    fast = replace(result, u=result.u * 5.0)
    line = trace_streamline_3d((0.2, 0.8, 0.8), result=slow, step_size=0.1, max_steps=10)
    fast_line = trace_streamline_3d((0.2, 0.8, 0.8), result=fast, step_size=0.1, max_steps=10)
    assert len(line) > 3
    np.testing.assert_allclose(line, fast_line, atol=1.0e-12)
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


def test_slice_selectors_and_prepared_history_are_plane_specific() -> None:
    result = _uniform_result()
    result = replace(
        result,
        t=np.array([0.0, 0.5, 1.0]),
        u=np.repeat(result.u, 3, axis=0),
        v=np.repeat(result.v, 3, axis=0),
        w=np.repeat(result.w, 3, axis=0),
        pressure=np.repeat(result.pressure, 3, axis=0),
        drag_coeff=np.zeros(3),
        lift_coeff=np.zeros(3),
        side_force_coeff=np.zeros(3),
        divergence_l2=np.zeros(3),
        max_speed=np.ones(3),
    )
    assert slice_index_count(result, "XY") == len(result.z)
    assert slice_index_count(result, "XZ") == len(result.y)
    assert slice_index_count(result, "YZ") == len(result.x)
    assert slice_center_index(result, "XZ") == len(result.y) // 2
    payload = prepare_slice_history(
        result,
        plane="XZ",
        index=slice_center_index(result, "XZ"),
        field="Speed",
    )
    assert payload.frames.shape == (len(result.t), len(result.z), len(result.x))
    assert payload.coordinate == result.y[len(result.y) // 2]


def test_streamline_cache_has_one_entry_per_saved_frame() -> None:
    result = _uniform_result()
    result = replace(
        result,
        t=np.array([0.0, 0.5, 1.0]),
        u=np.repeat(result.u, 3, axis=0),
        v=np.repeat(result.v, 3, axis=0),
        w=np.repeat(result.w, 3, axis=0),
    )
    cache = build_streamline_cache(result, density=2)
    assert cache.seed_density == 2
    assert cache.frame_count == len(result.t)
    assert len(cache.frames) == len(result.t)



def test_streamline_cache_reuses_one_velocity_interpolator_per_frame() -> None:
    result = _uniform_result()
    result = replace(
        result,
        t=np.array([0.0, 0.5, 1.0]),
        u=np.repeat(result.u, 3, axis=0),
        v=np.repeat(result.v, 3, axis=0),
        w=np.repeat(result.w, 3, axis=0),
    )
    with patch.object(
        result_dialog,
        "_velocity_interpolator",
        wraps=result_dialog._velocity_interpolator,
    ) as interpolator_factory:
        cache = build_streamline_cache(result, density=2)

    assert cache.frame_count == len(result.t)
    assert interpolator_factory.call_count == len(result.t)


def test_log_scaled_arrow_lengths_stay_within_fixed_bounds() -> None:
    magnitude = np.array([1.0e-6, 0.1, 0.5, 1.0, 5.0])
    lengths = _scaled_arrow_lengths(
        magnitude,
        1.0,
        minimum=0.04,
        maximum=0.16,
    )

    assert np.all(np.diff(lengths) >= 0.0)
    assert lengths[0] >= 0.04
    assert lengths[-1] == 0.16
    assert np.all(lengths <= 0.16)
