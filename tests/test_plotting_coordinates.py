"""Non-GUI coverage for coordinate-aware plotting data preparation."""

from __future__ import annotations

import numpy as np
import pytest

from plotting import create_polar_contour_plot, create_vector_field_plot
from plotting.coordinates import (
    cartesian_to_cylindrical,
    cartesian_to_polar,
    cartesian_to_spherical,
    cartesian_vector_to_cylindrical,
    cartesian_vector_to_polar,
    cartesian_vector_to_spherical,
    cylindrical_to_cartesian,
    cylindrical_vector_to_cartesian,
    extract_scalar_3d_slice,
    polar_to_cartesian,
    polar_vector_to_cartesian,
    prepare_scalar_axis_sweep_2d,
    prepare_scalar_axis_sweep_3d,
    resample_scalar_to_polar,
    spherical_to_cartesian,
    spherical_vector_to_cartesian,
    vector_field_data,
)


def test_coordinate_round_trips_away_from_singularities() -> None:
    """Planar, cylindrical, and spherical coordinates recover Cartesian data."""
    x = np.array([1.0, -2.0])
    y = np.array([2.0, 3.0])
    z = np.array([3.0, -4.0])
    radius, angle = cartesian_to_polar(x, y)
    np.testing.assert_allclose(polar_to_cartesian(radius, angle), (x, y))
    cylindrical = cartesian_to_cylindrical(x, y, z)
    np.testing.assert_allclose(cylindrical_to_cartesian(*cylindrical), (x, y, z))
    spherical = cartesian_to_spherical(x, y, z)
    np.testing.assert_allclose(spherical_to_cartesian(*spherical), (x, y, z))


def test_polar_vector_round_trip_and_origin_basis_are_deterministic() -> None:
    """The documented origin basis avoids ambiguous divisions and warnings."""
    vx = np.array([3.0, -1.0])
    vy = np.array([4.0, 2.0])
    radial, tangential = cartesian_vector_to_polar(vx, vy, [0.0, 1.0], [0.0, 0.0])
    np.testing.assert_allclose((radial[0], tangential[0]), (3.0, 4.0))
    np.testing.assert_allclose(
        polar_vector_to_cartesian(radial, tangential, [0.0, 1.0], [0.0, 0.0]), (vx, vy)
    )
    spherical = cartesian_to_spherical(0.0, 0.0, 0.0)
    np.testing.assert_allclose(spherical, (0.0, 0.0, 0.0))


def test_3d_vector_component_round_trips() -> None:
    """Cylindrical and spherical component conversions preserve vectors."""
    x, y, z = np.array([1.0]), np.array([2.0]), np.array([3.0])
    vx, vy, vz = np.array([4.0]), np.array([-2.0]), np.array([1.0])
    cylindrical = cartesian_vector_to_cylindrical(vx, vy, vz, x, y)
    np.testing.assert_allclose(cylindrical_vector_to_cartesian(*cylindrical, x, y), (vx, vy, vz))
    spherical = cartesian_vector_to_spherical(vx, vy, vz, x, y, z)
    np.testing.assert_allclose(spherical_vector_to_cartesian(*spherical, x, y, z), (vx, vy, vz))


def test_cylindrical_vector_transforms_broadcast_scalar_axial_component() -> None:
    """Cylindrical vector conversions broadcast an axial scalar over a 2D grid."""
    x, y = np.meshgrid(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    vx = x + y
    vy = x - y
    axial = 7.0

    cylindrical = cartesian_vector_to_cylindrical(vx, vy, axial, x, y)
    assert all(component.shape == x.shape for component in cylindrical)
    cartesian = cylindrical_vector_to_cartesian(*cylindrical, x, y)
    assert all(component.shape == x.shape for component in cartesian)
    np.testing.assert_allclose(cartesian, (vx, vy, np.full_like(x, axial)))

    inverse_with_scalar_axial = cylindrical_vector_to_cartesian(
        cylindrical[0], cylindrical[1], axial, x, y
    )
    assert all(component.shape == x.shape for component in inverse_with_scalar_axial)
    np.testing.assert_allclose(inverse_with_scalar_axial, (vx, vy, np.full_like(x, axial)))


def test_polar_resampling_preserves_shape_and_invalid_regions() -> None:
    """Polar display data has predictable dimensions and keeps source masks."""
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 5)
    values = np.add.outer(y, x)
    values[2, 2] = np.nan
    sampled = resample_scalar_to_polar(x, y, values, radial_points=7, angular_points=8)
    assert sampled.values.shape == (8, 7)
    assert np.isnan(sampled.values[:, 0]).all()
    assert np.isnan(sampled.values).any()


def test_polar_contour_plot_constructs_from_cartesian_scalar_grid() -> None:
    """Polar scalar plots construct successfully from a Cartesian grid."""
    import matplotlib.pyplot as plt

    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 5)
    values = np.add.outer(y, x)

    try:
        figure = create_polar_contour_plot(x, y, values, radial_points=7, angular_points=8)
    finally:
        plt.close("all")

    assert figure is not None


def test_vector_plot_data_computes_magnitude_and_validates_shape() -> None:
    """Quiver-ready fields retain grids and expose the Euclidean magnitude."""
    components = np.array([[[3.0, 0.0]], [[4.0, 5.0]]])
    field = vector_field_data([0.0, 1.0], [0.0], components)
    np.testing.assert_allclose(field.magnitude, [[5.0, 5.0]])


def test_three_component_fields_reject_planar_views_but_support_other_views() -> None:
    """Only exactly two components may be rendered as a planar vector field."""
    components = np.ones((3, 2, 2))
    with pytest.raises(ValueError, match="shape \\(2, len\\(y\\), len\\(x\\)\\)"):
        vector_field_data([0.0, 1.0], [0.0, 1.0], components)

    for view in ("components", "magnitude"):
        figure = create_vector_field_plot(
            np.array([0.0, 1.0]), np.array([0.0, 1.0]), components, view=view
        )
        figure.clf()
    for view in ("quiver", "stream", "radial_tangential"):
        with pytest.raises(ValueError, match="requires exactly two vector components"):
            create_vector_field_plot(
                np.array([0.0, 1.0]), np.array([0.0, 1.0]), components, view=view
            )


def test_extract_scalar_3d_slice_uses_documented_public_axis_order() -> None:
    """XZ extraction returns values in ``(z, x)`` contour order."""
    values = np.arange(24.0).reshape(2, 3, 4)
    extracted = extract_scalar_3d_slice([0, 1, 2, 3], [0, 1, 2], [0, 1], values, "XZ", 1)
    np.testing.assert_array_equal(extracted.values, values[:, 1, :])
    assert extracted.fixed_coordinate == 1.0


def test_scalar_axis_sweep_2d_preserves_both_profile_orientations() -> None:
    """2D sweeps use the spatial axis as frames without changing field order."""
    x = np.array([10.0, 20.0, 30.0, 40.0])
    y = np.array([1.0, 2.0, 3.0])
    values = np.arange(12.0).reshape(3, 4)

    x_sweep = prepare_scalar_axis_sweep_2d(x, y, values, "x")
    assert x_sweep.sweep_axis == "x"
    np.testing.assert_array_equal(x_sweep.sweep_coordinates, x)
    np.testing.assert_array_equal(x_sweep.axis_1, y)
    assert x_sweep.axis_2 is None
    np.testing.assert_array_equal(x_sweep.frames, values.T)

    y_sweep = prepare_scalar_axis_sweep_2d(x, y, values, "y")
    assert y_sweep.sweep_axis == "y"
    np.testing.assert_array_equal(y_sweep.sweep_coordinates, y)
    np.testing.assert_array_equal(y_sweep.axis_1, x)
    assert y_sweep.axis_2 is None
    np.testing.assert_array_equal(y_sweep.frames, values)


def test_scalar_axis_sweep_3d_matches_public_slice_order() -> None:
    """3D sweep frames retain the public ``(z, y, x)`` storage convention."""
    x = np.arange(4.0)
    y = np.arange(3.0)
    z = np.arange(2.0)
    values = np.arange(24.0).reshape(2, 3, 4)

    x_sweep = prepare_scalar_axis_sweep_3d(x, y, z, values, "x")
    np.testing.assert_array_equal(x_sweep.frames, np.moveaxis(values, 2, 0))
    np.testing.assert_array_equal(x_sweep.axis_1, y)
    np.testing.assert_array_equal(x_sweep.axis_2, z)
    assert x_sweep.frames.shape == (4, 2, 3)

    y_sweep = prepare_scalar_axis_sweep_3d(x, y, z, values, "y")
    np.testing.assert_array_equal(y_sweep.frames, np.moveaxis(values, 1, 0))
    np.testing.assert_array_equal(y_sweep.axis_1, x)
    np.testing.assert_array_equal(y_sweep.axis_2, z)
    assert y_sweep.frames.shape == (3, 2, 4)

    z_sweep = prepare_scalar_axis_sweep_3d(x, y, z, values, "z")
    np.testing.assert_array_equal(z_sweep.frames, values)
    np.testing.assert_array_equal(z_sweep.axis_1, x)
    np.testing.assert_array_equal(z_sweep.axis_2, y)
    assert z_sweep.frames.shape == (2, 3, 4)


@pytest.mark.parametrize(
    ("helper", "arguments"),
    [
        (prepare_scalar_axis_sweep_2d, ([0.0, 1.0], [0.0], np.zeros((2, 2)), "x")),
        (
            prepare_scalar_axis_sweep_3d,
            ([0.0], [0.0], [0.0], np.zeros((2, 1, 1)), "z"),
        ),
    ],
)
def test_scalar_axis_sweeps_reject_invalid_shapes(
    helper: object, arguments: tuple[object, ...]
) -> None:
    with pytest.raises(ValueError, match="shape"):
        helper(*arguments)  # type: ignore[operator]


def test_scalar_axis_sweeps_reject_invalid_axes() -> None:
    with pytest.raises(ValueError, match="sweep_axis"):
        prepare_scalar_axis_sweep_2d([0.0], [0.0], [[1.0]], "z")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sweep_axis"):
        prepare_scalar_axis_sweep_3d([0.0], [0.0], [0.0], [[[1.0]]], "q")  # type: ignore[arg-type]
