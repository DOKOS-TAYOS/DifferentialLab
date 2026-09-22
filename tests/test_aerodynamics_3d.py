"""Focused numerical tests for the 3D aerodynamics Advanced Problem."""

from __future__ import annotations

import numpy as np

from complex_problems.aerodynamics_3d import model
from complex_problems.aerodynamics_3d.solver import (
    _wave_numbers,
    project_velocity_fft,
    solve_aerodynamics_3d,
)


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 4.0, 16, endpoint=False)
    y = np.linspace(0.0, 2.0, 12, endpoint=False)
    z = np.linspace(0.0, 2.0, 10, endpoint=False)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return xx, yy, zz


def test_supported_obstacle_masks_and_reference_scales_are_nonempty() -> None:
    X, Y, Z = _grid()
    common = dict(X=X, Y=Y, Z=Z, center_x=1.5, center_y=1.0, center_z=1.0)
    cases = (
        {"shape": "sphere", "diameter": 0.5},
        {"shape": "ellipsoid", "size_x": 0.8, "size_y": 0.5, "size_z": 0.4},
        {"shape": "box", "size_x": 0.8, "size_y": 0.5, "size_z": 0.4},
        {"shape": "naca0012_wing", "chord": 0.8, "span": 0.8},
    )
    for case in cases:
        mask, area, length = model.build_obstacle_mask(**common, **case)
        assert mask.shape == X.shape
        assert np.any(mask)
        assert np.isfinite(area) and area > 0
        assert np.isfinite(length) and length > 0


def test_attack_angle_changes_a_non_axisymmetric_mask() -> None:
    X, Y, Z = _grid()
    kwargs = dict(
        shape="box",
        X=X,
        Y=Y,
        Z=Z,
        center_x=1.5,
        center_y=1.0,
        center_z=1.0,
        size_x=1.0,
        size_y=0.4,
        size_z=0.4,
    )
    mask_zero, _, _ = model.build_obstacle_mask(**kwargs, attack_deg=0.0)
    mask_rotated, _, _ = model.build_obstacle_mask(**kwargs, attack_deg=27.0)
    assert np.any(mask_zero)
    assert np.any(mask_rotated)
    assert not np.array_equal(mask_zero, mask_rotated)


def test_fft_projection_removes_nontrivial_spectral_divergence() -> None:
    nx, ny, nz = 12, 10, 8
    dx, dy, dz = 4.0 / nx, 2.0 / ny, 2.0 / nz
    kx, ky, kz, k2 = _wave_numbers(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
    rng = np.random.default_rng(42)
    u_star, v_star, w_star = rng.normal(size=(3, nz, ny, nx))
    before = np.sqrt(
        np.mean(
            np.real(
                np.fft.ifftn(
                    1j
                    * (
                        kx * np.fft.fftn(u_star)
                        + ky * np.fft.fftn(v_star)
                        + kz * np.fft.fftn(w_star)
                    )
                )
            )
            ** 2
        )
    )
    u, v, w, pressure = project_velocity_fft(
        u_star, v_star, w_star, dt=0.01, kx=kx, ky=ky, kz=kz, k2=k2
    )
    after = model.spectral_divergence_l2(u, v, w, kx, ky, kz)
    assert before > 1.0e-3
    assert after < 1.0e-12
    assert np.all(np.isfinite(pressure))
    np.testing.assert_allclose(np.mean(u), np.mean(u_star))
    np.testing.assert_allclose(np.mean(v), np.mean(v_star))
    np.testing.assert_allclose(np.mean(w), np.mean(w_star))


def test_short_stokes_and_nonlinear_runs_store_only_core_volumes() -> None:
    for approximation in ("stokes", "nonlinear_ns"):
        result = solve_aerodynamics_3d(
            approximation=approximation,
            nx=8,
            ny=8,
            nz=8,
            lx=2.0,
            ly=2.0,
            lz=2.0,
            t_max=0.006,
            dt=0.002,
            sample_every=2,
            nu=0.03,
            penalization=0.01,
            obstacle_diameter=0.5,
        )
        assert result.u.shape == (3, 8, 8, 8)
        assert result.v.shape == result.w.shape == result.pressure.shape == result.u.shape
        assert result.obstacle_mask.shape == (8, 8, 8)
        assert (
            result.drag_coeff.shape
            == result.lift_coeff.shape
            == result.side_force_coeff.shape
            == (3,)
        )
        assert result.divergence_l2.shape == result.max_speed.shape == (3,)
        assert all(
            np.all(np.isfinite(array))
            for array in (
                result.u,
                result.v,
                result.w,
                result.pressure,
                result.drag_coeff,
                result.divergence_l2,
            )
        )
        assert not hasattr(result, "speed")
        assert not hasattr(result, "vorticity")
