# --- spherical detector tests for DirectionalRadiosityFast -------------------

import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
import sparrowpy as sp


def _six_axis_pf_coords():
    """±x, ±y, ±z as pf.Coordinates (cartesian)."""
    x = np.array([ 1, -1,  0,  0,  0,  0], float)
    y = np.array([ 0,  0,  1, -1,  0,  0], float)
    z = np.array([ 0,  0,  0,  0,  1, -1], float)
    # Normalize (robust even if you change vectors later)
    n = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x / n, y / n, z / n
    return pf.Coordinates(x, y, z)  # same pattern you use elsewhere


def test_spherical_detector_shape_and_time_axis():
    """Returns pf.TimeData with shape (R,D,B,S) and preserves the time axis."""
    walls = sp.testing.shoebox_room_stub(1, 1, 1)
    radiosity = sp.DirectionalRadiosityFast.from_polygon(walls, 1)

    radiosity.bake_geometry()
    radiosity.init_source_energy(pf.Coordinates(.5, .5, .5))
    radiosity.calculate_energy_exchange(343, 1/1000, 1, 3)

    # Two receivers → pass as pf.Coordinates with vectorized xyz
    receivers = pf.Coordinates(
        np.array([.25, .75]),
        np.array([.25, .50]),
        np.array([.25, .25]),
    )
    detector_sphere = _six_axis_pf_coords()  # D = 6

    td_sph   = radiosity.collect_energy_at_spherical_detector(
        receivers, detector_sphere, direct_sound=False)
    td_patch = radiosity.collect_energy_receiver_patchwise(receivers)

    # Infer sizes from patchwise output (R,P,B,S); D from detector_sphere
    R, P, B, S = td_patch.time.shape
    D = detector_sphere.get_cart().shape[0]

    assert isinstance(td_sph, pf.TimeData)
    assert td_sph.time.shape == (R, D, B, S)
    npt.assert_array_equal(td_sph.times, td_patch.times)


def test_spherical_sum_matches_patch_sum_without_direct_sound():
    """Sum over detector directions equals sum over patches (per R,B,S)."""
    walls = sp.testing.shoebox_room_stub(1, 1, 1)
    radiosity = sp.DirectionalRadiosityFast.from_polygon(walls, 1)

    radiosity.bake_geometry()
    radiosity.init_source_energy(pf.Coordinates(.5, .5, .5))
    radiosity.calculate_energy_exchange(343, 1/1000, 1, 3)

    receivers = pf.Coordinates(.40, .40, .40)  # R = 1
    detector_sphere = _six_axis_pf_coords()    # D = 6

    td_sph   = radiosity.collect_energy_at_spherical_detector(
        receivers, detector_sphere, direct_sound=False)
    td_patch = radiosity.collect_energy_receiver_patchwise(receivers)

    # Values should be finite and non-zero somewhere
    assert np.isfinite(td_sph.time).all()
    assert np.isfinite(td_patch.time).all()
    assert np.any(td_sph.time != 0)
    assert np.any(td_patch.time != 0)

    # print("td_sph.time shape:", td_sph.time.shape)
    # print("td_sph.time sample:", td_sph.time[0, 0, 0, :50])
    # print("td_patch.time shape:", td_patch.time.shape)
    # print("td_patch.time sample:", td_patch.time[0, 0, 0, :50])

    # (R,D,B,S) → (R,B,S) vs (R,P,B,S) → (R,B,S)
    sph_sum   = td_sph.time.sum(axis=1)
    patch_sum = td_patch.time.sum(axis=1)

    # print("td_sph.time shape:",sph_sum.shape)
    # print("td_sph.time sample:",  sph_sum[0, 0, :50])
    # print("td_patch.time shape:", patch_sum.shape)
    # print("td_patch.time sample:", patch_sum[0, 0, :50])

    npt.assert_allclose(sph_sum, patch_sum, rtol=0, atol=1e-12)


def test_direct_sound_added_correctly_when_enabled():
    walls = sp.testing.shoebox_room_stub(1, 1, 1)
    radiosity = sp.DirectionalRadiosityFast.from_polygon(walls, 1)

    radiosity.bake_geometry()
    radiosity.init_source_energy(pf.Coordinates(.25, .5, .5))
    radiosity.calculate_energy_exchange(343, 1/1000, 1, 3)

    receivers = pf.Coordinates(.75, .5, .5)
    detector  = _six_axis_pf_coords()

    td_sph_off = radiosity.collect_energy_at_spherical_detector(
        receivers, detector, direct_sound=False)
    td_sph_on  = radiosity.collect_energy_at_spherical_detector(
        receivers, detector, direct_sound=True)

    sum_off = td_sph_off.time.sum(axis=1)
    sum_on  = td_sph_on.time.sum(axis=1)

    if hasattr(radiosity, "calculate_direct_sound"):
        ds_all, n_delay_all = radiosity.calculate_direct_sound(receivers)
        R, B, S = sum_off.shape
        expected = sum_off.copy()
        tt = np.clip(n_delay_all, 0, S - 1).astype(int)
        for r in range(R):
            expected[r, :, tt[r]] += ds_all[r]
        npt.assert_allclose(sum_on, expected, rtol=0, atol=1e-12)
    else:
        assert np.all(sum_on >= sum_off)
        assert np.any(sum_on > sum_off)


def test_spherical_detector_raises_if_receiver_on_patch_center():
    """If a receiver coincides with a patch center, a ValueError is raised."""
    # Build a small scene
    walls = sp.testing.shoebox_room_stub(1, 1, 1)
    rad = sp.DirectionalRadiosityFast.from_polygon(walls, 1)

    rad.bake_geometry()
    rad.init_source_energy(pf.Coordinates(.5, .5, .5))
    rad.calculate_energy_exchange(343, 1/1000, 1, 3)

    # Place receiver exactly at the FIRST patch center to force zero distance
    pc = rad.patches_center[0]          # (3,)
    receivers = pf.Coordinates(pc[0], pc[1], pc[2])

    # Simple 6-axis detector
    detector = _six_axis_pf_coords()

    # Expect a ValueError instead of a warning
    with pytest.raises(ValueError, match="patch->receiver"):
        rad.collect_energy_at_spherical_detector(
            receivers, detector, direct_sound=False)
