import pytest
import numpy as np
import numpy.testing as npt
import sparrowpy as sp
import pyfar as pf


def test_init_default_single_wall():
    walls = sp.testing.shoebox_room_stub(1, 1, 1)
    walls = [walls[0]]
    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        walls, 1)

    npt.assert_almost_equal(radiosity.patches_points.shape, (1, 4, 3))
    npt.assert_almost_equal(radiosity.patches_area.shape, (1,))
    npt.assert_almost_equal(radiosity.patches_center.shape, (1, 3))
    npt.assert_almost_equal(radiosity.patches_size.shape, (1, 3))
    npt.assert_almost_equal(radiosity.patches_normal.shape, (1, 3))


def test_io(tmpdir):
    walls = sp.testing.shoebox_room_stub(1, 1, 1)
    walls = walls[:2]
    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        walls, 1)
    radiosity.write(tmpdir / "test.far")
    radiosity_out = sp.DirectionalRadiosityFast.from_read(
        tmpdir / "test.far")
    assert radiosity_out == radiosity

    # set brdf and air attenuation
    frequencies = [1000]
    radiosity.set_wall_brdf(
        np.arange(len(walls)),
        pf.FrequencyData(np.ones_like(frequencies), frequencies),
        pf.Coordinates(0, 0, 1, weights=1),
        pf.Coordinates(0, 0, 1, weights=1))
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.ones_like(frequencies), frequencies))
    radiosity.write(tmpdir / "test.far")
    radiosity_out = sp.DirectionalRadiosityFast.from_read(
        tmpdir / "test.far")
    assert radiosity_out == radiosity

    # check readwrite after baking
    radiosity.bake_geometry()
    radiosity.write(tmpdir / "test.far")
    radiosity_out = sp.DirectionalRadiosityFast.from_read(
        tmpdir / "test.far")
    assert radiosity_out == radiosity

    # check readwrite after init_source_energy
    radiosity.init_source_energy(pf.Coordinates(.5, .5, .5))
    radiosity.write(tmpdir / "test.far")
    radiosity_out = sp.DirectionalRadiosityFast.from_read(
        tmpdir / "test.far")
    assert radiosity_out == radiosity

    # check readwrite after init_source_energy
    radiosity.calculate_energy_exchange(343, 1/1000, 1, 3)
    radiosity.write(tmpdir / "test.far")
    radiosity_out = sp.DirectionalRadiosityFast.from_read(
        tmpdir / "test.far")
    assert radiosity_out == radiosity

