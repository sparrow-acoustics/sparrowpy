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


@pytest.mark.parametrize("apply_brdf", [True, False])
@pytest.mark.parametrize("apply_attenuation", [True, False])
def test_io(apply_brdf, apply_attenuation, tmpdir):
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
    if apply_brdf:
        radiosity.set_wall_brdf(
            np.arange(len(walls)),
            pf.FrequencyData(np.ones_like(frequencies), frequencies),
            pf.Coordinates(0, 0, 1, weights=1),
            pf.Coordinates(0, 0, 1, weights=1))
    if apply_attenuation:
        radiosity.set_air_attenuation(
            pf.FrequencyData(np.ones_like(frequencies), frequencies))
    if apply_attenuation or apply_brdf:
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


def test_io_within_simulation(tmpdir):
    walls = sp.testing.shoebox_room_stub(1, 1, 1)
    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        walls, 1)

    # set brdf and air attenuation
    brdf_sources = pf.Coordinates(0, 0, 1, weights=1)
    brdf_receivers = pf.Coordinates(0, 0, 1, weights=1)
    frequencies = np.array([1000])
    brdf = sp.brdf.create_from_scattering(
        brdf_sources,
        brdf_receivers,
        pf.FrequencyData(1, frequencies),
        pf.FrequencyData(.1, frequencies))

    # set directional scattering data
    radiosity.set_wall_brdf(
        np.arange(len(walls)), brdf, brdf_sources, brdf_receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(
            np.zeros_like(brdf.frequencies),
            brdf.frequencies))
    radiosity.write(tmpdir / "test.far")
    radiosity_read = sp.DirectionalRadiosityFast.from_read(
        tmpdir / "test.far")

    # check readwrite after baking
    radiosity_read.bake_geometry()
    radiosity.bake_geometry()

    radiosity.write(tmpdir / "test.far")
    radiosity_read = sp.DirectionalRadiosityFast.from_read(
        tmpdir / "test.far")

    # check readwrite after init_source_energy
    radiosity.init_source_energy(pf.Coordinates(.5, .5, .5))
    radiosity_read.init_source_energy(pf.Coordinates(.5, .5, .5))

    radiosity.write(tmpdir / "test.far")
    radiosity_read = sp.DirectionalRadiosityFast.from_read(
        tmpdir / "test.far")

    # check readwrite after init_source_energy
    radiosity.calculate_energy_exchange(
        343, 1/1000, 1, 3, recalculate=True)
    radiosity_read.calculate_energy_exchange(
        343, 1/1000, 1, 3, recalculate=True)

    radiosity.write(tmpdir / "test.far")
    radiosity_read = sp.DirectionalRadiosityFast.from_read(
        tmpdir / "test.far")

    radiosity.collect_energy_receiver_mono(pf.Coordinates(.1, .5, .5))
    radiosity_read.collect_energy_receiver_mono(pf.Coordinates(.1, .5, .5))
