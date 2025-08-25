import pytest
import numpy as np
import numpy.testing as npt
import sparrowpy as sp
import pyfar as pf
from sparrowpy.classes.RadiosityFast import (
    get_brdf_incidence_directions_from_surface,
    )


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


@pytest.mark.parametrize(("delta_degree", "desired"), [
    (30, [25]),
    (10, [253, 254, 288, 289, 290, 324, 325, 326, 360]),
    ])
def test_get_brdf_incidence_directions_from_surface(delta_degree, desired):

    brdf_pos = pf.Coordinates(0, 0, 0, weights=1)
    brdf_directions = pf.samplings.sph_equal_angle(delta_degree)
    patch_edges = pf.Coordinates.from_spherical_elevation(
        np.array([-15, 15, 15, -15])/180*np.pi,
        np.array([-15, -15, 15, 15])/180*np.pi,
        1,
    )

    # get brdf incidence directions
    indexes = get_brdf_incidence_directions_from_surface(
        brdf_pos.cartesian[0],
        brdf_directions.cartesian,
        patch_edges.cartesian,
        np.array([-1, 0, 0], dtype=float))

    npt.assert_almost_equal(indexes, desired)
    npt.assert_almost_equal(indexes.shape, len(desired))


def test_get_brdf_incidence_directions_from_surface_nearest():

    brdf_pos = pf.Coordinates(0, 0, 0, weights=1)
    brdf_directions = pf.samplings.sph_equal_angle(30)
    patch_edges = pf.Coordinates.from_spherical_elevation(
        np.array([10, 10, 5, 5])/180*np.pi,
        np.array([10, 5, 5, 10])/180*np.pi,
        1,
    )

    # get brdf incidence directions
    indexes = get_brdf_incidence_directions_from_surface(
        brdf_pos.cartesian[0],
        brdf_directions.cartesian,
        patch_edges.cartesian,
        np.array([-1, 0, 0], dtype=float))

    npt.assert_almost_equal(indexes, 25)
    npt.assert_almost_equal(indexes.shape, (1))


def test_bake_patch_2_brdf_outgoing_mask():
    """
    Check if _patch_2_brdf_outgoing_mask parameter is set correctly.

    Test for 2 opposite walls.
    Last two directions are hitting the opposite wall.
    """
    brdf_directions = pf.Coordinates(
        [-1, 1, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0.1],
        [0, 0, 0, 0, 1, 1.1],
        weights=[1, 1, 1, 1, 1, 1],
    )
    frequencies = np.array([1000])

    # define specular brdf
    brdf = sp.brdf.create_from_scattering(
        brdf_directions.copy(),
        brdf_directions.copy(),
        pf.FrequencyData(0, frequencies),
        pf.FrequencyData(0, frequencies))

    # create get parallel walls
    parallel_walls = sp.testing.shoebox_room_stub(1, 1, 1)[:2]

    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        parallel_walls, 1)
    radiosity.set_wall_brdf(
        np.arange(len(parallel_walls)),
        brdf, brdf_directions, brdf_directions)
    radiosity.bake_geometry()

    assert radiosity._patch_2_brdf_outgoing_mask.shape == (2, 2, 6)
    npt.assert_almost_equal(
        radiosity._patch_2_brdf_outgoing_mask[0, 1],
        radiosity._patch_2_brdf_outgoing_mask[1, 0])
    npt.assert_array_equal(
        radiosity._patch_2_brdf_outgoing_mask[0, 1, -2], True)
    npt.assert_array_equal(
        radiosity._patch_2_brdf_outgoing_mask[0, 1, :-2], False)


@pytest.mark.parametrize("delta_angle", [
    15, 30])
def test_specular_reflections(delta_angle):
    """
    Test if specular reflections are larger than 0.

    For two parallel walls, and more than one direction of brdf facting
    towards it.
    """
    brdf_directions = pf.samplings.sph_equal_angle(delta_angle)
    brdf_directions.weights = pf.samplings.calculate_sph_voronoi_weights(
        brdf_directions)
    brdf_directions = brdf_directions[brdf_directions.z>0]
    frequencies = np.array([1000])
    brdf = sp.brdf.create_from_scattering(
        brdf_directions.copy(),
        brdf_directions.copy(),
        pf.FrequencyData(0, frequencies),
        pf.FrequencyData(0, frequencies))

    # create get parallel walls
    parallel_walls = sp.testing.shoebox_room_stub(1, 1, 1)[:2]

    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        parallel_walls, 1)
    radiosity.set_wall_brdf(
        np.arange(len(parallel_walls)),
        brdf, brdf_directions, brdf_directions)
    radiosity.bake_geometry()

    radiosity.init_source_energy(pf.Coordinates(.5, .5, .5))
    radiosity.calculate_energy_exchange(343, 1/100, 0.01, 2)
    etc = radiosity.collect_energy_receiver_mono(
        pf.Coordinates(.5, .5, .5))

    r_is_1 = 1
    r_is_2 = 2
    reflected_energy_analytic = 2*(
        1/(4 * np.pi * r_is_1**2) + 1/(4 * np.pi * r_is_2**2))
    assert np.sum(etc.time)>0
    npt.assert_almost_equal(np.sum(etc.time), reflected_energy_analytic)
