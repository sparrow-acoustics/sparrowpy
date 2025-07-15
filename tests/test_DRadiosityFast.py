"""Test radiosity module."""
import numpy as np
import numpy.testing as npt
import pytest
import pyfar as pf
import os
import sparrowpy as sp
import sofar as sf


def test_init(sample_walls):
    radiosity = sp.DirectionalRadiosityFast.from_polygon(sample_walls, 0.2)
    npt.assert_almost_equal(radiosity.patches_points.shape, (150, 4, 3))
    npt.assert_almost_equal(radiosity.patches_area.shape, (150))
    npt.assert_almost_equal(radiosity.patches_center.shape, (150, 3))
    npt.assert_almost_equal(radiosity.patches_size.shape, (150, 3))
    npt.assert_almost_equal(radiosity.patches_normal.shape, (150, 3))


def test_check_visibility(sample_walls):
    radiosity = sp.DirectionalRadiosityFast.from_polygon(sample_walls, 0.2)
    radiosity.bake_geometry()
    npt.assert_almost_equal(radiosity._visibility_matrix.shape, (150, 150))
    # npt.assert_array_equal(
    #     radiosity._visibility_matrix, radiosity._visibility_matrix.T)
    npt.assert_array_equal(radiosity._visibility_matrix[:25, :25], False)
    npt.assert_array_equal(radiosity._visibility_matrix[:25, 25:], True)
    npt.assert_array_equal(radiosity._visibility_matrix[25:50, 25:50], False)
    assert np.sum(radiosity._visibility_matrix) == 25*5*25*6/2


def test_check_visibility_wrapper(sample_walls):
    radiosity = sp.DirectionalRadiosityFast.from_polygon(sample_walls, 0.2)
    radiosity.bake_geometry()
    visibility_matrix = sp.geometry._check_patch2patch_visibility(
        radiosity.patches_center, radiosity.patches_normal,
        radiosity.patches_points)
    npt.assert_almost_equal(radiosity._visibility_matrix, visibility_matrix)


def test_compute_form_factors(sample_walls):
    radiosity = sp.DirectionalRadiosityFast.from_polygon(sample_walls, 0.2)
    radiosity.bake_geometry()
    npt.assert_almost_equal(radiosity.form_factors.shape, (150, 150))
    radiosity.bake_geometry()
    npt.assert_almost_equal(radiosity.form_factors.shape, (150, 150))



@pytest.mark.parametrize('walls', [
    # perpendicular walls
    [0, 2], [0, 3], [0, 4], [0, 5],
    [1, 2], [1, 3], [1, 4], [1, 5],
    [2, 0], [2, 1], [2, 4], [2, 5],
    [3, 0], [3, 1], [3, 4], [3, 5],
    [4, 0], [4, 1], [4, 2], [4, 3],
    [5, 0], [5, 1], [5, 2], [5, 3],
    # parallel walls
    [0, 1], [2, 3], [4, 5],
    [1, 0], [3, 2], [5, 4],
    ])
@pytest.mark.parametrize('patch_size', [
    0.5,
    ])
def test_form_factors_directivity_for_diffuse(
        sample_walls, walls, patch_size, sofa_data_diffuse):
    """Test form factor calculation for perpendicular walls."""
    wall_source = sample_walls[walls[0]]
    wall_receiver = sample_walls[walls[1]]
    walls = [wall_source, wall_receiver]

    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        walls, patch_size)
    data, sources, receivers = sofa_data_diffuse
    radiosity.set_wall_brdf(
        np.arange(len(walls)), data, sources, receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.bake_geometry()

    form_factors_from_tilde = np.max(radiosity._form_factors_tilde, axis=2)
    for i in range(4):
        npt.assert_almost_equal(
            radiosity.form_factors[:4, 4:], form_factors_from_tilde[:4, 4:, i])
        npt.assert_almost_equal(
            radiosity.form_factors[:4, 4:].T,
            form_factors_from_tilde[4:, :4, i])

    for i in range(8):
        npt.assert_almost_equal(radiosity._form_factors_tilde[i, i, :, :], 0)


def test_set_wall_scattering(sample_walls, sofa_data_diffuse):
    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        sample_walls, 0.2)
    (data, sources, receivers) = sofa_data_diffuse
    radiosity.set_wall_brdf(np.arange(6), data, sources, receivers)
    # check shape of scattering matrix
    assert len(radiosity._brdf) == 1
    npt.assert_almost_equal(radiosity._brdf[0].shape, (4, 4, 4))
    npt.assert_array_equal(radiosity._brdf[0], 1)
    npt.assert_array_equal(radiosity._brdf_index, 0)
    # check source and receiver direction
    for i in range(6):
        assert (np.sum(
            radiosity._brdf_incoming_directions[i].cartesian * \
                radiosity.walls_normal[i,:],
            axis=-1)>0).all()
        assert (np.sum(
            radiosity._brdf_outgoing_directions[i].cartesian * \
                radiosity.walls_normal[i,:],
            axis=-1)>0).all()


def test_set_wall_scattering_different(sample_walls, sofa_data_diffuse):
    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        sample_walls, 0.2)
    (data, sources, receivers) = sofa_data_diffuse
    radiosity.set_wall_brdf([0, 1, 2], data, sources, receivers)
    radiosity.set_wall_brdf([3, 4, 5], data, sources, receivers)
    # check shape of scattering matrix
    assert len(radiosity._brdf) == 2
    for i in range(2):
        npt.assert_almost_equal(radiosity._brdf[i].shape, (4, 4, 4))
        npt.assert_array_equal(radiosity._brdf[i], 1)
    npt.assert_array_equal(radiosity._brdf_index[:3], 0)
    npt.assert_array_equal(radiosity._brdf_index[3:], 1)
    # check source and receiver direction
    for i in range(6):
        assert (np.sum(
            radiosity._brdf_incoming_directions[i].cartesian * \
                radiosity.walls_normal[i,:],
            axis=-1)>0).all()
        assert (np.sum(
            radiosity._brdf_outgoing_directions[i].cartesian * \
                radiosity.walls_normal[i,:],
            axis=-1)>0).all()


def test_set_air_attenuation(sample_walls):
    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        sample_walls, 0.2)
    radiosity.set_air_attenuation(pf.FrequencyData([0.1, 0.2], [500, 1000]))
    npt.assert_array_equal(radiosity._air_attenuation, [0.1, 0.2])


def test_total_number_of_patches():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    result = sp.geometry._total_number_of_patches(points, 0.2)
    desired = 25
    assert result == desired


def test_init_source_with_directivity(sample_walls):
    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        sample_walls, 1)

    position = np.array([0, 0, 0])
    view = np.array([1, 0, 0])
    up = np.array([0, 0, 1])
    path = os.path.join(
        os.path.dirname(__file__), 'test_data',
        'Genelec8020_DAF_2016_1x1.v17.ms.sofa')
    directivity = sp.sound_object.DirectivityMS(path)
    sound_source = sp.sound_object.SoundSource(position, view, up, directivity)

    # try without directivity
    radiosity.init_source_energy(pf.Coordinates(*position))
    init_energy = radiosity._energy_init_source
    distance_patches_to_source = radiosity._distance_patches_to_source
    assert isinstance(radiosity._source, pf.Coordinates)

    # try with directivity
    radiosity.init_source_energy(sound_source)
    init_energy_dir = radiosity._energy_init_source
    distance_patches_to_source_dir = radiosity._distance_patches_to_source

    assert not np.allclose(init_energy, init_energy_dir)
    npt.assert_allclose(
        distance_patches_to_source, distance_patches_to_source_dir)
    assert isinstance(radiosity._source, sp.sound_object.SoundSource)


def test_init_source_with_directivity_frequency(sample_walls):
    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        sample_walls, 1)

    position = np.array([0, 0, 0])
    view = np.array([1, 0, 0])
    up = np.array([0, 0, 1])
    path = os.path.join(
        os.path.dirname(__file__), 'test_data',
        'Genelec8020_DAF_2016_1x1.v17.ms.sofa')

    # create directivity
    directivity = sp.sound_object.DirectivityMS(path)
    sound_source = sp.sound_object.SoundSource(position, view, up, directivity)

    # set air attenuation
    frequencies = sf.read_sofa(path, verbose=False).N
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(frequencies), frequencies))

    # try without directivity
    radiosity.init_source_energy(pf.Coordinates(*position))
    init_energy = radiosity._energy_init_source
    distance_patches_to_source = radiosity._distance_patches_to_source

    # try with directivity
    radiosity.init_source_energy(sound_source)
    init_energy_dir = radiosity._energy_init_source
    distance_patches_to_source_dir = radiosity._distance_patches_to_source

    assert not np.allclose(init_energy, init_energy_dir)
    npt.assert_allclose(
        distance_patches_to_source, distance_patches_to_source_dir)

    # test if first frequency bin is same as above
    npt.assert_allclose(
        np.array([
            [0.        ], [2.13215939], [0.        ],
            [2.16329116], [0.        ], [2.12899306]]),
            init_energy_dir[..., 0])

    # test if other frequency bins are not same as first one
    for i in range(1, frequencies.shape[0]):
        assert not np.allclose(
            np.array([
                [[0.        ]], [[2.13215939]], [[0.        ]],
                [[2.16329116]], [[0.        ]], [[2.12899306]]]),
                init_energy_dir[..., i])


def test_collect_receiver_direct_sound(sample_walls):
    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        sample_walls, 1)

    position = np.array([0, 0, 0])
    view = np.array([1, 0, 0])
    up = np.array([0, 0, 1])
    path = os.path.join(
        os.path.dirname(__file__), 'test_data',
        'Genelec8020_DAF_2016_1x1.v17.ms.sofa')

    # create directivity
    directivity = sp.sound_object.DirectivityMS(path)
    sound_source = sp.sound_object.SoundSource(position, view, up, directivity)

    # set air attenuation
    frequencies = sf.read_sofa(path, verbose=False).N
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(frequencies), frequencies))

    # try with directivity
    radiosity.init_source_energy(sound_source)
    radiosity.calculate_energy_exchange(343, 0.1, 1)

    receivers = pf.Coordinates([0.5, 0.5], 0, 0)
    direct_sound, delay_samples = radiosity.calculate_direct_sound(
        receivers)
    n_receivers = receivers.cshape[0]
    n_bins = radiosity.n_bins
    npt.assert_almost_equal(direct_sound.shape, (n_receivers, n_bins))
    npt.assert_almost_equal(delay_samples.shape, (n_receivers))

    npt.assert_almost_equal(direct_sound[0], direct_sound[1])
    npt.assert_almost_equal(delay_samples[0], delay_samples[1])


def test_collect_receiver_mono_direct_sound(sample_walls):
    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        sample_walls, 1)

    position = np.array([0, 0, 0])
    view = np.array([1, 0, 0])
    up = np.array([0, 0, 1])
    path = os.path.join(
        os.path.dirname(__file__), 'test_data',
        'Genelec8020_DAF_2016_1x1.v17.ms.sofa')

    # create directivity
    directivity = sp.sound_object.DirectivityMS(path)
    sound_source = sp.sound_object.SoundSource(position, view, up, directivity)

    # set air attenuation
    frequencies = sf.read_sofa(path, verbose=False).N
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(frequencies), frequencies))

    # try with directivity
    radiosity.init_source_energy(sound_source)
    radiosity.calculate_energy_exchange(343, 0.001, .1)

    receivers = pf.Coordinates([0.5, 0.5], 0, 0)
    direct_sound, delay_samples = radiosity.calculate_direct_sound(
        receivers)

    etc = radiosity.collect_energy_receiver_mono(
        receivers, True)
    npt.assert_almost_equal(
        etc.time[
            np.arange(len(delay_samples)), : , delay_samples], direct_sound)


def test_collect_receiver_mono_direct_sound_with_brdf(
        sample_walls, sofa_data_diffuse_full_third_octave):
    brdf, sources, receivers = sofa_data_diffuse_full_third_octave
    radiosity = sp.DirectionalRadiosityFast.from_polygon(
        sample_walls, 1)

    position = np.array([0, 0, 0])
    view = np.array([1, 0, 0])
    up = np.array([0, 0, 1])
    path = os.path.join(
        os.path.dirname(__file__), 'test_data',
        'Genelec8020_DAF_2016_1x1.v17.ms.sofa')

    # create directivity
    directivity = sp.sound_object.DirectivityMS(path)
    sound_source = sp.sound_object.SoundSource(position, view, up, directivity)

    # set air attenuation
    radiosity.set_wall_brdf(
        np.arange(6), brdf, sources, receivers)

    # try with directivity
    radiosity.init_source_energy(sound_source)
    radiosity.calculate_energy_exchange(343, 0.001, .1)

    receivers = pf.Coordinates([0.5, 0.5], 0, 0)
    direct_sound, delay_samples = radiosity.calculate_direct_sound(
        receivers)

    etc = radiosity.collect_energy_receiver_mono(
        receivers, True)
    npt.assert_almost_equal(
        etc.time[
            np.arange(len(delay_samples)), : , delay_samples], direct_sound)
