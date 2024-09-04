
"""Test the radiosity module with directional Patches."""
import os

import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
import sparapy as sp
import sparapy.geometry as geo
import sparapy.radiosity as radiosity
from sparapy.sound_object import Receiver, SoundSource


create_reference_files = False


@pytest.mark.parametrize('max_order_k', [2, 3])
def test_radiosity_directional_reference(max_order_k):
    """Test if the results changes."""
    path_sofa = os.path.join(
        os.path.dirname(__file__), 'test_data', 'ihta.E_sec_2.sofa')
    X = 10
    Y = 20
    Z = 15
    patch_size = 5
    ir_length_s = 5
    sampling_rate = 50

    ## create geometry
    ground = geo.Polygon(
        [[0, 0, 0], [X, 0, 0], [X, Y, 0], [0, Y, 0]], [1, 0, 0], [0, 0, 1])
    A_wall = geo.Polygon(
        [[0, 0, 0], [X, 0, 0], [X, 0, Z], [0, 0, Z]], [1, 0, 0], [0, 1, 0])
    B_wall = geo.Polygon(
        [[0, Y, 0], [X, Y, 0], [X, Y, Z], [0, Y, Z]], [1, 0, 0], [0, -1, 0])
    source = SoundSource([10, 6, 1], [0, 1, 0], [0, 0, 1])

    ## new approach
    radi = radiosity.DirectionalRadiosity(
        [ground, A_wall, B_wall], patch_size, max_order_k, ir_length_s,
        path_sofa, speed_of_sound=343, sampling_rate=sampling_rate)

    for patches in radi.patch_list:
        patches.directivity_data.freq = np.ones_like(
            patches.directivity_data.freq)

    radi.run(source)

    receiver = Receiver([20, 2, 1], [0, 1, 0], [0, 0, 1])
    irs_new = radi.energy_at_receiver(receiver)
    signal = pf.Signal(irs_new, sampling_rate)
    signal.time /= np.max(np.abs(signal.time))

    test_path = os.path.join(
        os.path.dirname(__file__), 'test_data')

    result = pf.io.read(
        os.path.join(
            test_path,
            f'simulation_X{X}_k{max_order_k}_{patch_size}m.far'))

    for i_frequency in range(signal.time.shape[0]):
        npt.assert_almost_equal(
            result['signal'].time[0, ...], signal.time[i_frequency, ...],
            decimal=4)
    npt.assert_almost_equal(result['signal'].times, signal.times)


@pytest.mark.parametrize('max_order_k', [2, 3])
def test_radiosity_directional_reference_read_write(max_order_k, tmpdir):
    """Test if the results changes."""
    X = 10
    Y = 20
    Z = 15
    patch_size = 5
    ir_length_s = 5
    sampling_rate = 50
    path_sofa = os.path.join(
        os.path.dirname(__file__), 'test_data', 'ihta.E_sec_2.sofa')

    ## create geometry
    ground = geo.Polygon(
        [[0, 0, 0], [X, 0, 0], [X, Y, 0], [0, Y, 0]], [1, 0, 0], [0, 0, 1])
    A_wall = geo.Polygon(
        [[0, 0, 0], [X, 0, 0], [X, 0, Z], [0, 0, Z]], [1, 0, 0], [0, 1, 0])
    B_wall = geo.Polygon(
        [[0, Y, 0], [X, Y, 0], [X, Y, Z], [0, Y, Z]], [1, 0, 0], [0, -1, 0])
    source = SoundSource([10, 6, 1], [0, 1, 0], [0, 0, 1])

    ## new approach
    radi = radiosity.DirectionalRadiosity(
        [ground, A_wall, B_wall], patch_size, max_order_k, ir_length_s,
        path_sofa, speed_of_sound=343, sampling_rate=sampling_rate)

    for patches in radi.patch_list:
        patches.directivity_data.freq = np.ones_like(
            patches.directivity_data.freq)

    radi.run(source)
    radi.write(os.path.join(tmpdir, 'radiosity.far'))
    radi = radiosity.DirectionalRadiosity.from_read(
        os.path.join(tmpdir, 'radiosity.far'))

    receiver = Receiver([20, 2, 1], [0, 1, 0], [0, 0, 1])
    irs_new = radi.energy_at_receiver(receiver)
    signal = pf.Signal(irs_new, sampling_rate)
    signal.time /= np.max(np.abs(signal.time))

    test_path = os.path.join(
        os.path.dirname(__file__), 'test_data')

    result = pf.io.read(
        os.path.join(
            test_path,
            f'simulation_X{X}_k{max_order_k}_{patch_size}m.far'))

    for i_frequency in range(signal.time.shape[0]):
        npt.assert_almost_equal(
            result['signal'].time[0, ...], signal.time[i_frequency, ...],
            decimal=4)
    npt.assert_almost_equal(result['signal'].times, signal.times)


@pytest.mark.parametrize('i_wall', [0, 1, 2, 3, 4, 5])
def test_init_energy_exchange(sample_walls, i_wall):
    """Test vs references for energy_exchange."""
    path_sofa = os.path.join(
        os.path.dirname(__file__), 'test_data', 'ihta.E_sec_2.sofa')
    patches = radiosity.PatchesDirectional.from_sofa(
        sample_walls[i_wall], 0.2, [], 0, path_sofa)
    max_order_k = 3
    ir_length_s = 5
    source = SoundSource([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1])
    patches.init_energy_exchange(max_order_k, ir_length_s, source, 1000)
    normal = np.array(patches.normal)
    # check if the sources are in the right direction
    eps = 1e-10
    for i in range(3):
        if normal[i] > eps:
            assert all(patches.directivity_sources.cartesian[:, i] > -eps)
        if normal[i] < -eps:
            assert all(patches.directivity_sources.cartesian[:, i] < eps)
    # check if the receivers are in the right direction
    for i in range(3):
        if normal[i] > eps:
            assert all(patches.directivity_receivers.cartesian[:, i] > -eps)
        if normal[i] < -eps:
            assert all(patches.directivity_receivers.cartesian[:, i] < eps)
    # check if receiver dimension is different
    # idx = np.argmax(patches.E_matrix[0, 0, :, :])
    # assert patches.E_matrix[0, 0, :, :] != patches.E_matrix[
    #     0, 0, idx, :]


@pytest.mark.parametrize('patch_size', [
    0.5,
    1,
    ])
@pytest.mark.parametrize('i_wall', [0, 1, 2, 3, 4, 5])
def test_init_energy_exchange_directional_omni(
        sample_walls, patch_size, i_wall):
    """Test vs refernces for energy_exchange."""
    reference_path = os.path.join(
        os.path.dirname(__file__), 'test_data',
        f'reference_matrix_directional_patch_size{patch_size}.far')
    path_sofa = os.path.join(
        os.path.dirname(__file__), 'test_data', 'ihta.E_sec_2.sofa')
    patches = radiosity.PatchesDirectional.from_sofa(
        sample_walls[i_wall], patch_size, [], 0, path_sofa)
    patches.directivity_data.freq = np.ones_like(
        patches.directivity_data.freq)
    max_order_k = 3
    ir_length_s = 5
    source = SoundSource([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1])
    patches.init_energy_exchange(max_order_k, ir_length_s, source, 1000)
    data = pf.io.read(reference_path)
    for i_frequency in range(patches.E_matrix.shape[0]):
        for i_rec in range(patches.E_matrix.shape[-1]):
            npt.assert_almost_equal(
                data['E_matrix'][0, ...],
                patches.E_matrix[i_frequency, ..., i_rec])


@pytest.mark.parametrize('perpendicular_walls', [
    [0, 2], [2, 0]
    ])
@pytest.mark.parametrize('patch_size', [
    0.5,
    1
    ])
def test_directional_energy_exchange(
        sample_walls, perpendicular_walls, patch_size):
    """Test vs refernces for energy_exchange."""
    max_order_k = 3
    ir_length_s = 5
    wall_source = sample_walls[perpendicular_walls[0]]
    wall_receiver = sample_walls[perpendicular_walls[1]]
    path_reference = os.path.join(
        os.path.dirname(__file__), 'test_data',
        f'reference_energy_exchange_size{patch_size}.far')
    path_sofa = os.path.join(
        os.path.dirname(__file__), 'test_data', 'ihta.E_sec_2.sofa')
    patch_1 = radiosity.PatchesDirectional.from_sofa(
        wall_source, patch_size, [1], 0, path_sofa)
    patch_1.directivity_data.freq = np.ones_like(patch_1.directivity_data.freq)
    patch_2 = radiosity.PatchesDirectional.from_sofa(
        wall_receiver, patch_size, [0], 1, path_sofa)
    patch_2.directivity_data.freq = np.ones_like(patch_2.directivity_data.freq)
    source = SoundSource([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1])
    patches = [patch_1, patch_2]
    patch_1.calculate_form_factor(patches)
    patch_1.init_energy_exchange(
        max_order_k, ir_length_s, source)
    patch_2.calculate_form_factor(patches)
    patch_2.init_energy_exchange(
        max_order_k, ir_length_s, source)
    for k in range(1, max_order_k + 1):
        patch_1.calculate_energy_exchange(patches, k)
        patch_2.calculate_energy_exchange(patches, k)

    data = pf.io.read(path_reference)

    for i_freq in range(patch_1.E_matrix.shape[0]):
        for i_rec in range(patch_1.E_matrix.shape[-1]):
            npt.assert_almost_equal(
                data['E_matrix'][0, ...], patch_1.E_matrix[i_freq, ..., i_rec],
                decimal=4)


# @pytest.mark.parametrize('perpendicular_walls', [
#     [0, 2],
#     [0, 4], [2, 0], [2, 4], [4, 0], [4, 2],
#     [1, 2], [1, 3], [1, 4], [1, 5],
#     [2, 1], [2, 5], [0, 3], [0, 5],
#     [3, 0], [3, 1], [3, 4], [3, 5],
#     [4, 1], [4, 3],
#     [5, 0], [5, 1], [5, 2], [5, 3],
#     ])
# @pytest.mark.parametrize('patch_size', [
#     0.5,
#     1
#     ])
# def test_directional_specular_reflections(
#         sample_walls, perpendicular_walls, patch_size):
#     """Test vs references for specular_reflections."""
#     max_order_k = 3
#     ir_length_s = 5
#     wall_source = sample_walls[perpendicular_walls[0]]
#     wall_receiver = sample_walls[perpendicular_walls[1]]
#     path_reference = os.path.join(
#         os.path.dirname(__file__), 'test_data',
#         f'reference_specular_reflections_{patch_size}.far')
#     path_sofa = os.path.join(
#         os.path.dirname(__file__), 'test_data', 'specular_gaussian5.sofa')
#     patch_1 = radiosity.PatchesDirectional.from_sofa(
#         wall_source, patch_size, [1], 0, path_sofa)
#     patch_2 = radiosity.PatchesDirectional.from_sofa(
#         wall_receiver, patch_size, [0], 1, path_sofa)
#     source = SoundSource([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1])
#     patches = [patch_1, patch_2]
#     patch_1.calculate_form_factor(patches)
#     patch_1.init_energy_exchange(
#         max_order_k, ir_length_s, source)
#     patch_2.calculate_form_factor(patches)
#     patch_2.init_energy_exchange(
#         max_order_k, ir_length_s, source)
#     for k in range(1, max_order_k + 1):
#         patch_1.calculate_energy_exchange(patches, k)
#         patch_2.calculate_energy_exchange(patches, k)
#     receiver = Receiver([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1])

#     ir = 0
#     ir += patch_1.energy_at_receiver(
#         max_order_k, receiver, ir_length_s,
#         speed_of_sound=346.18, sampling_rate=1000)
#     ir += patch_2.energy_at_receiver(
#         max_order_k, receiver, ir_length_s,
#         speed_of_sound=346.18, sampling_rate=1000)

#     if create_reference_files and (
#             perpendicular_walls[0] == 0) and (perpendicular_walls[1] == 2):
#         pf.io.write(path_reference, ir=ir)

#     data = pf.io.read(path_reference)

#     npt.assert_almost_equal(data['ir'], ir, decimal=4)


def test_PatchDirectional_to_from_dict(sample_walls):
    """Test if the results are correct with from_dict."""
    perpendicular_walls = [0, 2]
    patch_size = 0.2
    wall_source = sample_walls[perpendicular_walls[0]]
    path_sofa = os.path.join(
        os.path.dirname(__file__), 'test_data', 'specular_gaussian5.sofa')
    patch_1 = radiosity.PatchesDirectional.from_sofa(
        wall_source, patch_size, [1], 0, path_sofa)
    patch_1.init_energy_exchange(
        1, 1, SoundSource([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1]))
    reconstructed_patch = radiosity.PatchesDirectional.from_dict(
        patch_1.to_dict())
    assert reconstructed_patch.directivity_data == patch_1.directivity_data
    assert all(
        reconstructed_patch.directivity_sources ==
            patch_1.directivity_sources)
    assert all(
        reconstructed_patch.directivity_receivers
            == patch_1.directivity_receivers)
    npt.assert_array_equal(reconstructed_patch.E_matrix, patch_1.E_matrix)
    npt.assert_array_equal(
        reconstructed_patch.E_n_samples, patch_1.E_n_samples)
    npt.assert_array_equal(
        reconstructed_patch.form_factors, patch_1.form_factors)
    npt.assert_array_equal(reconstructed_patch.pts, patch_1.pts)
    npt.assert_array_equal(reconstructed_patch.up_vector, patch_1.up_vector)
    npt.assert_array_equal(reconstructed_patch.normal, patch_1.normal)
    assert reconstructed_patch.max_size == patch_1.max_size
    assert reconstructed_patch.other_wall_ids == patch_1.other_wall_ids
    assert reconstructed_patch.wall_id == patch_1.wall_id
    npt.assert_array_equal(reconstructed_patch.scattering, patch_1.scattering)
    npt.assert_array_equal(reconstructed_patch.absorption, patch_1.absorption)
    npt.assert_array_equal(
        reconstructed_patch.sound_attenuation_factor,
        patch_1.sound_attenuation_factor)


def test_RadiosityDirectional_to_from_dict():
    """Test if the results are correct with to_dict and from_dict."""
    max_order_k = 3
    patch_size = 5
    X = 10
    path_sofa = os.path.join(
        os.path.dirname(__file__), 'test_data', 'ihta.E_sec_2.sofa')
    Y = 20
    Z = 15
    ir_length_s = 5
    sampling_rate = 50

    # create geometry
    ground = geo.Polygon(
        [[0, 0, 0], [X, 0, 0], [X, Y, 0], [0, Y, 0]], [1, 0, 0], [0, 0, 1])
    A_wall = geo.Polygon(
        [[0, 0, 0], [X, 0, 0], [X, 0, Z], [0, 0, Z]], [1, 0, 0], [0, 1, 0])
    B_wall = geo.Polygon(
        [[0, Y, 0], [X, Y, 0], [X, Y, Z], [0, Y, Z]], [1, 0, 0], [0, -1, 0])
    source = SoundSource([10, 6, 1], [0, 1, 0], [0, 0, 1])

    # new approach
    radi = radiosity.DirectionalRadiosity(
        [ground, A_wall, B_wall], patch_size, max_order_k, ir_length_s,
        path_sofa, speed_of_sound=343, sampling_rate=sampling_rate)

    for patches in radi.patch_list:
        patches.directivity_data.freq = np.ones_like(
            patches.directivity_data.freq)

    radi.run(source)
    radi_dict = radi.to_dict()
    radi_reconstructed = radiosity.Radiosity.from_dict(radi_dict)

    # test
    assert isinstance(radi_reconstructed, radiosity.Radiosity)
    assert radi_reconstructed.patch_size == radi.patch_size
    assert radi_reconstructed.max_order_k == radi.max_order_k
    assert radi_reconstructed.ir_length_s == radi.ir_length_s
    assert radi_reconstructed.speed_of_sound == radi.speed_of_sound
    assert radi_reconstructed.sampling_rate == radi.sampling_rate
    assert len(radi_reconstructed.patch_list) == len(radi.patch_list)
    for patch, patch_reconstructed in zip(
            radi.patch_list, radi_reconstructed.patch_list):
        np.testing.assert_array_equal(
            patch.pts, patch_reconstructed.pts)
        np.testing.assert_array_equal(
            patch.up_vector, patch_reconstructed.up_vector)
        np.testing.assert_array_equal(
            patch._normal, patch_reconstructed._normal)
        assert patch.max_size == patch_reconstructed.max_size
        assert patch.wall_id == patch_reconstructed.wall_id
        np.testing.assert_array_equal(
            patch.scattering, patch_reconstructed.scattering)
        np.testing.assert_array_equal(
            patch.absorption, patch_reconstructed.absorption)
        np.testing.assert_array_equal(
            patch.sound_attenuation_factor,
            patch_reconstructed.sound_attenuation_factor)
        np.testing.assert_array_equal(
            patch.E_matrix, patch_reconstructed.E_matrix)


def test_RadiosityDirectional_read_write(tmpdir):
    """Test if the results are correct with read and write."""
    max_order_k = 3
    patch_size = 5
    X = 10
    path_sofa = os.path.join(
        os.path.dirname(__file__), 'test_data', 'ihta.E_sec_2.sofa')
    Y = 20
    Z = 15
    ir_length_s = 5
    sampling_rate = 50

    # create geometry
    ground = geo.Polygon(
        [[0, 0, 0], [X, 0, 0], [X, Y, 0], [0, Y, 0]], [1, 0, 0], [0, 0, 1])
    A_wall = geo.Polygon(
        [[0, 0, 0], [X, 0, 0], [X, 0, Z], [0, 0, Z]], [1, 0, 0], [0, 1, 0])
    B_wall = geo.Polygon(
        [[0, Y, 0], [X, Y, 0], [X, Y, Z], [0, Y, Z]], [1, 0, 0], [0, -1, 0])
    source = SoundSource([10, 6, 1], [0, 1, 0], [0, 0, 1])

    # new approach
    radi = radiosity.DirectionalRadiosity(
        [ground, A_wall, B_wall], patch_size, max_order_k, ir_length_s,
        path_sofa, speed_of_sound=343, sampling_rate=sampling_rate)

    for patches in radi.patch_list:
        patches.directivity_data.freq = np.ones_like(
            patches.directivity_data.freq)

    radi.run(source)
    json_path = os.path.join(tmpdir, 'radi.json')
    radi.write(json_path)
    radi_reconstructed = radiosity.DirectionalRadiosity.from_read(json_path)

    # test
    assert isinstance(radi_reconstructed, radiosity.DirectionalRadiosity)
    assert radi_reconstructed.patch_size == radi.patch_size
    assert radi_reconstructed.max_order_k == radi.max_order_k
    assert radi_reconstructed.ir_length_s == radi.ir_length_s
    assert radi_reconstructed.speed_of_sound == radi.speed_of_sound
    assert radi_reconstructed.sampling_rate == radi.sampling_rate
    assert len(radi_reconstructed.patch_list) == len(radi.patch_list)
    for patch, patch_reconstructed in zip(
            radi.patch_list, radi_reconstructed.patch_list):
        np.testing.assert_array_equal(
            patch.pts, patch_reconstructed.pts)
        np.testing.assert_array_equal(
            patch.up_vector, patch_reconstructed.up_vector)
        np.testing.assert_array_equal(
            patch._normal, patch_reconstructed._normal)
        assert patch.max_size == patch_reconstructed.max_size
        assert patch.wall_id == patch_reconstructed.wall_id
        np.testing.assert_array_equal(
            patch.scattering, patch_reconstructed.scattering)
        np.testing.assert_array_equal(
            patch.absorption, patch_reconstructed.absorption)
        np.testing.assert_array_equal(
            patch.sound_attenuation_factor,
            patch_reconstructed.sound_attenuation_factor)
        np.testing.assert_array_equal(
            patch.E_matrix, patch_reconstructed.E_matrix)


@pytest.mark.parametrize('patch_size', [
    1,
    1/3,
    1/5,
    1/7,
    ])
@pytest.mark.parametrize('sh_order', [
    1,
    3,
    5,
    ])
def test_specular_reflection(patch_size, sh_order, tmp_path_factory):
    """Test if the results are correct with to_dict and from_dict."""
    max_order_k = 2
    X = 1
    Y = 1
    x_s = 10
    y_s = 0
    z_s = 10
    x_r = -10
    y_r = 0
    z_r = 10
    ir_length_s = 0.15
    sampling_rate = 50

    filename = tmp_path_factory.mktemp("data") / "brdf_tmp.sofa"
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]
    sp.brdf.create_from_scattering(
        filename, coords, coords, pf.FrequencyData(0, [100]))

    # create geometry
    ground = geo.Polygon(
        [[-X/2, -Y/2, 0], [X/2, -Y/2, 0], [X/2, Y/2, 0], [-X/2, Y/2, 0]],
        [1, 0, 0], [0, 0, 1])
    source = SoundSource([x_s, y_s, z_s], [0, 1, 0], [0, 0, 1])

    # new approach
    radi = radiosity.DirectionalRadiosity(
        [ground], patch_size, max_order_k, ir_length_s,
        filename, speed_of_sound=343, sampling_rate=sampling_rate)

    radi.run(source)
    ir = radi.energy_at_receiver(
        Receiver([x_r, y_r, z_r], [0, 1, 0], [0, 0, 1]))

    ir_desired = np.zeros_like(ir)
    # add direct sound
    direct_distance = np.linalg.norm([x_r - x_s, y_r - y_s, z_r - z_s])
    direct_samples = int(direct_distance/343*sampling_rate)
    direct_energy = 1/(4*np.pi*direct_distance**2)
    ir_desired[:, direct_samples] = direct_energy
    # add reflection sound
    reflection_distance = np.linalg.norm([x_r - x_s, y_r - y_s, -z_r - z_s])
    reflection_samples = int(reflection_distance/343*sampling_rate)
    reflection_energy = 1/(4*np.pi*reflection_distance**2)
    ir_desired[:, reflection_samples] = reflection_energy

    npt.assert_almost_equal(
        10*np.log10(np.sum(ir)-direct_energy),
        10*np.log10(reflection_energy),
        decimal=0)


@pytest.mark.parametrize('patch_size', [
    1, 1/3, 1/5, 1/7, 1/9,
    ])
@pytest.mark.parametrize('sh_order', [
    1,
    3,
    5,
    21,
    ])
def test_specular_double_reflection(patch_size, sh_order, tmp_path_factory):
    """Test if the results are correct with to_dict and from_dict."""
    max_order_k = 3
    x_s = .5
    y_s = 0
    z_s = .5
    x_r = 2.5
    y_r = 0
    z_r = .5
    ir_length_s = 0.15
    sampling_rate = 300

    filename = tmp_path_factory.mktemp("data") / "brdf_tmp.sofa"
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]
    sp.brdf.create_from_scattering(
        filename, coords, coords, pf.FrequencyData(0, [100]))

    # create geometry
    ground = geo.Polygon(
        [[0.5, 0.5, 1], [0.5, -0.5, 1], [1.5, -0.5, 1], [1.5, 0.5, 1]],
        [1, 0, 0], [0, 0, 1])
    ground2 = geo.Polygon(
        [[1.5, 0.5, 0], [1.5, -0.5, 0], [2.5, -0.5, 0], [2.5, 0.5, 0]],
        [1, 0, 0], [0, 0, 1])
    source = SoundSource([x_s, y_s, z_s], [0, 1, 0], [0, 0, 1])

    # new approach
    radi = radiosity.DirectionalRadiosity(
        [ground, ground2], patch_size, max_order_k, ir_length_s,
        filename, speed_of_sound=343, sampling_rate=sampling_rate)

    radi.run(source)
    ir = radi.energy_at_receiver(
        Receiver([x_r, y_r, z_r], [0, 1, 0], [0, 0, 1]))

    ir_desired = np.zeros_like(ir)
    # add direct sound
    direct_distance = np.linalg.norm([x_r - x_s, y_r - y_s, z_r - z_s])
    direct_samples = int(direct_distance/343*sampling_rate)
    direct_energy = 1/(4*np.pi*direct_distance**2)
    ir_desired[:, direct_samples] = direct_energy
    # add reflection sound
    reflection_distance = 2*np.sqrt(2)
    reflection_samples = int(reflection_distance/343*sampling_rate)
    reflection_energy = 1/(4*np.pi*reflection_distance**2)
    ir_desired[:, reflection_samples] = reflection_energy

    npt.assert_almost_equal(
        10*np.log10(np.sum(ir)-direct_energy),
        10*np.log10(reflection_energy),
        decimal=1)
