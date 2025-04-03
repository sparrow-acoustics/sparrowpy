
"""Test the radiosity module with directional Patches."""
import os

import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
import sparrowpy.geometry as geo
import sparrowpy as sp
from sparrowpy.sound_object import Receiver, SoundSource


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
    radi = sp.DirectionalRadiosityKang(
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
    radi = sp.DirectionalRadiosityKang(
        [ground, A_wall, B_wall], patch_size, max_order_k, ir_length_s,
        path_sofa, speed_of_sound=343, sampling_rate=sampling_rate)

    for patches in radi.patch_list:
        patches.directivity_data.freq = np.ones_like(
            patches.directivity_data.freq)

    radi.run(source)
    radi.write(os.path.join(tmpdir, 'radiosity.far'))
    radi = sp.DirectionalRadiosityKang.from_read(
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
    patches = sp.PatchesDirectionalKang.from_sofa(
        sample_walls[i_wall], 0.2, [], 0, path_sofa)
    max_order_k = 3
    ir_length_s = 5
    source = SoundSource([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1])
    patches.init_energy_exchange(
        max_order_k, ir_length_s, source, 1000, 346.18)
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
    patches = sp.PatchesDirectionalKang.from_sofa(
        sample_walls[i_wall], patch_size, [], 0, path_sofa)
    patches.directivity_data.freq = np.ones_like(
        patches.directivity_data.freq)
    max_order_k = 3
    ir_length_s = 5
    source = SoundSource([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1])
    patches.init_energy_exchange(
        max_order_k, ir_length_s, source, 1000, 346.18)
    data = pf.io.read(reference_path)
    for i_frequency in range(patches.E_matrix.shape[0]):
        for i_rec in range(patches.E_matrix.shape[-1]):
            npt.assert_almost_equal(
                data['E_matrix'][0, ...],
                patches.E_matrix[i_frequency, ..., i_rec])


@pytest.mark.parametrize('perpendicular_walls', [
    [0, 2],
    [0, 4], [2, 0], [2, 4], [4, 0], [4, 2],
    [1, 2], [1, 3], [1, 4], [1, 5],
    [2, 1], [2, 5], [0, 3], [0, 5],
    [3, 0], [3, 1], [3, 4], [3, 5],
    [4, 1], [4, 3],
    [5, 0], [5, 1], [5, 2], [5, 3],
    ])
@pytest.mark.parametrize('patch_size', [
    0.5,
    1,
    ])
def test_directional_specular_reflections(
        sample_walls, perpendicular_walls, patch_size):
    """Test vs references for specular_reflections."""
    max_order_k=3
    ir_length_s=5
    sampling_rate = 1000
    speed_of_sound = 346.18
    wall_source = sample_walls[perpendicular_walls[0]]
    wall_receiver = sample_walls[perpendicular_walls[1]]
    path_reference = os.path.join(
        os.path.dirname(__file__), 'test_data',
        f'reference_specular_reflections_{patch_size}.far')
    path_sofa = os.path.join(
        os.path.dirname(__file__), 'test_data', 'specular_gaussian5.sofa')
    patch_1 = sp.PatchesDirectionalKang.from_sofa(
        wall_source, patch_size, [1], 0, path_sofa)
    patch_2 = sp.PatchesDirectionalKang.from_sofa(
        wall_receiver, patch_size, [0], 1, path_sofa)
    source = SoundSource([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1])
    patches = [patch_1, patch_2]
    patch_1.calculate_form_factor(patches)
    patch_1.init_energy_exchange(
        max_order_k, ir_length_s, source, sampling_rate, speed_of_sound)
    patch_2.calculate_form_factor(patches)
    patch_2.init_energy_exchange(
        max_order_k, ir_length_s, source, sampling_rate, speed_of_sound)
    for k in range(1, max_order_k+1):
        patch_1.calculate_energy_exchange(
            patches, k, speed_of_sound, sampling_rate)
        patch_2.calculate_energy_exchange(
            patches, k, speed_of_sound, sampling_rate)
    receiver = Receiver([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1])

    ir = 0
    ir += patch_1.energy_at_receiver(
        max_order_k, receiver,
        speed_of_sound=346.18, sampling_rate=1000)
    ir += patch_2.energy_at_receiver(
        max_order_k, receiver,
        speed_of_sound=346.18, sampling_rate=1000)

    if create_reference_files and (
            perpendicular_walls[0] == 0) and (perpendicular_walls[1] == 2):
        pf.io.write(path_reference, ir=ir)

    data = pf.io.read(path_reference)

    npt.assert_almost_equal(data['ir'], ir, decimal=4)


def test_PatchDirectional_to_from_dict(sample_walls):
    """Test if the results are correct with from_dict."""
    perpendicular_walls = [0, 2]
    patch_size = 0.2
    sampling_rate = 1000
    speed_of_sound = 346.18
    wall_source = sample_walls[perpendicular_walls[0]]
    path_sofa = os.path.join(
        os.path.dirname(__file__), 'test_data', 'specular_gaussian5.sofa')
    patch_1 = sp.PatchesDirectionalKang.from_sofa(
        wall_source, patch_size, [1], 0, path_sofa)
    patch_1.init_energy_exchange(
        1, 1, SoundSource([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1]),
        sampling_rate, speed_of_sound)
    reconstructed_patch = sp.PatchesDirectionalKang.from_dict(
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
    radi = sp.DirectionalRadiosityKang(
        [ground, A_wall, B_wall], patch_size, max_order_k, ir_length_s,
        path_sofa, speed_of_sound=343, sampling_rate=sampling_rate)

    for patches in radi.patch_list:
        patches.directivity_data.freq = np.ones_like(
            patches.directivity_data.freq)

    radi.run(source)
    radi_dict = radi.to_dict()
    radi_reconstructed = sp.RadiosityKang.from_dict(radi_dict)

    # test
    assert isinstance(radi_reconstructed, sp.RadiosityKang)
    assert radi_reconstructed.patch_size == radi.patch_size
    assert radi_reconstructed.max_order_k == radi.max_order_k
    assert radi_reconstructed.ir_length_s == radi.ir_length_s
    assert radi_reconstructed.speed_of_sound == radi.speed_of_sound
    assert radi_reconstructed.sampling_rate == radi.sampling_rate
    assert len(radi_reconstructed.patch_list) == len(radi.patch_list)
    for patch, patch_reconstructed in zip(
            radi.patch_list, radi_reconstructed.patch_list, strict=True):
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
    radi = sp.DirectionalRadiosityKang(
        [ground, A_wall, B_wall], patch_size, max_order_k, ir_length_s,
        path_sofa, speed_of_sound=343, sampling_rate=sampling_rate)

    for patches in radi.patch_list:
        patches.directivity_data.freq = np.ones_like(
            patches.directivity_data.freq)

    radi.run(source)
    json_path = os.path.join(tmpdir, 'radi.json')
    radi.write(json_path)
    radi_reconstructed = sp.DirectionalRadiosityKang.from_read(
        json_path)

    # test
    assert isinstance(radi_reconstructed, sp.DirectionalRadiosityKang)
    assert radi_reconstructed.patch_size == radi.patch_size
    assert radi_reconstructed.max_order_k == radi.max_order_k
    assert radi_reconstructed.ir_length_s == radi.ir_length_s
    assert radi_reconstructed.speed_of_sound == radi.speed_of_sound
    assert radi_reconstructed.sampling_rate == radi.sampling_rate
    assert len(radi_reconstructed.patch_list) == len(radi.patch_list)
    for patch, patch_reconstructed in zip(
            radi.patch_list, radi_reconstructed.patch_list, strict=True):
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

