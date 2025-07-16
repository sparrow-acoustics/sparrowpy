"""Test the radiosity.Radiosity module."""
import os

import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
import sparrowpy as sp
import sparrowpy.geometry as geo
from sparrowpy.sound_object import Receiver, SoundSource


create_reference_files = False


def test_small_room_and_shift():
    """Test if the results changes for shifted walls."""
    X = 5
    Y = 6
    Z = 4
    r_x = 3
    r_y = 2
    r_z = 3
    patch_size = 1
    ir_length_s = 1
    sampling_rate = 1
    max_order_k = 10
    speed_of_sound = 343
    irs_new = []
    E_matrix = []
    form_factors = []
    for i in range(2):
        walls = sp.testing.shoebox_room_stub(X, Y, Z)
        if i == 0:
            delta_x = 0.
            delta_y = 0.
            delta_z = 0.
        elif i == 1:
            delta_x = 2.
            delta_y = 4.
            delta_z = 5.

        receiver_pos = [r_x+delta_x, r_y+delta_y, r_z+delta_z]
        for wall in walls:
            wall.pts += np.array([delta_x, delta_y, delta_z])

        # create geometry
        source = sp.geometry.SoundSource(
            [2+delta_x, 2+delta_y, 2+delta_z], [0, 1, 0], [0, 0, 1])

        ## new approach
        radi = sp.RadiosityKang(
            walls, patch_size, max_order_k, ir_length_s,
            speed_of_sound=speed_of_sound, sampling_rate=sampling_rate,
            absorption=0.1)

        # run simulation
        radi.run(source)

        E_matrix.append(np.concatenate([
            radi.patch_list[i].E_matrix for i in range(6)], axis=-2))
        form_factors.append([
            radi.patch_list[i].form_factors for i in range(6)])

        # test energy at receiver
        receiver = sp.geometry.Receiver(receiver_pos, [0, 1, 0], [0, 0, 1])
        irs_new.append(radi.energy_at_receiver(receiver, ignore_direct=True))

    # compare E_matrix
    E_matrix = np.array(E_matrix)
    npt.assert_array_equal(E_matrix[0], E_matrix[1])

    # compare form factors
    for i in range(6):
        npt.assert_array_equal(form_factors[0][i], form_factors[1][i])

    # rotate all walls
    irs_new = np.array(irs_new).squeeze()
    npt.assert_array_equal(irs_new, irs_new[0])


def test_small_room_and_rotate():
    """Test if the results changes for rotated walls."""
    X = 5
    Y = 3
    Z = 4
    r_x = 1
    r_y = 2
    r_z = 1.5
    patch_size = 1
    ir_length_s = 1
    sampling_rate = 1
    max_order_k = 10
    speed_of_sound = 343
    irs_new = []
    E_matrix = []
    E_matrix_sum = []
    for i in range(6):
        if i == 0:
            walls = sp.testing.shoebox_room_stub(X, Y, Z)
            receiver_pos = [r_x, r_y, r_z]
        elif i == 1:
            walls = sp.testing.shoebox_room_stub(X, Z, Y)
            receiver_pos = [r_x, r_z, r_y]
        elif i == 2:
            walls = sp.testing.shoebox_room_stub(Y, Z, X)
            receiver_pos = [r_y, r_z, r_x]
        elif i == 3:
            walls = sp.testing.shoebox_room_stub(Y, X, Z)
            receiver_pos = [r_y, r_x, r_z]
        elif i == 4:
            walls = sp.testing.shoebox_room_stub(Z, X, Y)
            receiver_pos = [r_z, r_x, r_y]
        else:
            walls = sp.testing.shoebox_room_stub(Z, Y, X)
            receiver_pos = [r_z, r_y, r_x]

        # create geometry
        source = sp.geometry.SoundSource([2, 2, 2], [0, 1, 0], [0, 0, 1])

        ## new approach
        radi = sp.RadiosityKang(
            walls, patch_size, max_order_k, ir_length_s,
            speed_of_sound=speed_of_sound, sampling_rate=sampling_rate,
            absorption=0.1)

        # run simulation
        radi.run(source)

        E_matrix.append(np.concatenate([
            radi.patch_list[i].E_matrix for i in range(6)], axis=-2))
        E_matrix_sum.append(
            [radi.patch_list[i].E_matrix.sum() for i in range(6)])
        # test energy at receiver
        receiver = sp.geometry.Receiver(receiver_pos, [0, 1, 0], [0, 0, 1])
        irs_new.append(radi.energy_at_receiver(receiver, ignore_direct=True))

    # rotate all walls
    irs_new = np.array(irs_new).squeeze()
    npt.assert_array_almost_equal(irs_new, irs_new[0], decimal=4)


def test_small_room_and_rotate_init_energy():
    """Test if the results changes for rotated walls."""
    X = 5
    Y = 6
    Z = 4
    patch_size = 1
    ir_length_s = 1
    sampling_rate = 1
    max_order_k = 0
    speed_of_sound = 343
    E_matrix = []
    E_matrix_sum = []
    for i in range(6):
        if i == 0:
            walls = sp.testing.shoebox_room_stub(X, Y, Z)
        elif i == 1:
            walls = sp.testing.shoebox_room_stub(X, Z, Y)
        elif i == 2:
            walls = sp.testing.shoebox_room_stub(Y, Z, X)
        elif i == 3:
            walls = sp.testing.shoebox_room_stub(Y, X, Z)
        elif i == 4:
            walls = sp.testing.shoebox_room_stub(Z, X, Y)
        else:
            walls = sp.testing.shoebox_room_stub(Z, Y, X)

        # create geometry
        source = sp.geometry.SoundSource([2, 2, 2], [0, 1, 0], [0, 0, 1])

        ## new approach
        radi = sp.RadiosityKang(
            walls, patch_size, max_order_k, ir_length_s,
            speed_of_sound=speed_of_sound, sampling_rate=sampling_rate,
            absorption=0.1)

        # run init energy
        # B. First-order patch
        for patches in radi.patch_list:
            patches.init_energy_exchange(
                radi.max_order_k, radi.ir_length_s, source,
                sampling_rate=radi.sampling_rate,
                speed_of_sound=speed_of_sound)

        E_matrix.append(np.concatenate([
            radi.patch_list[i].E_matrix for i in range(6)], axis=-2))
        E_matrix_sum.append(
            [radi.patch_list[i].E_matrix.sum() for i in range(6)])

    # rotate all walls
    E_matrix_sum = np.array(E_matrix_sum)
    matrix = np.zeros(E_matrix_sum.shape, dtype=int)
    reference_energy = E_matrix_sum[:, 2] # formula was given for the ground
    for i, ref in enumerate(reference_energy):
        matrix[np.abs(E_matrix_sum-ref)<1e-10] = i+1
    E_matrix = np.array(E_matrix)
    E_matrix = np.sum(E_matrix[:, 0, 0, :, 0], axis=-1)
    npt.assert_array_almost_equal(E_matrix, E_matrix[0], decimal=4)


@pytest.mark.parametrize('patch_size', [1, 0.5])
def test_cube_and_rotate_init_energy(patch_size):
    """Test if the results changes for rotated walls."""
    ir_length_s = 1
    sampling_rate = 1
    max_order_k = 0
    speed_of_sound = 343
    walls = sp.testing.shoebox_room_stub(1, 1, 1)
    # create geometry
    source = sp.geometry.SoundSource([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1])

    ## new approach
    radi = sp.RadiosityKang(
        walls, patch_size, max_order_k, ir_length_s,
        speed_of_sound=speed_of_sound, sampling_rate=sampling_rate,
        absorption=0.1)

    # run init energy
    # B. First-order patch sources
    for patches in radi.patch_list:
        print(patches.normal)
        patches.init_energy_exchange(
            radi.max_order_k, radi.ir_length_s, source,
            sampling_rate=radi.sampling_rate, speed_of_sound=speed_of_sound)

    E_matrix= np.concatenate([
        radi.patch_list[i].E_matrix for i in range(6)], axis=-2)

    # rotate all walls
    assert E_matrix.flatten()[0] > 0
    npt.assert_array_equal(E_matrix, E_matrix.flatten()[0])


@pytest.mark.parametrize('max_order_k', [2, 3])
def test_radiosity_reference(max_order_k):
    """Test if the results changes."""
    X = 10
    Y = 20
    Z = 15
    patch_size = 5
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

    ## new approach
    radi = sp.RadiosityKang(
        [ground, A_wall, B_wall], patch_size, max_order_k, ir_length_s,
        speed_of_sound=343, sampling_rate=sampling_rate)

    radi.run(source)

    receiver = Receiver([20, 2, 1], [0, 1, 0], [0, 0, 1])
    irs_new = radi.energy_at_receiver(receiver)
    signal = pf.Signal(irs_new, sampling_rate)
    signal.time /= np.max(np.abs(signal.time))

    test_path = os.path.join(
        os.path.dirname(__file__), 'test_data')

    # write test file
    if create_reference_files:
        pf.io.write(
            os.path.join(
                test_path,
                f'simulation_X{X}_k{max_order_k}_{patch_size}m.far'),
            signal=signal)

    result = pf.io.read(
        os.path.join(
            test_path,
            f'simulation_X{X}_k{max_order_k}_{patch_size}m.far'))

    npt.assert_almost_equal(
        result['signal'].time[0, ...], signal.time[0, ...], decimal=2)
    npt.assert_almost_equal(result['signal'].times, signal.times)


@pytest.mark.parametrize('max_order_k', [2, 3])
def test_radiosity_reference_with_read_write(max_order_k, tmpdir):
    """Test if the results changes."""
    X = 10
    Y = 20
    Z = 15
    patch_size = 5
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

    ## new approach
    radi = sp.RadiosityKang(
        [ground, A_wall, B_wall], patch_size, max_order_k, ir_length_s,
        speed_of_sound=343, sampling_rate=sampling_rate)

    radi.run(source)
    path = os.path.join(tmpdir, 'radiosity.far')
    radi.write(path)
    radi = sp.RadiosityKang.from_read(path)
    receiver = Receiver([20, 2, 1], [0, 1, 0], [0, 0, 1])
    irs_new = radi.energy_at_receiver(receiver)
    signal = pf.Signal(irs_new, sampling_rate)
    signal.time /= np.max(np.abs(signal.time))

    test_path = os.path.join(
        os.path.dirname(__file__), 'test_data')

    # write test file
    result = pf.io.read(
        os.path.join(
            test_path,
            f'simulation_X{X}_k{max_order_k}_{patch_size}m.far'))

    npt.assert_almost_equal(
        result['signal'].time[0, ...], signal.time[0, ...], decimal=2)
    npt.assert_almost_equal(result['signal'].times, signal.times)


@pytest.mark.parametrize('patch_size', [
    0.5,
    1,
    ])
@pytest.mark.parametrize('i_wall', [0, 1, 2, 3, 4, 5])
def test_init_energy_exchange_normal(sample_walls, patch_size, i_wall):
    """Test init energy exchange."""
    path_sofa = os.path.join(
        os.path.dirname(__file__), 'test_data',
        f'reference_matrix_directional_patch_size{patch_size}.far')
    patches = sp.PatchesKang(sample_walls[i_wall], patch_size, [], 0)
    max_order_k = 3
    ir_length_s = 5
    source = SoundSource([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1])
    patches.init_energy_exchange(
        max_order_k, ir_length_s, source, 1000, 346.18)
    if create_reference_files and sample_walls[i_wall] == sample_walls[0]:
        pf.io.write(path_sofa, E_matrix=patches.E_matrix)
    data = pf.io.read(path_sofa)
    assert np.sum(patches.E_matrix>0) > 0
    npt.assert_almost_equal(
        data['E_matrix'], patches.E_matrix, decimal=4)


@pytest.mark.parametrize('parallel_walls', [
    [0, 1], [1, 0], [2, 3], [3, 2], [4, 5], [5, 4],
    ])
@pytest.mark.parametrize('patch_size', [
    0.5,
    1,
    ])
def test_calc_form_factor_parallel(sample_walls, parallel_walls, patch_size):
    """Test form factor calculation for parallel walls."""
    wall_source = sample_walls[parallel_walls[0]]
    wall_receiver = sample_walls[parallel_walls[1]]
    path_sofa = os.path.join(
        os.path.dirname(__file__), 'test_data',
        f'reference_form_factor_parallel_size{patch_size}.far')
    patch_1 = sp.PatchesKang(wall_source, patch_size, [1], 0)
    patch_2 = sp.PatchesKang(wall_receiver, patch_size, [0], 1)
    patches = [patch_1, patch_2]
    patch_1.calculate_form_factor(patches)
    if create_reference_files and (
            parallel_walls[0] == 0) and (parallel_walls[1] == 1):
        pf.io.write(path_sofa, form_factor=patch_1.form_factors)
    data = pf.io.read(path_sofa)
    npt.assert_almost_equal(data['form_factor'], patch_1.form_factors)

@pytest.mark.parametrize('perpendicular_walls', [
    [0, 2], [0, 3], [0, 4], [0, 5],
    [1, 2], [1, 3], [1, 4], [1, 5],
    [2, 0], [2, 1], [2, 4], [2, 5],
    [3, 0], [3, 1], [3, 4], [3, 5],
    [4, 0], [4, 1], [4, 2], [4, 3],
    [5, 0], [5, 1], [5, 2], [5, 3],
    ])
@pytest.mark.parametrize('patch_size', [
    0.5,
    1,
    ])
def test_calc_form_factor_perpendicular(
        sample_walls, perpendicular_walls, patch_size):
    """Test form factor calculation for perpendicular walls."""
    wall_source = sample_walls[perpendicular_walls[0]]
    wall_receiver = sample_walls[perpendicular_walls[1]]
    idx_sort = [0, 0] if patch_size == 1 else perpendicular_walls
    path_sofa = os.path.join(
        os.path.dirname(__file__), 'test_data',
        f'reference_form_factor_perpendicular_{idx_sort[0]}_{idx_sort[1]}'
        f'_size{patch_size}.far')
    patch_1 = sp.PatchesKang(wall_source, patch_size, [1], 0)
    patch_2 = sp.PatchesKang(wall_receiver, patch_size, [0], 1)
    patches = [patch_1, patch_2]
    patch_1.calculate_form_factor(patches)
    if create_reference_files:
        pf.io.write(path_sofa, form_factor=patch_1.form_factors)
    data = pf.io.read(path_sofa)
    npt.assert_almost_equal(data['form_factor'], patch_1.form_factors)


@pytest.mark.parametrize('perpendicular_walls', [
    [0, 2], [0, 3], [0, 4], [0, 5],
    [1, 2], [1, 3], [1, 4], [1, 5],
    [2, 0], [2, 1], [2, 4], [2, 5],
    [3, 0], [3, 1], [3, 4], [3, 5],
    [4, 0], [4, 1], [4, 2], [4, 3],
    [5, 0], [5, 1], [5, 2], [5, 3],
    ])
@pytest.mark.parametrize('patch_size', [
    0.5,
    ])
def test_calc_form_factor_perpendicular_distance(
        sample_walls, perpendicular_walls, patch_size):
    """Test form factor calculation for perpendicular walls."""
    wall_source = sample_walls[perpendicular_walls[0]]
    wall_receiver = sample_walls[perpendicular_walls[1]]
    path_sofa = os.path.join(
        os.path.dirname(__file__), 'test_data',
        f'reference_form_factor_perpendicular_size{patch_size}.far')
    patch_1 = sp.PatchesKang(wall_source, patch_size, [1], 0)
    patch_2 = sp.PatchesKang(wall_receiver, patch_size, [0], 1)
    patches = [patch_1, patch_2]
    patch_1.calculate_form_factor(patches)
    ff_sort = patch_1.form_factors.flatten()
    ff_sort.sort()
    if create_reference_files and (
            perpendicular_walls[0] == 0) and (perpendicular_walls[1] == 2):
        pf.io.write(path_sofa, form_factor=ff_sort)
    data = pf.io.read(path_sofa)
    npt.assert_almost_equal(data['form_factor'], ff_sort)


@pytest.mark.parametrize('perpendicular_walls', [
    [0, 2], [2, 0],
    ])
@pytest.mark.parametrize('patch_size', [
    0.5,
    1,
    ])
def test_energy_exchange(
        sample_walls, perpendicular_walls, patch_size):
    """Test energy exchange."""
    max_order_k=3
    ir_length_s=5
    wall_source = sample_walls[perpendicular_walls[0]]
    wall_receiver = sample_walls[perpendicular_walls[1]]
    path_sofa = os.path.join(
        os.path.dirname(__file__), 'test_data',
        f'reference_energy_exchange_size{patch_size}.far')
    patch_1 = sp.PatchesKang(
        wall_source, patch_size, [1], 0)
    patch_2 = sp.PatchesKang(
        wall_receiver, patch_size, [0], 1)
    source = SoundSource([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1])
    patches = [patch_1, patch_2]
    patch_1.calculate_form_factor(patches)
    patch_2.calculate_form_factor(patches)
    patch_1.init_energy_exchange(
        max_order_k, ir_length_s, source, 1000, 346.18)
    patch_2.init_energy_exchange(
        max_order_k, ir_length_s, source, 1000, 346.18)
    for k in range(1, max_order_k+1):
        patch_1.calculate_energy_exchange(
            patches, k, speed_of_sound=346.18, E_sampling_rate=1000)
        patch_2.calculate_energy_exchange(
            patches, k, speed_of_sound=346.18, E_sampling_rate=1000)

    if create_reference_files and (
            perpendicular_walls[0] == 0) and (perpendicular_walls[1] == 2):
        pf.io.write(path_sofa, E_matrix=patch_1.E_matrix)
    data = pf.io.read(path_sofa)

    assert np.sum(patch_1.E_matrix>0) > 0
    npt.assert_almost_equal(
        10*np.log10(data['E_matrix']),
        10*np.log10(patch_1.E_matrix), decimal=1)


def test_Patch_to_from_dict(sample_walls):
    """Test Patches from dict."""
    perpendicular_walls = [0, 2]
    patch_size = 0.5
    max_order_k=3
    ir_length_s=5
    wall_source = sample_walls[perpendicular_walls[0]]
    patch_1 = sp.PatchesKang(
        wall_source, patch_size, [1], 0)
    source = SoundSource([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1])
    patch_1.init_energy_exchange(
        max_order_k, ir_length_s, source, 1000, 346.18)
    reconstructed_patch = sp.PatchesKang.from_dict(
        patch_1.to_dict())
    npt.assert_array_equal(reconstructed_patch.E_matrix, patch_1.E_matrix)
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


def test_radiosity_to_from_dict():
    """Test Radiosity the results changes for to and from dict."""
    max_order_k = 3
    patch_size = 5
    X = 10
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

    ## new approach
    radi = sp.RadiosityKang(
        [ground, A_wall, B_wall], patch_size, max_order_k, ir_length_s,
        speed_of_sound=343, sampling_rate=sampling_rate)

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


def test_radiosity_read_write(tmpdir):
    """Test Radiosity the results changes for read and write."""
    max_order_k = 3
    patch_size = 5
    X = 10
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

    ## new approach
    radi = sp.RadiosityKang(
        [ground, A_wall, B_wall], patch_size, max_order_k, ir_length_s,
        speed_of_sound=343, sampling_rate=sampling_rate)

    radi.run(source)

    json_path = os.path.join(tmpdir, 'radi.far')
    radi.write(json_path)
    radi_reconstructed = sp.RadiosityKang.from_read(json_path)

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


@pytest.mark.parametrize('patch_size', [1, 0.5, 0.2, 1/3])
def test_init_energy_larger_0(patch_size):
    """Test if the results changes for rotated walls."""
    ir_length_s = 1
    sampling_rate = 1
    max_order_k = 0
    speed_of_sound = 343
    walls = sp.testing.shoebox_room_stub(1, 1, 1)
    # create geometry
    source = sp.geometry.SoundSource([0.5, 0.5, 0.5], [0, 1, 0], [0, 0, 1])

    ## new approach
    radi = sp.RadiosityKang(
        walls, patch_size, max_order_k, ir_length_s,
        speed_of_sound=speed_of_sound, sampling_rate=sampling_rate,
        absorption=0.1)

    # run init energy
    # B. First-order patch sources
    for patches in radi.patch_list:
        patches.init_energy_exchange(
            radi.max_order_k, radi.ir_length_s, source,
            sampling_rate=radi.sampling_rate, speed_of_sound=speed_of_sound)

    E_matrix= np.concatenate([
        radi.patch_list[i].E_matrix for i in range(6)], axis=-2)

    # rotate all walls
    assert E_matrix.flatten()[0] > 0

