"""Test radiosity module."""
import numpy as np
import numpy.testing as npt
import pytest
import os
import pyfar as pf

import sparapy as sp
import time
import matplotlib.pyplot as plt


create_reference_files = False

def test_init(sample_walls):
    radiosity = sp.DRadiosityFast.from_polygon(sample_walls, 0.2)
    npt.assert_almost_equal(radiosity.patches_points.shape, (150, 4, 3))
    npt.assert_almost_equal(radiosity.patches_area.shape, (150))
    npt.assert_almost_equal(radiosity.patches_center.shape, (150, 3))
    npt.assert_almost_equal(radiosity.patches_size.shape, (150, 3))
    npt.assert_almost_equal(radiosity.patches_normal.shape, (150, 3))


def test_check_visibility(sample_walls):
    radiosity = sp.DRadiosityFast.from_polygon(sample_walls, 0.2)
    radiosity.bake_geometry()
    npt.assert_almost_equal(radiosity._visibility_matrix.shape, (150, 150))
    # npt.assert_array_equal(
    #     radiosity._visibility_matrix, radiosity._visibility_matrix.T)
    npt.assert_array_equal(radiosity._visibility_matrix[:25, :25], False)
    npt.assert_array_equal(radiosity._visibility_matrix[:25, 25:], True)
    npt.assert_array_equal(radiosity._visibility_matrix[25:50, 25:50], False)
    assert np.sum(radiosity._visibility_matrix) == 25*5*25*6/2


def test_check_visibility_wrapper(sample_walls):
    radiosity = sp.DRadiosityFast.from_polygon(sample_walls, 0.2)
    radiosity.bake_geometry()
    visibility_matrix = sp.radiosity_fast.geometry.check_visibility(
        radiosity.patches_center, radiosity.patches_normal)
    npt.assert_almost_equal(radiosity._visibility_matrix, visibility_matrix)


def test_compute_form_factors(sample_walls):
    radiosity = sp.DRadiosityFast.from_polygon(sample_walls, 0.2)
    radiosity.bake_geometry()
    npt.assert_almost_equal(radiosity.form_factors.shape, (150, 150))
    radiosity.bake_geometry(ff_method='universal')
    npt.assert_almost_equal(radiosity.form_factors.shape, (150, 150))

def test_compute_form_factor_vals(sample_walls):
    # just to compile the programs
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(sample_walls, 1)
    radiosity.bake_geometry(ff_method='kang')
    radiosity.bake_geometry(ff_method='universal')

    # actual run
    radiosity = sp.radiosity_fast.DRadiosityFast.from_polygon(sample_walls, .2)
    
    t0 = time.time()
    radiosity.bake_geometry(ff_method='universal')
    tuniv = time.time()-t0
    univ = radiosity.form_factors

    t0 = time.time()
    radiosity.bake_geometry(ff_method='kang')
    tkang = time.time()-t0
    kang = radiosity.form_factors

    diff = np.abs(kang-univ)

    plt.figure()
    plt.imshow(diff/kang * 100)
    plt.colorbar()
    plt.savefig(".\\tests\\test_data\\kang_vs_univ_ff_rel_diff.png")
    
    diff = diff[kang!=0]
    univ = univ[kang!=0]
    kang = kang[kang!=0]

    plt.figure()
    plt.title("form factor values")
    plt.plot(kang.flatten(), label="kang")
    plt.plot(univ.flatten(), label="univ")
    plt.legend()
    plt.savefig(".\\tests\\test_data\\kang_vs_univ_ff_abs_values.png")

    plt.figure()
    plt.title("form factor difference")
    plt.plot(diff, label="kang-univ")
    plt.legend()
    plt.savefig(".\\tests\\test_data\\kang_vs_univ_ff_abs_diff.png")

    maximo = np.max(diff)
    rms = np.sqrt(np.sum(np.square(diff)))/(diff.shape[0]**2)
    mmean = np.mean(diff)

    maximo_rel = 100*maximo/kang[np.argmax(diff)]

    rms_rel = 100*rms/np.mean(kang)

    mean_rel = 100*mmean/np.mean(kang)

    assert maximo_rel < 25
    assert mean_rel < 10
    assert rms_rel < 1
    


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
def test_calc_form_factor_perpendicular_distance(
        sample_walls, walls, patch_size):
    """Test form factor calculation for perpendicular walls."""
    wall_source = sample_walls[walls[0]]
    wall_receiver = sample_walls[walls[1]]
    patch_1 = sp.radiosity.Patches(wall_source, patch_size, [1], 0)
    patch_2 = sp.radiosity.Patches(wall_receiver, patch_size, [0], 1)
    patches = [patch_1, patch_2]
    patch_1.calculate_form_factor(patches)
    patch_2.calculate_form_factor(patches)

    radiosity = sp.DRadiosityFast.from_polygon(
        [wall_source, wall_receiver], patch_size)
    radiosity.bake_geometry()

    patch_pos = np.array([patch.center for patch in patch_1.patches])
    if (np.abs(patch_pos- radiosity.patches_center[:4, :])<1e-5).all():
        npt.assert_almost_equal(
            radiosity.form_factors[:4, 4:], patch_1.form_factors)
    else:
        npt.assert_almost_equal(
            radiosity.form_factors[:4, 4:], patch_1.form_factors.T)

    patch_pos = np.array([patch.center for patch in patch_2.patches])
    if (np.abs(patch_pos- radiosity.patches_center[4:, :])<1e-5).all():
        npt.assert_almost_equal(radiosity.form_factors[4:, :4], 0)
    else:
        npt.assert_almost_equal(radiosity.form_factors[4:, :4], 0)


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

    radiosity = sp.DRadiosityFast.from_polygon(
        walls, patch_size)
    data, sources, receivers = sofa_data_diffuse
    radiosity.set_wall_scattering(
        np.arange(len(walls)), data, sources, receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.set_wall_absorption(
        np.arange(len(walls)),
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.bake_geometry()
    # radiosity.calculate_energy_exchange(k)
    # radiosity.init_energy(source_pos)
    # histogram = radiosity.collect_energy_receiver(
    #     receiver_pos, speed_of_sound=speed_of_sound,
    #     histogram_time_resolution=time_resolution,
    #     histogram_time_length=length_histogram)

    form_factors_from_tilde = np.max(radiosity._form_factors_tilde, axis=0)
    for i in range(4):
        npt.assert_almost_equal(
            radiosity.form_factors[:4, 4:], form_factors_from_tilde[:4, 4:, i])
        npt.assert_almost_equal(
            radiosity.form_factors[:4, 4:].T, form_factors_from_tilde[4:, :4, i])

    for i in range(8):
        npt.assert_almost_equal(radiosity._form_factors_tilde[i, i, :, :], 0)
        npt.assert_almost_equal(radiosity._form_factors_tilde[:, i, i, :], 0)


def test_set_wall_scattering(sample_walls, sofa_data_diffuse):
    radiosity = sp.DRadiosityFast.from_polygon(
        sample_walls, 0.2)
    (data, sources, receivers) = sofa_data_diffuse
    radiosity.set_wall_scattering(np.arange(6), data, sources, receivers)
    # check shape of scattering matrix
    assert len(radiosity._scattering) == 1
    npt.assert_almost_equal(radiosity._scattering[0].shape, (4, 4, 4))
    npt.assert_array_equal(radiosity._scattering[0], 1)
    npt.assert_array_equal(radiosity._scattering_index, 0)
    # check source and receiver direction
    for i in range(6):
        assert (np.sum(
            radiosity._sources[i].cartesian*radiosity.walls_normal[i,:],
            axis=-1)>0).all()
        assert (np.sum(
            radiosity._receivers[i].cartesian*radiosity.walls_normal[i,:],
            axis=-1)>0).all()


def test_set_wall_scattering_different(sample_walls, sofa_data_diffuse):
    radiosity = sp.DRadiosityFast.from_polygon(
        sample_walls, 0.2)
    (data, sources, receivers) = sofa_data_diffuse
    radiosity.set_wall_scattering([0, 1, 2], data, sources, receivers)
    radiosity.set_wall_scattering([3, 4, 5], data, sources, receivers)
    # check shape of scattering matrix
    assert len(radiosity._scattering) == 2
    for i in range(2):
        npt.assert_almost_equal(radiosity._scattering[i].shape, (4, 4, 4))
        npt.assert_array_equal(radiosity._scattering[i], 1)
    npt.assert_array_equal(radiosity._scattering_index[:3], 0)
    npt.assert_array_equal(radiosity._scattering_index[3:], 1)
    # check source and receiver direction
    for i in range(6):
        assert (np.sum(
            radiosity._sources[i].cartesian*radiosity.walls_normal[i,:],
            axis=-1)>0).all()
        assert (np.sum(
            radiosity._receivers[i].cartesian*radiosity.walls_normal[i,:],
            axis=-1)>0).all()


def test_set_wall_absorption(sample_walls):
    radiosity = sp.DRadiosityFast.from_polygon(
        sample_walls, 0.2)
    radiosity.set_wall_absorption(
        np.arange(6), pf.FrequencyData([0.1, 0.2], [500, 1000]))
    npt.assert_array_equal(radiosity._absorption[0], [0.1, 0.2])
    npt.assert_array_equal(radiosity._absorption_index, 0)


def test_set_wall_absorption_different(sample_walls):
    radiosity = sp.DRadiosityFast.from_polygon(
        sample_walls, 0.2)
    radiosity.set_wall_absorption(
        [0, 1, 2], pf.FrequencyData([0.1, 0.1], [500, 1000]))
    radiosity.set_wall_absorption(
        [3, 4, 5], pf.FrequencyData([0.2, 0.2], [500, 1000]))
    npt.assert_array_equal(radiosity._absorption[0], [0.1, 0.1])
    npt.assert_array_equal(radiosity._absorption[1], [0.2, 0.2])
    npt.assert_array_equal(radiosity._absorption_index[:3], 0)
    npt.assert_array_equal(radiosity._absorption_index[3:], 1)


def test_set_air_attenuation(sample_walls):
    radiosity = sp.DRadiosityFast.from_polygon(
        sample_walls, 0.2)
    radiosity.set_air_attenuation(pf.FrequencyData([0.1, 0.2], [500, 1000]))
    npt.assert_array_equal(radiosity._air_attenuation, [0.1, 0.2])


def test_total_number_of_patches():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    result = sp.radiosity_fast.geometry.total_number_of_patches(points, 0.2)
    desired = 25
    assert result == desired


@pytest.mark.parametrize('patch_size', [
    0.5,
    1,
    ])
@pytest.mark.parametrize('source_pos', [
    np.array([0.25, 0.75, 0.3]),
    ])
@pytest.mark.parametrize('receiver_pos', [
    np.array([0.6, 0.3, 0.7]),
    ])
@pytest.mark.parametrize('max_order_k', [3])
def test_recursive_vs_old_implementation(
        patch_size, source_pos, receiver_pos, max_order_k, sample_walls,
        sofa_data_diffuse):
    # note that order k=0 means one reflection and k=1 means two reflections
    # (2nd order)
    data, sources, receivers = sofa_data_diffuse
    walls = [0, 1]

    # source_pos = np.array([0.3, 0.5, 0.5])
    # receiver_pos = np.array([0.5, 0.5, 0.5])
    wall_source = sample_walls[walls[0]]
    wall_receiver = sample_walls[walls[1]]
    walls = [wall_source, wall_receiver]
    length_histogram = 0.2
    time_resolution = 1e-3
    speed_of_sound = 346.18

    radiosity_old = sp.radiosity.Radiosity(
        walls, patch_size, max_order_k, length_histogram,
        speed_of_sound=speed_of_sound,
        sampling_rate=1/time_resolution, absorption=0)

    radiosity_old.run(
        sp.geometry.SoundSource(source_pos, [1, 0, 0], [0, 0, 1]))
    histogram_old = radiosity_old.energy_at_receiver(
        sp.geometry.Receiver(receiver_pos, [1, 0, 0], [0, 0, 1]), ignore_direct=True)

    radiosity = sp.DRadiosityFast.from_polygon(
        walls, patch_size)

    radiosity.set_wall_scattering(
        np.arange(len(walls)), data, sources, receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.set_wall_absorption(
        np.arange(len(walls)),
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.bake_geometry()

    radiosity.init_source_energy(source_pos)
    histogram = radiosity.calculate_energy_exchange_receiver(
        receiver_pos, speed_of_sound, time_resolution, length_histogram,
        threshold=0, max_time=5, max_depth=max_order_k)

    patches_center = []
    patches_normal = []
    patches_size = []
    form_factors = []
    for patch in radiosity_old.patch_list:
        form_factors.append(patch.form_factors)
        for p in patch.patches:
            patches_center.append(p.center)
            patches_size.append(p.size)
            patches_normal.append(p.normal)
    patches_center = np.array(patches_center)
    patches_normal = np.array(patches_normal)
    patches_size = np.array(patches_size)
    n_patches = patches_center.shape[0]

    # test form factors
    form_factors = np.array(form_factors).reshape((n_patches, int(n_patches/2)))
    for i in range(int(n_patches/2)):
        for j in range(int(n_patches/2)):
            if (i < j) and (i!=j):
                npt.assert_almost_equal(
                    radiosity.form_factors[i, int(j+n_patches/2)], form_factors[i, j])
                npt.assert_almost_equal(
                    radiosity.form_factors[i, int(j+n_patches/2)], form_factors[j, i])
    npt.assert_almost_equal(
        patches_center, radiosity.patches_center)
    npt.assert_almost_equal(
        patches_normal, radiosity.patches_normal)
    npt.assert_almost_equal(
        patches_size, radiosity.patches_size)

    # test form factors directivity
    for i in range(n_patches):
        for j in range(n_patches):
            if (i < j) and (i!=j):
                ffd_iji = radiosity._form_factors_tilde[i, j, i, 0]
                ffd_jij = radiosity._form_factors_tilde[j, i, j, 0]
                ff_ij = radiosity.form_factors[i, j]
                assert ffd_iji == ff_ij
                assert ffd_jij == ff_ij

    # compare histogram
    for i in range(4):
        assert np.sum(histogram[i, :])>0
        npt.assert_allclose(
            np.sum(histogram[i, :]), np.sum(histogram_old[0, :]),
            err_msg=f'histogram i_bin={i}')
        # npt.assert_almost_equal(
        # histogram[0, histogram[0,:]>0],
        # histogram_old[0, histogram_old[0,:]>0])


@pytest.mark.parametrize('patch_size', [
    1,
    ])
@pytest.mark.parametrize('source_pos', [
    np.array([2, 2, 2]),
    ])
@pytest.mark.parametrize('receiver_pos', [
    np.array([3, 4, 2]),
    ])
@pytest.mark.parametrize('max_order_k', [2])
def test_room_recursive_vs_old_implementation(
        patch_size, source_pos, receiver_pos, max_order_k,
        sofa_data_diffuse):
    # note that order k=0 means one reflection and k=1 means two reflections
    # (2nd order)
    data, sources, receivers = sofa_data_diffuse

    # source_pos = np.array([0.3, 0.5, 0.5])
    # receiver_pos = np.array([0.5, 0.5, 0.5])
    length_histogram = 0.2
    time_resolution = 1e-3
    speed_of_sound = 346.18
    walls = sp.testing.shoebox_room_stub(5, 6, 4)

    radiosity_old = sp.radiosity.Radiosity(
        walls, patch_size, max_order_k, length_histogram,
        speed_of_sound=speed_of_sound,
        sampling_rate=1/time_resolution, absorption=0)

    radiosity_old.run(
        sp.geometry.SoundSource(source_pos, [1, 0, 0], [0, 0, 1]))
    histogram_old_all = radiosity_old.energy_at_receiver(
        sp.geometry.Receiver(receiver_pos, [1, 0, 0], [0, 0, 1]),
        ignore_direct=True)
    histogram_old_1 = radiosity_old.energy_at_receiver(
        sp.geometry.Receiver(receiver_pos, [1, 0, 0], [0, 0, 1]),
        ignore_direct=True, max_order_k=max_order_k-1)
    histogram_old = histogram_old_all-histogram_old_1

    radiosity = sp.DRadiosityFast.from_polygon(
        walls, patch_size)

    radiosity.set_wall_scattering(
        np.arange(len(walls)), data, sources, receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.set_wall_absorption(
        np.arange(len(walls)),
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.bake_geometry()

    radiosity.init_source_energy(source_pos)
    histogram_all = radiosity.calculate_energy_exchange_receiver(
        receiver_pos, speed_of_sound, time_resolution, length_histogram,
        threshold=0, max_time=np.inf, max_depth=max_order_k)
    histogram_1 = radiosity.calculate_energy_exchange_receiver(
        receiver_pos, speed_of_sound, time_resolution, length_histogram,
        threshold=0, max_time=np.inf, max_depth=max_order_k-1)
    histogram = histogram_all-histogram_1

    # compare histogram
    for i in range(4):
        assert np.sum(histogram[i, :])>0
        npt.assert_allclose(
            np.sum(histogram[i, :]), np.sum(histogram_old[0, :]),
            err_msg=f'histogram i_bin={i}', rtol=0.0005)
        # npt.assert_almost_equal(
        # histogram[0, histogram[0,:]>0], histogram_old[0, histogram_old[0,:]>0])


@pytest.mark.parametrize('patch_size', [1, 0.5])
def test_recursive_reference(
        patch_size, sample_walls, sofa_data_diffuse):
    """Test if the results changes."""
    # note that order k=0 means one reflection and k=1 means two reflections
    # (2nd order)
    data, sources, receivers = sofa_data_diffuse
    data = pf.FrequencyData(data.freq[..., :1], data.frequencies[0])

    source_pos = np.array([0.3, 0.5, 0.5])
    receiver_pos = np.array([0.5, 0.5, 0.5])
    length_histogram = 0.1
    time_resolution = 1e-3
    speed_of_sound = 346.18

    radiosity = sp.DRadiosityFast.from_polygon(
        sample_walls, patch_size)

    radiosity.set_wall_scattering(
        np.arange(len(sample_walls)), data, sources, receivers)
    radiosity.set_air_attenuation(
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.set_wall_absorption(
        np.arange(len(sample_walls)),
        pf.FrequencyData(np.zeros_like(data.frequencies), data.frequencies))
    radiosity.bake_geometry()

    radiosity.init_source_energy(source_pos)
    histogram = radiosity.calculate_energy_exchange_receiver(
        receiver_pos, speed_of_sound, time_resolution, length_histogram,
        max_time=0.011)

    signal = pf.Signal(histogram, 1/time_resolution)
    signal.time /= np.max(np.abs(signal.time))

    test_path = os.path.join(
        os.path.dirname(__file__), 'test_data')

    reference_path = os.path.join(
            test_path,
            f'sim_recursive_{patch_size}.far')
    if create_reference_files:
        pf.io.write(reference_path, signal=signal)
    result = pf.io.read(reference_path)

    for i_frequency in range(signal.time.shape[0]):
        npt.assert_almost_equal(
            result['signal'].time[0, ...], signal.time[i_frequency, ...],
            decimal=4)
    npt.assert_almost_equal(result['signal'].times, signal.times)
