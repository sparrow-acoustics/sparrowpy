
"""Test the radiosity module with directional Patches."""
import os

import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
import sparrowpy as sp
import sofar as sf


def check_energy_conservation(
        receivers, receiver_weights, data, absorption_data=0):
    integration_weights = receiver_weights
    integration_weights *= 2 * np.pi / np.sum(integration_weights)
    integration_weights = integration_weights * np.cos(receivers.colatitude)
    integration_weights = integration_weights[..., np.newaxis]
    for i_source in range(data.cshape[0]):
        energy = np.sum(data.freq[i_source] * integration_weights, axis=0)
        npt.assert_almost_equal(
            energy, 1 - absorption_data, decimal=1,
            err_msg=f"source {i_source}")


def check_reciprocity(data):
    data_array = data.freq
    for i_source in range(data_array.shape[0]):
        for i_receiver in range(i_source + 1, data_array.shape[1]):
            npt.assert_almost_equal(
                data_array[i_source, i_receiver],
                data_array[i_receiver, i_source])


def test_create_from_scattering_with_valid_data(tmp_path):
    # Prepare test data
    scattering_data = [0, 0.2, 1]
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmp_path, "test_brdf.sofa")
    coords = pf.samplings.sph_gaussian(sh_order=1)
    coords = coords[coords.z > 0]

    # Call the function
    sp.brdf.create_from_scattering(
        coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        file_path=file_path,
        )
    data, s, r = pf.io.read_sofa(file_path)

    # test coords
    npt.assert_almost_equal(s.cartesian, coords.cartesian)
    npt.assert_almost_equal(r.cartesian, coords.cartesian)
    # Assert the expected outcome
    assert data.freq.shape == (4, 4, 3)
    assert data.freq.shape == (s.csize, r.csize, 3)
    # test
    for i in range(4):
        for j in range(4):
            npt.assert_almost_equal(data.freq[i, j], data.freq[j, i])


def test_create_from_scattering_1(tmp_path):
    # Prepare test data
    scattering_data = np.ones((3, ))
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmp_path, "test_brdf.sofa")
    coords = pf.samplings.sph_gaussian(sh_order=1)
    coords = coords[coords.z > 0]

    # Call the function
    sp.brdf.create_from_scattering(
        coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        file_path=file_path,
        )
    data, s, r = pf.io.read_sofa(file_path)

    # Assert the expected outcome
    assert data.freq.shape == (4, 4, 3)
    assert data.freq.shape == (s.csize, r.csize, 3)
    # test
    npt.assert_almost_equal(data.freq, 1 / np.pi)


def test_create_from_scattering_0(tmp_path):
    # Prepare test data
    scattering_data = np.zeros((3, ))
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmp_path, "test_brdf.sofa")
    coords = pf.samplings.sph_gaussian(sh_order=1)
    coords = coords[coords.z > 0]

    # Call the function
    sp.brdf.create_from_scattering(
        coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        file_path=file_path,
        )
    data, s, r = pf.io.read_sofa(file_path)

    # Assert the expected outcome
    assert data.freq.shape == (4, 4, 3)
    assert data.freq.shape == (s.csize, r.csize, 3)
    # test
    npt.assert_almost_equal(data.freq[0, 2], 1.10265779)
    npt.assert_almost_equal(data.freq[0, not 2], 0)


def test_create_from_scattering_0_3(tmp_path):
    # Prepare test data
    scattering_data = 0.3
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmp_path, "test_brdf.sofa")
    coords = pf.samplings.sph_gaussian(sh_order=1)
    coords = coords[coords.z > 0]

    # Call the function
    sp.brdf.create_from_scattering(
        coords, coords,
        pf.FrequencyData(scattering_data + np.zeros((3, )), frequency_data),
        file_path=file_path,
        )
    data, s, r = pf.io.read_sofa(file_path)

    # Assert the expected outcome
    assert data.freq.shape == (4, 4, 3)
    assert data.freq.shape == (s.csize, r.csize, 3)
    desired_scat = scattering_data / np.pi
    desired_spec = 0.86735342 - desired_scat

    # test
    npt.assert_almost_equal(data.freq[0, 2], desired_spec + desired_scat)
    npt.assert_almost_equal(data.freq[0, not 2], desired_scat)


def test_create_from_scattering_0_3_with_absorption(tmp_path):
    # Prepare test data
    scattering_data = 0.3
    absorption_data = 0.3
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmp_path, "test_brdf.sofa")
    coords = pf.samplings.sph_gaussian(sh_order=1)
    coords = coords[coords.z > 0]

    # Call the function
    sp.brdf.create_from_scattering(
        coords, coords,
        pf.FrequencyData(scattering_data + np.zeros((3, )), frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
        file_path=file_path,
        )
    data, s, r = pf.io.read_sofa(file_path)

    # Assert the expected outcome
    assert data.freq.shape == (4, 4, 3)
    assert data.freq.shape == (s.csize, r.csize, 3)
    desired_scat = scattering_data / np.pi * (1 - absorption_data)
    desired_spec = 0.60714739 - desired_scat

    # test
    npt.assert_almost_equal(data.freq[0, 2], desired_spec + desired_scat)
    npt.assert_almost_equal(data.freq[0, not 2], desired_scat)


def test_create_from_scattering_with_invalid_data(tmp_path):
    # Prepare test data
    scattering_data = pf.FrequencyData(1, 100)
    coords = pf.samplings.sph_gaussian(sh_order=1)

    # Call the function and expect it to raise an error
    with pytest.raises(
            TypeError,
            match="scattering_coefficient must be a pf.FrequencyData object"):
        sp.brdf.create_from_scattering(
            coords, coords, 'invalid', file_path=tmp_path)
    # Call the function and expect it to raise an error
    with pytest.raises(
            TypeError,
            match="source_directions must be a pf.Coordinates object"):
        sp.brdf.create_from_scattering(
            'coords', coords, scattering_data, file_path=tmp_path)
    with pytest.raises(
            TypeError,
            match="receiver_directions must be a pf.Coordinates object"):
        sp.brdf.create_from_scattering(
            coords, 'coords', scattering_data, file_path=tmp_path)


@pytest.mark.parametrize('scattering_data', [0, 0.3, 1])
@pytest.mark.parametrize('absorption_data', [0, 0.3, 1])
@pytest.mark.parametrize('sh_order', [5, 11])
def test_create_from_scattering_energy_conservation(
        tmp_path, scattering_data, absorption_data, sh_order):
    # Prepare test data
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmp_path, "test_brdf.sofa")
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]

    # Call the function
    sp.brdf.create_from_scattering(
        coords, coords,
        pf.FrequencyData(scattering_data + np.zeros((3, )), frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
        file_path=file_path,
        )
    sofa = sf.read_sofa(file_path)
    data, s, r = pf.io.convert_sofa(sofa)

    # test energy conservation
    integration_weights = sofa.ReceiverWeights
    integration_weights *= 2 * np.pi / np.sum(integration_weights)
    integration_weights = integration_weights * np.cos(r.colatitude)
    integration_weights = integration_weights[..., np.newaxis]
    for i_source in range(s.csize):
        energy = np.sum(data.freq[i_source] * integration_weights, axis=0)
        npt.assert_almost_equal(energy, 1 - absorption_data, decimal=1)


@pytest.mark.parametrize('scattering_data', [0, 0.3, 1])
@pytest.mark.parametrize('absorption_data', [0, 0.3, 1])
@pytest.mark.parametrize('sh_order', [5, 11])
def test_create_from_scattering_reciprocal_principle(
        tmp_path, scattering_data, absorption_data, sh_order):
    # Prepare test data
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmp_path, "test_brdf.sofa")
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]

    # Call the function
    sp.brdf.create_from_scattering(
        coords, coords,
        pf.FrequencyData(scattering_data + np.zeros((3, )), frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
        file_path=file_path,
        )
    sofa = sf.read_sofa(file_path)
    data, s, r = pf.io.convert_sofa(sofa)

    # test energy conservation
    for i_source in range(s.csize):
        for i_receiver in range(r.csize):
            npt.assert_almost_equal(
                data.freq[i_source, i_receiver],
                data.freq[i_receiver, i_source])


@pytest.mark.parametrize('absorption_data', [0, 0.5])
@pytest.mark.parametrize('sh_order', [5, 11])
def test_directional_energy_conservation_specular(
        tmp_path, absorption_data, sh_order):
    # Prepare test data
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmp_path, "test_brdf.sofa")
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]
    scattering_data = np.zeros((coords.csize, coords.csize, 3))
    scattering_data[np.arange(coords.csize), np.arange(coords.csize)] = 1
    npt.assert_array_equal(np.sum(scattering_data, axis=1), 1)

    # Call the function
    sp.brdf.create_from_directional_scattering(
        coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
        file_path=file_path,
        )
    sofa = sf.read_sofa(file_path)
    data, _s, r = pf.io.convert_sofa(sofa)

    # test energy conservation and reciprocity
    check_energy_conservation(
        r, sofa.ReceiverWeights, data, absorption_data)
    check_reciprocity(data)


@pytest.mark.parametrize('absorption_data', [0])
@pytest.mark.parametrize('sh_order', [5, 11])
def test_directional_energy_conservation_diffuse(
        tmp_path, absorption_data, sh_order):
    # Prepare test data
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmp_path, "test_brdf.sofa")
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]
    cos_factor = (np.cos(
            coords.colatitude) * coords.weights)
    scattering_data = np.ones(
        (coords.csize, coords.csize, 3)) * 4 * cos_factor[..., np.newaxis]

    # Call the function
    sp.brdf.create_from_directional_scattering(
        coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
        file_path=file_path,
        )
    sofa = sf.read_sofa(file_path)
    data, _s, r = pf.io.convert_sofa(sofa)

    # test energy conservation and reciprocity
    check_energy_conservation(
        r, sofa.ReceiverWeights, data, absorption_data)
    check_reciprocity(data)

    npt.assert_almost_equal(data.freq, 1 / np.pi, decimal=2)


@pytest.mark.parametrize('absorption_data', [0, 0.3, 1])
@pytest.mark.parametrize('sh_order', [5, 11])
def test_directional_reciprocal_principle_specular(
        tmp_path, absorption_data, sh_order):
    # Prepare test data
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmp_path, "test_brdf.sofa")
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]
    scattering_data = np.zeros((coords.csize, coords.csize, 3))
    scattering_data[np.arange(coords.csize), np.arange(coords.csize)] = 1
    npt.assert_array_equal(np.sum(scattering_data, axis=1), 1)

    # Call the function
    sp.brdf.create_from_directional_scattering(
        coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
        file_path=file_path,
        )
    sofa = sf.read_sofa(file_path)
    data, _, r = pf.io.convert_sofa(sofa)

    # test energy conservation and reciprocity
    check_energy_conservation(
        r, sofa.ReceiverWeights, data, absorption_data)
    check_reciprocity(data)


@pytest.mark.parametrize('absorption_data', [0, 0.5])
@pytest.mark.parametrize('sh_order', [5, 11])
def test_directional_energy_conservation_two_directions(
        tmp_path, absorption_data, sh_order):
    # Prepare test data
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmp_path, "test_brdf.sofa")
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]
    scattering_data = np.zeros((coords.csize, coords.csize, 3))
    scattering_data[np.arange(coords.csize), np.arange(coords.csize)] = 0.5
    refl = coords.copy()
    refl.azimuth += np.pi
    idx = coords.find_nearest(refl)[0]
    scattering_data[np.arange(coords.csize), idx] = 0.5
    npt.assert_array_equal(np.sum(scattering_data, axis=1), 1)

    # Call the function
    sp.brdf.create_from_directional_scattering(
        coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
        file_path=file_path,
        )
    sofa = sf.read_sofa(file_path)
    data, _s, r = pf.io.convert_sofa(sofa)

    # test energy conservation and reciprocity
    check_energy_conservation(
        r, sofa.ReceiverWeights, data, absorption_data)
    check_reciprocity(data)


@pytest.mark.parametrize('absorption_data', [0, 0.5])
@pytest.mark.parametrize('sh_order', [5, 11])
def test_directional_energy_conservation_two_directions_rand(
        tmp_path, absorption_data, sh_order):
    # Prepare test data
    frequency_data = [100, 200, 400]
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]
    scattering_data = np.zeros((coords.csize, coords.csize, 3))
    scattering_data[
        np.arange(coords.csize), np.arange(coords.csize)] = 0.5
    specular_coords = coords.copy()
    specular_coords.azimuth += np.pi
    idx = coords.find_nearest(specular_coords)[0]
    scattering_data[
        np.arange(coords.csize), idx] += 0.5
    npt.assert_array_equal(np.sum(scattering_data, axis=1), 1)

    # Call the function
    file_path = os.path.join(tmp_path, "test_brdf.sofa")
    sp.brdf.create_from_directional_scattering(
        coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
        file_path=file_path,
        )
    sofa = sf.read_sofa(file_path)
    data, _s, r = pf.io.convert_sofa(sofa)

    # test energy conservation and reciprocity
    check_energy_conservation(
        r, sofa.ReceiverWeights, data, absorption_data)
    check_reciprocity(data)


@pytest.mark.parametrize('absorption_data', [0, 0.5])
@pytest.mark.parametrize('sh_order', [5, 11])
def test_directional_random_directional_scattering(
        tmp_path, absorption_data, sh_order):
    # Prepare test data
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmp_path, "test_brdf.sofa")
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]
    rng = np.random.default_rng(1337)
    scattering_data = rng.random((coords.csize, coords.csize, 3))
    scattering_data /= np.sum(scattering_data, axis=1, keepdims=True)
    npt.assert_almost_equal(np.sum(scattering_data, axis=1), 1)

    # Call the function
    sp.brdf.create_from_directional_scattering(
        coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
        file_path=file_path,
        )
    sofa = sf.read_sofa(file_path)
    data, _, r = pf.io.convert_sofa(sofa)

    # test energy conservation and reciprocity
    check_energy_conservation(
        r, sofa.ReceiverWeights, data, absorption_data)
