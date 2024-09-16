
"""Test the radiosity module with directional Patches."""
import os

import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
import sparapy as sp
import sofar as sf


create_reference_files = False


@pytest.mark.parametrize('absorption_data', [0, 0.5])
@pytest.mark.parametrize('sh_order', [5, 11])
def test_energy_conservation_specular(
        tmpdir, absorption_data, sh_order):
    # Prepare test data
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]
    scattering_data = np.zeros((coords.csize, coords.csize, 3))
    scattering_data[np.arange(coords.csize), np.arange(coords.csize)] = 1
    npt.assert_array_equal(np.sum(scattering_data, axis=1), 1)

    # Call the function
    sp.brdf.create_from_directional_scattering(
        file_path, coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
        )
    sofa = sf.read_sofa(file_path)
    data, s, r = pf.io.convert_sofa(sofa)

    # test energy conservation
    integration_weights = sofa.ReceiverWeights
    integration_weights *= 2*np.pi/np.sum(integration_weights)
    integration_weights = integration_weights*np.cos(s.colatitude)
    integration_weights = integration_weights[..., np.newaxis]
    for i_source in range(s.csize):
        energy = np.sum(data.freq[i_source]*integration_weights, axis=0)
        npt.assert_almost_equal(
            energy, 1 - absorption_data, decimal=1,
            err_msg=f"source {i_source}")


@pytest.mark.parametrize('absorption_data', [0])
@pytest.mark.parametrize('sh_order', [5, 11])
def test_energy_conservation_diffuse(
        tmpdir, absorption_data, sh_order):
    # Prepare test data
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]
    scattering_data = np.ones((coords.csize, coords.csize, 3)) / coords.csize

    # Call the function
    sp.brdf.create_from_directional_scattering(
        file_path, coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
        )
    sofa = sf.read_sofa(file_path)
    data, s, r = pf.io.convert_sofa(sofa)

    npt.assert_almost_equal(data.freq, 1/np.pi, decimal=2)


@pytest.mark.parametrize('absorption_data', [0, 0.3, 1])
@pytest.mark.parametrize('sh_order', [5, 11])
def test_reciprocal_principle_specular(
        tmpdir, absorption_data, sh_order):
    # Prepare test data
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]
    scattering_data = np.zeros((coords.csize, coords.csize, 3))
    scattering_data[np.arange(coords.csize), np.arange(coords.csize)] = 1
    npt.assert_array_equal(np.sum(scattering_data, axis=1), 1)

    # Call the function
    sp.brdf.create_from_directional_scattering(
        file_path, coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
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
def test_energy_conservation_two_directions(
        tmpdir, absorption_data, sh_order):
    # Prepare test data
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)
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
        file_path, coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
        )
    sofa = sf.read_sofa(file_path)
    data, s, r = pf.io.convert_sofa(sofa)

    # test energy conservation
    integration_weights = sofa.ReceiverWeights
    integration_weights *= 2*np.pi/np.sum(integration_weights)
    integration_weights = integration_weights*np.cos(s.colatitude)
    integration_weights = integration_weights[..., np.newaxis]
    for i_source in range(s.csize):
        energy = np.sum(data.freq[i_source]*integration_weights, axis=0)
        npt.assert_almost_equal(
            energy, 1 - absorption_data, decimal=1,
            err_msg=f"source {i_source}")


@pytest.mark.parametrize('absorption_data', [0, 0.5])
@pytest.mark.parametrize('sh_order', [5, 11])
def test_energy_conservation_two_directions_rand(
        tmpdir, absorption_data, sh_order):
    # Prepare test data
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]
    scattering_data = np.zeros((coords.csize, coords.csize, 3))
    scattering_data[
        np.arange(coords.csize), np.arange(coords.csize)] = 0.5
    scattering_data[
        np.arange(coords.csize), np.arange(coords.csize)[::-1]] += 0.5
    npt.assert_array_equal(np.sum(scattering_data, axis=1), 1)

    # Call the function
    sp.brdf.create_from_directional_scattering(
        file_path, coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
        )
    sofa = sf.read_sofa(file_path)
    data, s, r = pf.io.convert_sofa(sofa)

    # test energy conservation
    integration_weights = sofa.ReceiverWeights
    integration_weights *= 2*np.pi/np.sum(integration_weights)
    integration_weights = integration_weights*np.cos(s.colatitude)
    integration_weights = integration_weights[..., np.newaxis]
    for i_source in range(s.csize):
        energy = np.sum(data.freq[i_source]*integration_weights, axis=0)
        npt.assert_almost_equal(
            energy, 1 - absorption_data, decimal=1,
            err_msg=f"source {i_source}")


@pytest.mark.parametrize('absorption_data', [0, 0.5])
@pytest.mark.parametrize('sh_order', [5, 11])
def test_random_directional_scattering(
        tmpdir, absorption_data, sh_order):
    # Prepare test data
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]
    scattering_data = np.random.rand(coords.csize, coords.csize, 3)
    scattering_data /= np.sum(scattering_data, axis=1, keepdims=True)
    npt.assert_almost_equal(np.sum(scattering_data, axis=1), 1)

    # Call the function
    sp.brdf.create_from_directional_scattering(
        file_path, coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
        )
    sofa = sf.read_sofa(file_path)
    data, s, r = pf.io.convert_sofa(sofa)

    # test energy conservation
    integration_weights = sofa.ReceiverWeights
    integration_weights *= 2*np.pi/np.sum(integration_weights)
    integration_weights = integration_weights*np.cos(s.colatitude)
    integration_weights = integration_weights[..., np.newaxis]
    for i_source in range(s.csize):
        energy = np.sum(data.freq[i_source]*integration_weights, axis=0)
        npt.assert_almost_equal(
            energy, 1 - absorption_data, decimal=1,
            err_msg=f"source {i_source}")
