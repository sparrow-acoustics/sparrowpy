
"""Test the radiosity module with directional Patches."""
import os

import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
import sparapy as sp
import sofar as sf


create_reference_files = False


def test_create_from_scattering_with_valid_data(tmpdir):
    # Prepare test data
    scattering_data = [0, 0.2, 1]
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)
    coords = pf.samplings.sph_gaussian(sh_order=1)
    coords = coords[coords.z > 0]

    # Call the function
    sp.brdf.create_from_scattering(
        file_path, coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
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


def test_create_from_scattering_1(tmpdir):
    # Prepare test data
    scattering_data = np.ones((3, ))
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)
    coords = pf.samplings.sph_gaussian(sh_order=1)
    coords = coords[coords.z > 0]

    # Call the function
    sp.brdf.create_from_scattering(
        file_path, coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        )
    data, s, r = pf.io.read_sofa(file_path)

    # Assert the expected outcome
    assert data.freq.shape == (4, 4, 3)
    assert data.freq.shape == (s.csize, r.csize, 3)
    # test
    npt.assert_almost_equal(data.freq, 1 / np.pi)


def test_create_from_scattering_0(tmpdir):
    # Prepare test data
    scattering_data = np.zeros((3, ))
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)
    coords = pf.samplings.sph_gaussian(sh_order=1)
    coords = coords[coords.z > 0]

    # Call the function
    sp.brdf.create_from_scattering(
        file_path, coords, coords,
        pf.FrequencyData(scattering_data, frequency_data),
        )
    data, s, r = pf.io.read_sofa(file_path)

    # Assert the expected outcome
    assert data.freq.shape == (4, 4, 3)
    assert data.freq.shape == (s.csize, r.csize, 3)
    # test
    npt.assert_almost_equal(data.freq[0, 2], 1.10265779)
    npt.assert_almost_equal(data.freq[0, not 2], 0)


def test_create_from_scattering_0_3(tmpdir):
    # Prepare test data
    scattering_data = 0.3
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)
    coords = pf.samplings.sph_gaussian(sh_order=1)
    coords = coords[coords.z > 0]

    # Call the function
    sp.brdf.create_from_scattering(
        file_path, coords, coords,
        pf.FrequencyData(scattering_data + np.zeros((3, )), frequency_data),
        )
    data, s, r = pf.io.read_sofa(file_path)

    # Assert the expected outcome
    assert data.freq.shape == (4, 4, 3)
    assert data.freq.shape == (s.csize, r.csize, 3)
    desired_scat = scattering_data / np.pi
    desired_spec = 0.86735342-desired_scat

    # test
    npt.assert_almost_equal(data.freq[0, 2], desired_spec + desired_scat)
    npt.assert_almost_equal(data.freq[0, not 2], desired_scat)


def test_create_from_scattering_0_3_with_absorption(tmpdir):
    # Prepare test data
    scattering_data = 0.3
    absorption_data = 0.3
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)
    coords = pf.samplings.sph_gaussian(sh_order=1)
    coords = coords[coords.z > 0]

    # Call the function
    sp.brdf.create_from_scattering(
        file_path, coords, coords,
        pf.FrequencyData(scattering_data + np.zeros((3, )), frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
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


def test_create_from_scattering_with_invalid_data(tmpdir):
    # Prepare test data
    scattering_data = pf.FrequencyData(1, 100)
    coords = pf.samplings.sph_gaussian(sh_order=1)

    # Call the function and expect it to raise an error
    with pytest.raises(
            TypeError,
            match="scattering_coefficient must be a pf.FrequencyData object"):
        sp.brdf.create_from_scattering(tmpdir, coords, coords, 'invalid')
    # Call the function and expect it to raise an error
    with pytest.raises(
            TypeError,
            match="source_directions must be a pf.Coordinates object"):
        sp.brdf.create_from_scattering(
            tmpdir, 'coords', coords, scattering_data)
    with pytest.raises(
            TypeError,
            match="receiver_directions must be a pf.Coordinates object"):
        sp.brdf.create_from_scattering(
            tmpdir, coords, 'coords', scattering_data)


@pytest.mark.parametrize('scattering_data', [0, 0.3, 1])
@pytest.mark.parametrize('absorption_data', [0, 0.3, 1])
@pytest.mark.parametrize('sh_order', [5, 11])
def test_energy_conservation(
        tmpdir, scattering_data, absorption_data, sh_order):
    # Prepare test data
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]

    # Call the function
    sp.brdf.create_from_scattering(
        file_path, coords, coords,
        pf.FrequencyData(scattering_data + np.zeros((3, )), frequency_data),
        pf.FrequencyData(absorption_data + np.zeros((3, )), frequency_data),
        )
    sofa = sf.read_sofa(file_path)
    data, s, r = pf.io.convert_sofa(sofa)

    # test energy conservation
    integration_weights = sofa.ReceiverWeights
    integration_weights *= 2*np.pi/np.sum(integration_weights)
    integration_weights = integration_weights*np.cos(r.colatitude)
    integration_weights = integration_weights[..., np.newaxis]
    for i_source in range(s.csize):
        energy = np.sum(data.freq[i_source]*integration_weights, axis=0)
        npt.assert_almost_equal(energy, 1 - absorption_data, decimal=1)


@pytest.mark.parametrize('scattering_data', [0, 0.3, 1])
@pytest.mark.parametrize('absorption_data', [0, 0.3, 1])
@pytest.mark.parametrize('sh_order', [5, 11])
def test_reciprocal_principle(
        tmpdir, scattering_data, absorption_data, sh_order):
    # Prepare test data
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)
    coords = pf.samplings.sph_gaussian(sh_order=sh_order)
    coords = coords[coords.z > 0]

    # Call the function
    sp.brdf.create_from_scattering(
        file_path, coords, coords,
        pf.FrequencyData(scattering_data + np.zeros((3, )), frequency_data),
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

