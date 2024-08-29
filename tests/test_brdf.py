
"""Test the radiosity module with directional Patches."""
import os

import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
import sparapy as sp


create_reference_files = False


def test_create_from_scattering_with_valid_data(tmpdir):
    # Prepare test data
    scattering_data = [0, 0.2, 1]
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)

    # Call the function
    sp.brdf.create_from_scattering(
        file_path, scattering_data, 1, frequency_data)
    data, s, r = pf.io.read_sofa(file_path)

    # Assert the expected outcome
    assert data.freq.shape == (4, 4, 3)
    assert data.freq.shape == (s.csize, r.csize, 3)
    # test
    for i in range(4):
        for j in range(4):
            npt.assert_almost_equal(data.freq[i, j], data.freq[j, i])


def test_create_from_scattering_1(tmpdir):
    # Prepare test data
    scattering_data = 1
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)

    # Call the function
    sp.brdf.create_from_scattering(
        file_path, scattering_data, 1, frequency_data)
    data, s, r = pf.io.read_sofa(file_path)

    # Assert the expected outcome
    assert data.freq.shape == (4, 4, 3)
    assert data.freq.shape == (s.csize, r.csize, 3)
    # test
    npt.assert_almost_equal(data.freq, 1/np.pi)


def test_create_from_scattering_0(tmpdir):
    # Prepare test data
    scattering_data = 0
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)

    # Call the function
    sp.brdf.create_from_scattering(
        file_path, scattering_data, 1, frequency_data)
    data, s, r = pf.io.read_sofa(file_path)

    # Assert the expected outcome
    assert data.freq.shape == (4, 4, 3)
    assert data.freq.shape == (s.csize, r.csize, 3)
    # test
    npt.assert_almost_equal(data.freq[0, 2], 1/np.cos(s.elevation[0]))
    npt.assert_almost_equal(data.freq[0, not 2], 0)


def test_create_from_scattering_0_3(tmpdir):
    # Prepare test data
    scattering_data = 0.3
    frequency_data = [100, 200, 400]
    file_path = os.path.join(tmpdir, "test_brdf.sofa",)

    # Call the function
    sp.brdf.create_from_scattering(
        file_path, scattering_data, 1, frequency_data)
    data, s, r = pf.io.read_sofa(file_path)

    # Assert the expected outcome
    assert data.freq.shape == (4, 4, 3)
    assert data.freq.shape == (s.csize, r.csize, 3)
    desired_scat = scattering_data/np.pi
    desired_spec = (1 - scattering_data)/np.cos(s.elevation)
    # test
    npt.assert_almost_equal(data.freq[0, 2], desired_spec[0]+desired_scat)
    npt.assert_almost_equal(data.freq[0, not 2], desired_scat)


def test_create_from_scattering_with_invalid_data(tmpdir):
    # Prepare test data
    scattering_data = "invalid_data"

    # Call the function and expect it to raise an error
    with pytest.raises(TypeError):
        sp.brdf.create_from_scattering(tmpdir, scattering_data)

    with pytest.raises(
            TypeError, match='Frequencies must be monotonously increasing.'):
        sp.brdf.create_from_scattering(tmpdir, [0, 1], 1, [100])
