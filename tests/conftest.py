"""Fixtures for the tests in the tests/ directory."""
import sparrowpy as sp
import pyfar as pf
import numpy as np
import pytest


@pytest.fixture
def brdf_s_0(tmp_path_factory):
    """Temporary small SOFA file.

    To be used when data needs to be read from a SOFA file for testing.
    Contains custom data for "Data_IR", "GLOBAL_RoomType" and
    "Data_SamplingRate_Units".

    Returns
    -------
    filename : SOFA file
        Filename of temporary SOFA file

    """
    filename = tmp_path_factory.mktemp("data") / "brdf_tmp.sofa"
    coords = pf.samplings.sph_gaussian(sh_order=1)
    coords = coords[coords.z > 0]
    sp.brdf.create_from_scattering(
        coords, coords, pf.FrequencyData(0, [100]), file_path=filename)
    return filename


@pytest.fixture
def brdf_s_1(tmp_path_factory):
    """Temporary small SOFA file.

    To be used when data needs to be read from a SOFA file for testing.
    Contains custom data for "Data_IR", "GLOBAL_RoomType" and
    "Data_SamplingRate_Units".

    Returns
    -------
    filename : SOFA file
        Filename of temporary SOFA file

    """
    filename = tmp_path_factory.mktemp("data") / "brdf_tmp.sofa"
    coords = pf.samplings.sph_gaussian(sh_order=1)
    coords = coords[coords.z > 0]
    sp.brdf.create_from_scattering(
        coords, coords, pf.FrequencyData(1, [100]), file_path=filename)
    return filename


@pytest.fixture()
def sample_walls():
    """Return a list of 6 walls, which form a cube."""
    return sp.testing.shoebox_room_stub(1, 1, 1)


@pytest.fixture()
def sofa_data_diffuse():
    """Return a list of 6 walls, which form a cube."""
    gaussian = pf.samplings.sph_gaussian(sh_order=1)
    gaussian = gaussian[gaussian.z>0]
    sources = gaussian.copy()
    receivers = gaussian.copy()
    frequencies = pf.dsp.filter.fractional_octave_frequencies(
        1, (100, 1000))[0]
    data = np.ones((sources.csize, receivers.csize, frequencies.size))
    return (pf.FrequencyData(data, frequencies), sources, receivers)

@pytest.fixture()
def basicscene():
    scene=dict()
    scene["patch_size"] = .2
    scene["ir_length_s"] = 1.
    scene["sampling_rate"] = 100
    scene["max_order_k"] = 10
    scene["speed_of_sound"] = 343
    scene["absorption"] = 0.1
    scene["walls"] = sp.testing.shoebox_room_stub(1, 1, 1)

    return scene