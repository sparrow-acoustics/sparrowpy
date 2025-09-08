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


@pytest.fixture
def sample_walls():
    """Return a list of 6 walls, which form a cube."""
    return sp.testing.shoebox_room_stub(1, 1, 1)


@pytest.fixture
def sofa_data_diffuse():
    """Return a diffuse brdf set for five octave bands."""
    gaussian = pf.samplings.sph_gaussian(sh_order=1)
    gaussian = gaussian[gaussian.z>0]
    sources = gaussian.copy()
    receivers = gaussian.copy()
    frequencies = np.array([125, 250, 500, 1000])
    brdf = sp.brdf.create_from_scattering(
        sources, receivers,
        pf.FrequencyData(np.ones_like(frequencies), frequencies))
    return (brdf, sources, receivers)


@pytest.fixture
def sofa_data_diffuse_full_third_octave():
    """Return a diffuse brdf set for the full third octave band."""
    gaussian = pf.samplings.sph_gaussian(sh_order=1)
    gaussian = gaussian[gaussian.z>0]
    sources = gaussian.copy()
    receivers = gaussian.copy()
    frequencies = np.array(
        [20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0,
         250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
         2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0,
         12500.0, 16000.0, 20000.0])
    brdf = sp.brdf.create_from_scattering(
        sources, receivers,
        pf.FrequencyData(np.ones_like(frequencies), frequencies))
    return (brdf, sources, receivers)


@pytest.fixture
def basicscene():
    scene = {}
    scene["patch_size"] = .2
    scene["ir_length_s"] = 1.
    scene["sampling_rate"] = 100
    scene["max_order_k"] = 10
    scene["speed_of_sound"] = 343
    scene["absorption"] = 0.1
    scene["walls"] = sp.testing.shoebox_room_stub(1, 1, 1)

    return scene
