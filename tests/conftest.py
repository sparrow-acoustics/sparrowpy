"""Fixtures for the tests in the tests/ directory."""
import sparapy as sp
import pyfar as pf
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
        filename, coords, coords, pf.FrequencyData(0, [100]))
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
        filename, coords, coords, pf.FrequencyData(1, [100]))
    return filename


@pytest.fixture()
def sample_walls():
    """Return a list of 6 walls, which form a cube."""
    return sp.testing.shoebox_room_stub(1, 1, 1)
