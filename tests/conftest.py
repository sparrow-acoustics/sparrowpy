import sparapy as sp
import pytest

@pytest.fixture()
def sample_walls():
    """Return a list of 6 walls, which form a cube."""
    return sp.testing.shoebox_room_stub(1, 1, 1)
