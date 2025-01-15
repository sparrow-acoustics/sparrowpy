"""Stub utilities for testing."""
import sparrowpy as sp


def shoebox_room_stub(length_x, length_y, length_z):
    """Create a shoebox room with the given dimensions.

    Parameters
    ----------
    length_x : float
        Length of the room in x-direction in meters.
    length_y : float
        Length of the room in y-direction in meters
    length_z : float
        Length of the room in z-direction in meters

    Returns
    -------
    room : list[geo.Polygon]
        List of the walls of the room.

    """
    return [
        sp.geometry.Polygon(
            [[0, 0, 0], [length_x, 0, 0],
            [length_x, 0, length_z], [0, 0, length_z]],
            [1, 0, 0], [0, 1, 0]),
        sp.geometry.Polygon(
            [[0, length_y, 0], [length_x, length_y, 0],
            [length_x, length_y, length_z], [0, length_y, length_z]],
            [1, 0, 0], [0, -1, 0]),
        sp.geometry.Polygon(
            [[0, 0, 0], [length_x, 0, 0],
            [length_x, length_y, 0], [0, length_y, 0]],
            [1, 0, 0], [0, 0, 1]),
        sp.geometry.Polygon(
            [[0, 0, length_z], [length_x, 0, length_z],
            [length_x, length_y, length_z], [0, length_y, length_z]],
            [1, 0, 0], [0, 0, -1]),
        sp.geometry.Polygon(
            [[0, 0, 0], [0, 0, length_z],
            [0, length_y, length_z], [0, length_y, 0]],
            [0, 0, 1], [1, 0, 0]),
        sp.geometry.Polygon(
            [[length_x, 0, 0], [length_x, 0, length_z],
            [length_x, length_y, length_z], [length_x, length_y, 0]],
            [0, 0, 1], [-1, 0, 0]),
        ]
