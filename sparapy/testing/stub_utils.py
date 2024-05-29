import sparapy.geometry as geo


def room_stub(length_x, length_y, length_z):
    return [
    geo.Polygon(
        [[0, 0, 0], [length_x, 0, 0],
         [length_x, 0, length_z], [0, 0, length_z]],
        [1, 0, 0], [0, 1, 0]),
    geo.Polygon(
        [[0, length_y, 0], [length_x, length_y, 0],
         [length_x, length_y, length_z], [0, length_y, length_z]],
        [1, 0, 0], [0, -1, 0]),
    geo.Polygon(
        [[0, 0, 0], [length_x, 0, 0],
         [length_x, length_y, 0], [0, length_y, 0]],
        [1, 0, 0], [0, 0, 1]),
    geo.Polygon(
        [[0, 0, length_z], [length_x, 0, length_z],
         [length_x, length_y, length_z], [0, length_y, length_z]],
        [1, 0, 0], [0, 0, -1]),
    geo.Polygon(
        [[0, 0, 0], [0, 0, length_z],
         [0, length_y, length_z], [0, length_y, 0]],
        [0, 0, 1], [1, 0, 0]),
    geo.Polygon(
        [[length_x, 0, 0], [length_x, 0, length_z],
         [length_x, length_y, length_z], [length_x, length_y, 0]],
        [0, 0, 1], [-1, 0, 0]),
]
