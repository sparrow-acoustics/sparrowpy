import sparapy as sp


def read_geometry(path):
    """Read geometry from a file.

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    geometry : sparapy.geometry.Polygons
        The polygon.

    """
    # use trimesh or sth else to read obj, stl, fbx files
    return sp.geometry.Polygon()

