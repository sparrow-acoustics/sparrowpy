"""functions to generate energy histogram of "infinite" plane scenario."""
import sparapy as sp
import numpy as np

def infinite_plane(point:np.ndarray, ratio=15.):
    """Create an "infinite" xy plane relative to eval point position.

    Parameters
    ----------
    point: np.array(3,)
        evaluation position in 3D space (receiver)
    ratio: float
        ratio between plane dimensions and
        distance between receiver and plane.

    Returns
    -------
    plane: sparapy.geometry.Polygon
        plane surface in format compatible with sparapy.

    """
    length = ratio*point[2] # dimensions of plane's side

    return [
            sp.geometry.Polygon([ [length/2, length/2, 0],
                                [-length/2, length/2, 0],
                                [-length/2, -length/2, 0],
                                [length/2, -length/2, 0]
                                ])
    ]

def get_histogram():
    """Generate histogram of infinite plane scenario."""
    source = np.array([0.,0.,1.])
    receiver = np.array([0.,0.,1.])

    plane = infinite_plane(point=receiver)


