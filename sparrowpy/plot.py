"""Module for plotting functions in SparrowPy.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def polygons_3d(edge_points, energy, colorbar=True):
    """Show the energy of the polygons in 3D.

    The polygons can represent patches or walls and are defined by the
    edge points.

    Parameters
    ----------
    edge_points : np.ndarray, list
        The points in cartesian coordinates of the polygons
        of shape (#polygons, #points, 3).
    energy : np.ndarray
        Energy for each polygon of shape (#polygons,).
    colorbar : bool, optional
        Whether to show a colorbar or not. Default is ``True``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    """
    # test input types
    try:
        edge_points = np.array(edge_points, dtype=float)
    except Exception as e:
        raise ValueError(
            "edge_points must be convertible to a numpy array of floats.",
            ) from e
    try:
        energy = np.array(energy, dtype=float)
    except Exception as e:
        raise ValueError(
            "energy must be convertible to a numpy array of floats.") from e

    # test input properties
    if energy.ndim != 1:
        raise ValueError("energy must be a 1D array.")
    if np.array(edge_points).ndim != 3:
        raise ValueError(
            "edge_points must be of shape (#polygons, #points, 3).")
    if edge_points.shape[0] != energy.shape[0]:
        raise ValueError(
            "The number of polygons in edge_points must match the number of "
            "energy values.")
    if edge_points.shape[-1] != 3:
        raise ValueError(
            "The last dimension of edge_points must be 3 "
            "(cartesian coordinates).")

    ax = plt.axes(projection='3d')
    cmap = cm.viridis
    v_min, v_max = np.min(energy), np.max(energy)
    if v_min == v_max:
        v_min *= 0.9
        v_max *= 1.1
    norm = mpl.colors.Normalize(v_min, v_max)
    mappable =  mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(energy)

    for i in range(edge_points.shape[0]):
        poly = Poly3DCollection(
            edge_points[i][np.newaxis, ...],
            color=mappable.to_rgba(energy[i]),
            )
        ax.add_collection3d(poly)

    if colorbar:
        plt.colorbar(
            mappable=mappable,
            ax=ax,
            )

    return ax
