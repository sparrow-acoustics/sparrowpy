"""Module for plotting functions in SparrowPy.
"""

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pyfar as pf
from matplotlib import cm
import matplotlib as mpl
import scipy.spatial as sspat
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
from spharpy import samplings


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


def balloon(
    coordinates,
    data,
    translate=None,
    rotate_normal=None,
    scale=None,
    cmap=None,
    colorbar=False,
    ax=None,
    **kwargs,
):
    """Adaptation to the spharpy balloon plot function to include translation,
    rotation and scaling.
    Plot data (e.g. BRDFs) on a sphere defined by the coordinates.
    The magnitude information is mapped onto the radius of the sphere.
    The colormap represents the magnitude of the data array.

    Note
    ----
    Be aware that other objects (e.g. polygons_3d with Poly3DCollection) might
    cover (parts of) the balloon plot depending on the viewing angle and
    the alpha value of the other objects because of matplotlib's inaccurate
    collection depth buffer.

    Parameters
    ----------
    coordinates : :class:`spharpy.samplings.Coordinates`
        Coordinates defining a sphere.
    data : ndarray, double
        Data for each angle, must have size corresponding to the number of
        points given in coordinates.
    translate : np.ndarray, optional
        Translation vector of shape (3,) to move the balloon plot to a
        specific position in 3D space e.g. the middle of a polygon.
        Default is ``None`` (point of origin).
    rotate_normal : np.ndarray, optional
        Vector of shape (3,) to rotate the balloon to this orientation e.g.
        the normal vector of a polygon. Default is ``None``, which creates a
        balloon plot on the xy-plane equal to rotate_normal of [0,0,1].
    scale : float, optional
        Manual scaling factor for the balloon size. It is advised to use a
        scaling based on the polygon area (with geometry._calculate_area()).
        Default is ``None``.
    cmap : matplotlib colormap, optional
        Colormap for the plot, see matplotlib.cm
    colorbar : bool, optional
        Whether to show a colorbar or not. Default is ``False``.
    ax : matplotlib.axis, optional
        The matplotlib axis object used for plotting. By default ``None``,
        which will create a new axis object.
    **kwargs : optional
        Additional keyword arguments passed to Poly3DCollection.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    """
    # check and convert coordinates
    if type(coordinates) is pf.Coordinates:
        if coordinates.sh_order is None:
            coordinates = samplings.Coordinates.from_pyfar(coordinates)
        else:
            coordinates = samplings.SamplingSphere.from_pyfar(coordinates)

    x, y, z = samplings.sph2cart(
        np.abs(data),
        coordinates.elevation,
        coordinates.azimuth,
    )
    xyz = np.vstack((x, y, z))
    hull = sspat.ConvexHull(
        np.asarray(
            samplings.sph2cart(
                np.ones(coordinates.n_points),
                coordinates.elevation,
                coordinates.azimuth,
            ),
        ).T,
    )

    # calculate rotation (approach equal to a transformation matrix)
    if translate is None:
        translate = np.array([0.0, 0.0, 0.0])
    if rotate_normal is not None:
        rotate_normal = rotate_normal / np.linalg.norm(rotate_normal)
        axis = np.cross(rotate_normal, [0, 0, 1])
        dot_of_normal_and_world = np.dot(rotate_normal, [0, 0, 1])
        if np.linalg.norm(axis) < 1e-8:
            # parallel or anti-parallel (that needs 180° rot around x axis)
            if dot_of_normal_and_world > 0:
                RotM = np.eye(3)
            else:
                rotvec = np.array([1.0, 0.0, 0.0]) * np.pi
                RotM = Rotation.from_rotvec(rotvec).as_matrix()
        else:
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.clip(dot_of_normal_and_world, -1, 1))
            RotM = Rotation.from_rotvec(axis * angle).as_matrix()
    else:
        # no rotation given
        RotM = np.eye(3)

    # apply scaling
    if scale is not None:
        xyz *= scale

    # apply rotation and translation
    xyz = RotM @ xyz + translate[..., None]

    x, y, z = xyz
    tri = mtri.Triangulation(x, y, triangles=hull.simplices)

    # create axis if not provided
    if ax is None:
        ax = plt.axes(projection="3d")

    if "3d" not in ax.name:
        raise ValueError("The projection of the axis needs to be '3d'.")

    # create cmap if not provided
    if cmap is None:
        cmap = plt.get_cmap("viridis")

    plot = ax.plot_trisurf(
        tri,
        z,
        cmap=cmap,
        antialiased=True,
        vmin=np.min(data),
        vmax=np.max(data),
        **kwargs,
    )

    if colorbar:
        plt.gcf().colorbar(plot, ax=ax, label="Amplitude")

    return ax
