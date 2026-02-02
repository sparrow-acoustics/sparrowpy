"""Module for plotting functions in SparrowPy."""

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pyfar as pf
import scipy.spatial as sspat
from matplotlib import cm, colormaps, colors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
from spharpy import samplings

from sparrowpy import geometry


def polygons_3d(
    edge_points,
    energy,
    v_min=None,
    v_max=None,
    colorbar=True,
    brdf_points=None,
    brdf_data=None,
    scale_brdf=0.5,
    ax=None,
    **kwargs,
):
    """Show the energy of the polygons in 3D.

    The polygons can represent patches or walls and are defined by the
    edge points.
    For the BRDF visualization, the sampling points AND the BRDF data
    for each polygon have to be provided.

    Parameters
    ----------
    edge_points : np.ndarray, list
        The points in cartesian coordinates of the polygons
        of shape (#polygons, #points, 3 or 4).
    energy : np.ndarray
        Energy for each polygon of shape (#polygons,).
    v_min : float, optional
        Minimum value for color scaling. If ``None``, the minimum of the energy
        array is used. Default is ``None``.
    v_max : float, optional
        Maximum value for color scaling. If ``None``, the maximum of the energy
        array is used. Default is ``None``.
    colorbar : bool, optional
        Whether to show a colorbar or not. Default is ``True``.
    brdf_points : :py:class:`~pyfar.classes.coordinates.Coordinates`, optional
        The sampling (receiver) points of the brdf_data (#brdf_outgoing_angles)
        Assumes the same point distribution for every given brdf.
        If ``None``, no BRDF is visualized. Default is ``None``.
    brdf_data : :py:class:`~pyfar.classes.audio.FrequencyData`, optional
        Plot BRDF for each polygon of shape (#polygons, #brdf_outgoing_angles).
        This implementation requires a specific frequency band and incoming
        angle before calling the function!
        If ``None``, no BRDF is visualized. Default is ``None``.
    scale_brdf : float, optional
        Scaling factor for the BRDF balloon plot size. Default is ``0.5``.
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
            "energy must be convertible to a numpy array of floats.",
        ) from e

    # test input properties
    if energy.ndim != 1:
        raise ValueError("energy must be a 1D array.")
    if np.array(edge_points).ndim != 3:
        raise ValueError(
            "edge_points must be of shape (#polygons, #points, 3 or 4).",
        )
    if edge_points.shape[0] != energy.shape[0]:
        raise ValueError(
            "The number of polygons in edge_points must match the number of "
            "energy values.",
        )
    if edge_points.shape[-1] != 3:
        raise ValueError(
            "The last dimension of edge_points must be 3 "
            "(cartesian coordinates).",
        )

    # create axis if not provided
    if ax is None:
        ax = plt.axes(projection="3d")

    if "3d" not in ax.name:
        raise ValueError("The projection of the axis needs to be '3d'.")

    cmap = colormaps.get_cmap("viridis")
    if v_min is None:
        v_min = np.min(energy)
    if v_max is None:
        v_max = np.max(energy)
    if v_min == v_max:
        v_min *= 0.9
        v_max *= 1.1
    norm = colors.Normalize(v_min, v_max)
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(energy)

    for i in range(edge_points.shape[0]):
        poly = Poly3DCollection(
            edge_points[i][np.newaxis, ...],
            color=mappable.to_rgba(energy[i]),
            **kwargs,
        )
        ax.add_collection3d(poly)

    if colorbar:
        plt.colorbar(mappable=mappable, ax=ax)

    if brdf_data is None:
        if brdf_points is None:
            return ax
        raise ValueError(
            "brdf_data must be provided if brdf_points are provided.",
        )

    # brdf_data is given
    # test brdf properties
    if isinstance(brdf_data, pf.FrequencyData):
        if brdf_data.frequencies.shape[0] != 1:
            raise ValueError(
                "brdf_data must have only one frequency - otherwise "
                "unclear what data should be plotted.",
            )
        if np.iscomplex(brdf_data.freq).any():
            raise ValueError(
                "brdf_data must be real-valued for plotting.",
            )
        brdf_array = np.real(brdf_data.freq)  # remove possible +0j
    else:
        raise ValueError("brdf_data must be of type pyfar.FrequencyData.")
    if brdf_array.ndim >= 3:
        if brdf_array.ndim == 3 and brdf_array.shape[-1] == 1:
            brdf_array = brdf_array[..., 0]
        else:
            raise ValueError(
                "brdf_data must be reducible to shape "
                "(#polygons, #brdf_outgoing_angles).",
            )
    if brdf_array.shape[0] != energy.shape[0]:
        raise ValueError(
            "The number of polygons in brdf must match the number of "
            "polygons with energy values.",
        )
    if brdf_points is None:
        raise ValueError(
            "brdf_points must be provided if brdf_data is provided.",
        )
    if brdf_points.cartesian.shape[0] != brdf_array.shape[1]:
        raise ValueError(
            "brdf_points and brdf_data do not contain the same number of "
            "points as brdf_outgoing_angles.",
        )

    # change polygons from Poly3DCollection to be see through for better
    # visibility of brdf plots when using Matplotlib for 3D rendering
    try:
        for coll in ax.collections:
            if isinstance(coll, Poly3DCollection):
                coll.set_alpha(0.7)
    except Exception:
        pass

    # plot brdf
    normal_vectors = geometry._calculate_normals(edge_points)
    for idx, points in enumerate(edge_points):  # for each polygon
        middle_point = np.sum(points, axis=0) / points.shape[0]

        scale_plot = (
            scale_brdf
            * np.sqrt(geometry._calculate_area(points[np.newaxis, ...]))
            / 2
        )
        balloon(
            brdf_points,
            brdf_array[idx, ...],
            translate=middle_point,
            rotate_normal=normal_vectors[idx],
            scale=scale_plot,
            colorbar=False,
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
    """
    # equal to coordinates = convert_coordinates(coordinates):
    if type(coordinates) is pf.Coordinates:
        if coordinates.sh_order is None:
            coordinates = samplings.Coordinates.from_pyfar(coordinates)
        else:
            coordinates = samplings.SamplingSphere.from_pyfar(coordinates)

    # similar to tri, xyz = _triangulation_sphere(sampling = coordinates, data)
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

    if scale is not None:
        xyz *= scale

    xyz = RotM @ xyz + translate[..., None]

    x, y, z = xyz
    tri = mtri.Triangulation(x, y, triangles=hull.simplices)

    # create axis if not provided
    if ax is None:
        ax = plt.axes(projection="3d")

    if "3d" not in ax.name:
        raise ValueError("The projection of the axis needs to be '3d'.")

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

    return plot


def animate_polygons_3d(
    edge_points,
    energy_over_time,
    ax=None,
    v_min=None,
    v_max=None,
    colorbar=True,
    animation_fps=50,
    show_current_sample_text=True,
    **kwargs,
):
    """Animate the energy of polygons over time in 3D.

    The polygons can represent patches or walls and are defined by the
    edge points.

    Parameters
    ----------
    edge_points : np.ndarray, list
        The points in cartesian coordinates of the polygons
        of shape (#polygons, #points, 3).
    energy_over_time : np.ndarray
        Time dependent energy for each polygon of shape (#polygons, #samples).
    ax : matplotlib.axis, None, optional
        The matplotlib axis object used for plotting. By default ``None``,
        which will create a new axis object.
    v_min : float, optional
        Minimum value for color scaling. If ``None``, the minimum of the energy
        array is used. Default is ``None``.
    v_max : float, optional
        Maximum value for color scaling. If ``None``, the maximum of the energy
        array is used. Default is ``None``.
    colorbar : bool, optional
        Whether to show a colorbar or not. Default is ``True``.
    animation_fps : int, optional
        Number of frames per second for the plot animation. Default is ``50``.
    show_current_sample_text : bool, optional
        Whether to show the current sample index as text in the plot title.
        Default is ``True``.
    **kwargs : optional
        Additional keyword arguments passed to Poly3DCollection.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The created animation object.
    ax : matplotlib.axes.Axes
        The axes used for plotting.

    """
    # test input types
    try:
        edge_points = np.array(edge_points, dtype=float)
    except Exception as e:
        raise ValueError(
            "edge_points must be convertible to a numpy array of floats.",
        ) from e
    try:
        energy_over_time = np.array(energy_over_time, dtype=float)
    except Exception as e:
        raise ValueError(
            "energy must be convertible to a numpy array of floats.",
        ) from e

    # test input properties
    if energy_over_time.ndim != 2:
        raise ValueError(
            "energy_over_time must be of shape (#polygons, #samples).",
        )
    n_polys, n_samples = energy_over_time.shape
    if (
        edge_points.ndim != 3
        or edge_points.shape[0] != n_polys
        or edge_points.shape[-1] != 3
    ):
        raise ValueError(
            "edge_points must have shape (#polygons, #points, 3) matching "
            "energy_over_time.",
        )

    # create axis if not provided
    if ax is None:
        ax = plt.axes(projection="3d")
    if "3d" not in ax.name:
        raise ValueError("The projection of the axis needs to be '3d'")

    cmap = colormaps.get_cmap("viridis")
    if v_min is None:
        v_min = np.min(energy_over_time)
    if v_max is None:
        v_max = np.max(energy_over_time)
    if v_min == v_max:
        v_min *= 0.9
        v_max *= 1.1
    norm = colors.Normalize(v_min, v_max)
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(energy_over_time)

    # create polygon objects once
    polys = []
    for i in range(n_polys):
        color = mappable.to_rgba(energy_over_time[i, 0])
        poly = Poly3DCollection(
            edge_points[i][np.newaxis, ...],
            facecolor=color,
            **kwargs,
        )
        ax.add_collection3d(poly)
        polys.append(poly)

    if colorbar:
        plt.colorbar(mappable=mappable, ax=ax)

    def _update(frame):
        energies_in_frame = energy_over_time[:, frame]
        if show_current_sample_text:
            ax.set_title(f"Sample = {frame}")
        colors = mappable.to_rgba(energies_in_frame)
        for poly, c in zip(polys, colors, strict=False):
            poly.set_facecolor(c)
        # update mappable array
        try:
            mappable.set_array(energies_in_frame)
        except Exception:
            pass
        return polys

    anim = FuncAnimation(
        ax.get_figure(),
        _update,
        frames=range(n_samples),
        interval=1000 / animation_fps,
        repeat=True,
        blit=False,
    )

    # keep reference to animation on the axis to prevent garbage collection
    try:
        ax._animation = anim
    except Exception:
        pass

    return anim, ax
