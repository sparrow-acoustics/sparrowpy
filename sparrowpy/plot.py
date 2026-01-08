"""Module for plotting functions in SparrowPy."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def polygons_3d(edge_points, energy, ax=None, v_min=None, v_max=None,
    colorbar=True, **kwargs):
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

    # create axis if not provided
    if ax is None:
        ax = plt.axes(projection="3d")

    if "3d" not in ax.name:
        raise ValueError("The projection of the axis needs to be '3d'.")

    cmap = cm.get_cmap("viridis")
    if v_min is None:
        v_min = np.min(energy)
    if v_max is None:
        v_max = np.max(energy)
    if v_min == v_max:
        v_min *= 0.9
        v_max *= 1.1
    norm = colors.Normalize(v_min, v_max)
    mappable =  cm.ScalarMappable(cmap=cmap, norm=norm)
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

    return ax


def animate_polygons_3d(
    edge_points,
    energy_over_time,
    ax=None,
    v_min=None,
    v_max=None,
    colorbar=True,
    animation_fps=50,
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

    cmap = cm.get_cmap("viridis")
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
        interval=1000/animation_fps,
        repeat=True,
        blit=False,
    )

    # keep reference to animation on the axis to prevent garbage collection
    try:
        ax._animation = anim
    except Exception:
        pass

    return anim, ax
