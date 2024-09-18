
import matplotlib.pyplot as plt
import numpy as np
import pyfar as pf
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mplstereonet  # noqa: F401


def brdf_3d(
        receivers, data, source_pos=None, ax=None,
        grid_color='black'):
    """Plot the BRDF data in a 3d space.

    Parameters
    ----------
    receivers : pf.Coordinates
        _description_
    data : np.ndarray
        The data to plot, same shape as the cshape of the receivers.
    source_pos : pf.Coordinates, optional
        A direction which is highlighted in red, if not None, by default None
    ax : matplotlib.axes.Axes, optional
        The axes to plot on., by default None
    grid_color : str, optional
        color of the grid of the sampling, by default 'black'. It is calculated
        by the Voronoi diagram.

    Returns
    -------
    _type_
    ax : matplotlib.axes.Axes
        The axes of the plot.

    """
    if ax is None:
        ax = plt.axes(projection='3d')
    receivers_cp = receivers.copy()
    receivers_cp.z = -receivers_cp.z
    receivers_cp = receivers_cp[receivers_cp.z<-1e-3]
    rec = pf.Coordinates(
        np.concatenate((receivers.x, receivers_cp.x)),
        np.concatenate((receivers.y, receivers_cp.y)),
        np.concatenate((receivers.z, receivers_cp.z)),
    )

    sv = pf.samplings.SphericalVoronoi(
        rec, center=[0, 0, 0], round_decimals=12)
    sv.sort_vertices_of_regions()

    vertices = sv.vertices
    vertices_coords = pf.Coordinates(
        vertices[:,0], vertices[:,1], vertices[:,2])
    cmap = cm.viridis
    if np.max(data) > 1:
        zi_min = np.min(data)
        data = (data-zi_min)/np.max((data-zi_min))
    for i, reg in enumerate(sv.regions):
        reg_cp = reg.copy()
        reg_cp.append(reg_cp[0])
        coords = vertices_coords[reg_cp].copy()
        if i >= data.shape[0]:
            continue
        zi = data[i]
        color = zi
        coords.radius = zi*1.5
        xi = coords.x
        yi = coords.y
        zii = coords.z
        if zi > 0:
            if grid_color is not None:
                cax = ax.plot(
                    xi, yi, zii, color=grid_color)
                plo = np.array([xi, yi, zii]).T
                ax.add_collection3d(Poly3DCollection(
                    [plo], color=cmap(color), alpha=0.5))

            for i in range(xi.size-1):
                ax.plot(
                    [0, xi[i]], [0, yi[i]], [0, zii[i]], color=grid_color)
                plo = np.array(
                    [[0, xi[i], xi[i+1]],
                    [0, yi[i], yi[i+1]],
                    [0, zii[i], zii[i+1]]]).T
                ax.add_collection3d(Poly3DCollection(
                    [plo], color=cmap(color), alpha=0.5))

    if source_pos is not None:
        ax.plot(
            [0, source_pos.x[0]],
            [0, source_pos.y[0]],
            [0, source_pos.z[0]],
            color='r', linestyle='-')
    data_max = np.max(data)
    ax.grid(True)
    ax.set_xlim((-data_max, data_max))
    ax.set_ylim((-data_max, data_max))
    ax.set_zlim((0, data_max))
    ax.set_aspect('equal')
    return ax


def brdf_polar(
        receivers, data_in, source_pos=None, ax=None,
        projection='polar', plot_empty=False,
        factor_cmap=1, grid_color='black'):

    if ax is None:
        ax = plt.axes(projection=projection)
    receivers_cp = receivers.copy()
    receivers_cp.z = -receivers_cp.z
    receivers_cp = receivers_cp[receivers_cp.z<-1e-3]
    rec = pf.Coordinates(
        np.concatenate((receivers.x, receivers_cp.x)),
        np.concatenate((receivers.y, receivers_cp.y)),
        np.concatenate((receivers.z, receivers_cp.z)),
    )

    sv = pf.samplings.SphericalVoronoi(
        rec, center=[0, 0, 0], round_decimals=12)
    sv.sort_vertices_of_regions()

    vertices = sv.vertices
    vertices_coords = pf.Coordinates(
        vertices[:,0], vertices[:,1], vertices[:,2])
    cmap = cm.viridis
    data_plot = data_in/np.max((data_in))
    for i, reg in enumerate(sv.regions):
        reg_cp = reg.copy()
        reg_cp.append(reg_cp[0])
        xi = vertices_coords[reg_cp].azimuth
        yi = vertices_coords[reg_cp].colatitude*180/np.pi
        if i >= data_plot.shape[0]:
            continue
        zi = data_plot[i]
        if zi > 0 or not plot_empty:
            assert zi*factor_cmap <= 1, f'zi={zi}; factor={factor_cmap}'
            cax = ax.fill(
                xi, yi, color=cmap(zi))
            # print(f'plotting {i}; c={color}')
        if grid_color is not None:
            ax.plot(
                xi, yi, grid_color, linewidth=0.1)
    if source_pos is not None:
        xi = source_pos.azimuth
        yi = source_pos.colatitude*180/np.pi
        ax.scatter(xi, yi, c='red', marker='x')

    ax.grid(True)
    ax.set_rlim((0, 90))

    plt.colorbar(mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(0, np.max(data_in)), cmap=cmap),
        ax=ax, orientation='vertical')
    return ax


def patches(patches_points, energy, ax=None):
    """Show the energy of the patches in 3d.

    Parameters
    ----------
    patches_points : np.ndarray, list
        The points of the patches of shape (#patches, #points, 3).
    energy : _type_
        energy for each patch of shape (#patches,).
    ax : matplotlib.axes.Axes, optional
        The axes to plot on., by default None

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes to plot on.

    """
    if ax is None:
        ax = plt.axes(projection='3d')

    cmap = cm.viridis
    energy_normalized = energy / np.max(energy)

    if len(patches_points.shape) == 2:
        patches_points = np.array([patches_points])

    for i in range(patches_points.shape[0]):
        ax.add_collection3d(Poly3DCollection(
            patches_points[i][np.newaxis, ...], color=cmap(energy_normalized[i])))

    
    plt.colorbar(mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(0, np.max(energy)), cmap=cmap),
        ax=ax, orientation='vertical')
    return ax
