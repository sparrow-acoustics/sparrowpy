import os
import pytest
import sparrowpy as sp
from pyfar.testing.plot_utils import create_figure, save_and_compare
import numpy as np
import matplotlib.pyplot as plt

"""
For general information on testing plot functions see
https://pyfar-gallery.readthedocs.io/en/latest/contribute/contribution_guidelines.html#testing-plot-functions

Important:
- `create_baseline` and `compare_output` must be ``False`` when pushing
  changes to pyfar.
- `create_baseline` must only be ``True`` if the behavior of a plot function
  changed. In this case it is best practice to recreate only the baseline plots
  of the plot function (plot behavior) that changed.
"""
# global parameters -----------------------------------------------------------
create_baseline = False

# file type used for saving the plots
file_type = "png"

# if true, the plots will be compared to the baseline and an error is raised
# if there are any differences. In any case, differences are written to
# output_path as images
compare_output = False

# path handling
base_path = os.path.join('tests', 'test_plot_data')
baseline_path = os.path.join(base_path, 'baseline')
output_path = os.path.join(base_path, 'output')

if not os.path.isdir(base_path):
    os.mkdir(base_path)
if not os.path.isdir(baseline_path):
    os.mkdir(baseline_path)
if not os.path.isdir(output_path):
    os.mkdir(output_path)

# remove old output files
for file in os.listdir(output_path):
    os.remove(os.path.join(output_path, file))

# the naming scheme of the baseline is as follows:
# <function_name>_<parameter_name>_<parameters>.png

# testing ---------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _close_all_figures():
    """
    Close all matplotlib figures after each test to prevent test
    pollution.
    """
    yield
    plt.close('all')


def test_polygons_3d_default():
    single_patch = np.array([[[0, 0, 0],
                              [1, 0, 0],
                              [1, 1, 0],
                              [0, 1, 0]]])
    energy = np.array([1])

    # do plotting
    filename = 'polygons_3d_default'
    create_figure()
    sp.plot.polygons_3d(single_patch, energy)
    save_and_compare(
        create_baseline, baseline_path, output_path, filename,
        file_type, compare_output)


@pytest.mark.parametrize("colorbar", [True, False])
def test_polygons_3d_colorbar(colorbar):
    single_patch = np.array([[[0, 0, 0],
                              [1, 0, 0],
                              [1, 1, 0],
                              [0, 1, 0]]])
    energy = np.array([1])

    # do plotting
    filename = f'polygons_3d_colorbar_{colorbar}'
    create_figure()
    sp.plot.polygons_3d(single_patch, energy, colorbar=colorbar)
    save_and_compare(
        create_baseline, baseline_path, output_path, filename,
        file_type, compare_output)


def test_polygons_3d_shoebox_room(sample_walls):
    edge_points = np.array([wall.pts for wall in sample_walls])
    energy = np.arange(len(sample_walls)) + 1

    # do plotting
    filename = 'polygons_3d_shoebox_room'
    create_figure()
    sp.plot.polygons_3d(edge_points, energy)
    save_and_compare(
        create_baseline, baseline_path, output_path, filename,
        file_type, compare_output)

def test_polygons_3d_energy_not_1d():
    edge_points = np.array([[[0, 0, 0],
                                [1, 0, 0],
                                [1, 1, 0],
                                [0, 1, 0]]])
    energy = np.array([[1]])  # 2D array -> invalid
    with pytest.raises(ValueError, match="energy must be a 1D array."):
        sp.plot.polygons_3d(edge_points, energy)


def test_polygons_3d_edge_points_not_3d():
    edge_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0]])  # 2D array -> invalid
    energy = np.array([1])
    match="edge_points must be of shape "
    with pytest.raises(ValueError, match=match):
        sp.plot.polygons_3d(edge_points, energy)


def test_polygons_3d_mismatch_count():
    edge_points = np.array([
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    ])  # two polygons
    energy = np.array([1])  # only one energy value
    match = (
        "The number of polygons in edge_points must match the number of "
        "energy values.")
    with pytest.raises(ValueError, match=match):
        sp.plot.polygons_3d(edge_points, energy)


def test_polygons_3d_edge_points_not_convertible():
    energy = np.array([1])
    match = "edge_points must be convertible to a numpy array."
    with pytest.raises(ValueError, match=match):
        sp.plot.polygons_3d('not convertible', energy)


def test_polygons_3d_energy_not_convertible():
    edge_points = np.array([[
            [0, 0, 0],
          [1, 0, 0],
          [1, 1, 0],
          [0, 1, 0]]])
    match = "energy must be convertible to a numpy array."
    with pytest.raises(ValueError, match=match):
        sp.plot.polygons_3d(edge_points, 'not convertible')
