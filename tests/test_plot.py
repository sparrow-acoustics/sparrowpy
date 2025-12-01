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

