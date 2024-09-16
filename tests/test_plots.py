import os
import pytest
import pyfar as pf
from pyfar.testing.plot_utils import create_figure, save_and_compare
import numpy as np
import sparapy as sp

"""
Testing plots is difficult, as matplotlib does not create the exact same
figures on different systems (e.g. single pixels vary).
Therefore, this file serves several purposes:
1. The usual call of pytest, which only checks, if the functions do not raise
errors.
2. Creating baseline figures. If the global parameter `create_baseline` is
set to True, figures are created in the corresponding folder. These need to be
updated and manually inspected and if the plot look changed.
3. Comparing the created images to the baseline images by setting the global
parameter `compare_output`. This function should only be activated if intended.

IMPORTANT: IN THE REPOSITORY, BOTH `CREATE_BASELINE` AND `COMPARE_OUTPUT` NEED
TO BE SET TO FALSE, SO THE TRAVIS-CI CHECKS DO NOT FAIL.
"""
# global parameters -----------------------------------------------------------
create_baseline = False

# file type used for saving the plots
file_type = "pgf"

# if true, the plots will be compared to the baseline and an error is raised
# if there are any differences. In any case, differences are written to
# output_path as images
compare_output = True

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


# testing ---------------------------------------------------------------------
@pytest.mark.parametrize('function', [
    sp.plot.brdf_3d,
    sp.plot.brdf_polar,

])
def test_brdf(function, brdf_s_0):
    print(f"Testing: {function.__name__}")
    data, _, r = pf.io.read_sofa(brdf_s_0)

    # initial plot
    filename = function.__name__ + '_default'
    create_figure()
    function(r, np.abs(data.freq[1, :, 0]))
    save_and_compare(
        create_baseline, baseline_path, output_path, filename,
        file_type, compare_output)


@pytest.mark.parametrize('function', [
    sp.plot.brdf_3d,
    sp.plot.brdf_polar,

])
def test_brdf_with_source(function, brdf_s_0):
    print(f"Testing: {function.__name__}")
    data, s, r = pf.io.read_sofa(brdf_s_0)

    # initial plot
    filename = function.__name__ + '_source'
    create_figure()
    function(r, np.abs(data.freq[1, :, 0]), source_pos=s[1])
    save_and_compare(
        create_baseline, baseline_path, output_path, filename,
        file_type, compare_output)
