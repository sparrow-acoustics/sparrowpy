# %% 
# imports

import os
import pytest
import pyfar as pf
from pyfar.testing.plot_utils import create_figure, save_and_compare
import numpy as np
import sparapy as sp

# %% 
# first patches function

# global parameters -----------------------------------------------------------
create_baseline = True # this was True before

# file type used for saving the plots
file_type = "png"

# if true, the plots will be compared to the baseline and an error is raised
# if there are any differences. 
# In any case, differences are written to
# output_path as images
base_path = os.path.join('tests', 'test_plot_data')
baseline_path = os.path.join(base_path, 'baseline')
output_path = os.path.join(base_path, 'output')

compare_output = True # this was True before. where in the repository should this be false? here?
filename =  'patches_default'
create_figure()
sp.plot.patches(
    np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
    np.array([1]))
    
# initial plot
save_and_compare(
    create_baseline, baseline_path, output_path, filename,
    file_type, compare_output)
# %%
