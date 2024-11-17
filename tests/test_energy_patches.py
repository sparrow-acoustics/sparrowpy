# %% 
# imports

import os
import pytest
import pyfar as pf
from pyfar.testing.plot_utils import create_figure, save_and_compare
import numpy as np
import sparapy as sp

# %% 
# create simple energy matrix 

# Define the shape of the array
shape = (2, 3, 2, 4)  # 2 frequencies, 3 orders, 2 patches, 4 time slots

# Create the array with random integers between 0 and 9
data = np.random.randint(0, 10, size=shape)

# Print the array
print("4D Array (frequency, order, patch, time):")
print(data)


# %%

energy_patches = sp.plot.energy_patches(data)
print(energy_patches)
# %%
