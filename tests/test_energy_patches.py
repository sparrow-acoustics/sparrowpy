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

array = np.array([[
    [[2, 0, 1, 6], [0, 4, 8, 0]],
    [[2, 2, 0, 8], [2, 0, 2, 8]],
    [[2, 7, 7, 2], [2, 4, 3, 1]]
], [
    [[9, 1, 3, 8], [2, 5, 2, 4]],
    [[4, 9, 8, 7], [3, 3, 8, 0]],
    [[3, 2, 3, 8], [8, 6, 6, 7]]
]])

print(array)

# %%
array = np.array([
    [[6., 9., 8., 16.],
     [4., 8., 13., 9.]],
    
    [[16., 12., 14., 23.],
     [13., 14., 16., 11.]]
])