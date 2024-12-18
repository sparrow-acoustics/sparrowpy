import os
import pytest
import pyfar as pf
from pyfar.testing.plot_utils import create_figure, save_and_compare
import numpy as np
import sparapy as sp

def test_energy_matrix_patches():
    
    E_matrix = np.array([
        [
            [[2, 0, 1, 6], [0, 4, 8, 0]],
            [[2, 2, 0, 8], [2, 0, 2, 8]],
            [[2, 7, 7, 2], [2, 4, 3, 1]]
        ], 
        [
            [[9, 1, 3, 8], [2, 5, 2, 4]],
            [[4, 9, 8, 7], [3, 3, 8, 0]],
            [[3, 2, 3, 8], [8, 6, 6, 7]]
        ]
    ])

    # Assuming sp.plot.energy_patches computes some transformation on E_matrix
    energy = sp.plot.energy_patches(E_matrix)

    expected_energy = np.array([
        [[6., 9., 8., 16.],
         [4., 8., 13., 9.]],
        
        [[16., 12., 14., 23.],
         [13., 14., 16., 11.]]
    ])

    assert np.array_equal(energy, expected_energy), "Energy computation does not match expected values."