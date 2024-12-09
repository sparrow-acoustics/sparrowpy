# %%
# imports

import numpy as np
import trimesh
from sparapy import geometry as geo
import matplotlib.pyplot as plt
import pyfar as pf
from sparapy import io
from sparapy import brdf
import sparapy as sp

def test_loadread_stl():

    # Read STL file that includes a 3x3x3 shoebox room. 6 faces in total.
    list_polygon = io.read_geometry('shoebox_3.stl','rectangle')

    # Test if function actually returns list of polygons
    assert all(isinstance(polygon, geo.Polygon) for polygon in list_polygon), \
        "Not all objects are of type sparapy.geometry.Polygon"

    # Test if the number of polygons in the list is correct
    assert len(list_polygon) == 6, f"Expected 6 polygons, but got {len(list_polygon)}"

    # Check if centers of the polygons are correct, order does not matter
    expected_centers = [
        np.array([0.0, 1.5, 1.5]),
        np.array([1.5, 1.5, 0.0]),
        np.array([1.5, 1.5, 3.0]),
        np.array([3.0, 1.5, 1.5]),
        np.array([1.5, 0.0, 1.5]),
        np.array([1.5, 3.0, 1.5])
    ]

    expected_normals = [
        np.array([1.0, 0.0, 0.0]),  # Corresponding normal for center [0.0, 1.5, 1.5]
        np.array([0.0, 0.0, 1.0]),  # Corresponding normal for center [1.5, 1.5, 0.0]
        np.array([0.0, 0.0, -1.0]),  # Corresponding normal for center [1.5, 1.5, 3.0]
        np.array([1.0, 0.0, 0.0]),  # Corresponding normal for center [3.0, 1.5, 1.5]
        np.array([0.0, -1.0, 0.0]),  # Corresponding normal for center [1.5, 0.0, 1.5]
        np.array([0.0, 1.0, 0.0])  # Corresponding normal for center [1.5, 3.0, 1.5]
    ]

    actual_centers = [polygon.center for polygon in list_polygon]
    actual_normals = [polygon.normal for polygon in list_polygon]

    # Check centers and normals
    for expected_center, expected_normal in zip(expected_centers, expected_normals):
        # Find the polygon whose center matches this expected center
        matching_index = None
        for idx, actual_center in enumerate(actual_centers):
            if np.allclose(expected_center, actual_center):
                matching_index = idx
                break
        
        # Assert that a matching center was found
        assert matching_index is not None, f"Expected center {expected_center} not found in actual centers"

        # Check if the corresponding normal (in absolute values) matches
        actual_normal = actual_normals[matching_index]
        assert np.allclose(np.abs(expected_normal), np.abs(actual_normal)), \
            f"Expected normal {np.abs(expected_normal)} but got {np.abs(actual_normal)} for center {expected_center}"

   
