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



def test_read_stl():

    # read stl file that includes a 3x3x3 shoebox room. 6 faces in total. 
    list_polygon = io.read_geometry('models/shoebox_3.stl')

    # test if function actually returns list of polygons 
    assert all(isinstance(polygon, geo.Polygon) for polygon in list_polygon), "Not all objects are of type sparapy.geometry.Polygon"

    # test if the number of polygons in the list are correct
    assert len(list_polygon) == 6, f"Expected 6 polygons, but got {len(list_polygon)}"

    # check if centers of the polygon are correct, order does not matter
    expected_centers = [
        np.array([1.5, 1.5, 0.0]),
        np.array([1.5, 1.5, 3.0]),
        np.array([3.0, 1.5, 1.5]),
        np.array([1.5, 0.0, 1.5]),
        np.array([1.5, 3.0, 1.5]),
        np.array([0.0, 1.5, 1.5])
    ]
    actual_centers = [polygon.center for polygon in list_polygon]
    for expected_center in expected_centers:
        assert any(np.allclose(expected_center, actual_center) for actual_center in actual_centers), \
            f"Expected center {expected_center} not found in actual centers"
        
    # check if the normals of the polygon are correct  



# %%
# manual testing
list_polygon = io.read_geometry('models/shoebox_3.stl')

for polygon in list_polygon:
    print(polygon.normal)


# %%
