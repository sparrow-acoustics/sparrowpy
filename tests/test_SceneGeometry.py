"""Test the SceneGeometry class and its methods."""
import os

import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
import sparrowpy as sp

@pytest.mark.parametrize('filename', [
    "cube.blend",
    "cube.stl",
])
def test_walls_from_file(filename,file_path=os.getcwd()+"/tests/test_data/"):

    scene_geom = sp.SceneGeometry.walls_from_file(file_path=file_path+filename)

    stl_model = ".stl" in filename

    assert scene_geom._vertices.shape == (8,3)
    assert scene_geom._patches_connectivity is None

    if stl_model:
        assert len(scene_geom._walls_connectivity) == 6
        assert len(scene_geom._walls_connectivity[0]) == 4
        assert scene_geom._material_id_to_wall is None
        assert scene_geom._material_name_list is None

    else:
        assert len(scene_geom._walls_connectivity) == 12
        assert len(scene_geom._walls_connectivity[0]) == 3
        assert scene_geom._material_name_list == ["matA","matB"]
        npt.assert_array_equal(scene_geom._material_id_to_wall,
                               np.array([0,0,0,1,0,1,1,1,1,0,1,0]))


@pytest.mark.parametrize('filepath', [
    "tests/test_data/sample_walls.blend",
    ])
def test_init_comparison(filepath, sample_walls):
    file_scene = sp.SceneGeometry.walls_from_file(filepath)
    poly_scene = sp.SceneGeometry.from_polygon(sample_walls,
                                                        patch_size=1)
