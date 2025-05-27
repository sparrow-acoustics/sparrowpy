"""Test the SceneGeometry class and its methods."""
import os

import numpy as np
import numpy.testing as npt
import pytest
import sparrowpy as sp
import sparrowpy.utils.blender as bd

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


@pytest.mark.parametrize('base_filename', [
    "cube.stl",
    "cube_simple.blend"])
@pytest.mark.parametrize('patch_filename', ["cube_simple.blend"])
def test_patches_to_walls_map(
                     base_filename,
                     patch_filename,
                     file_path=os.getcwd()+"/tests/test_data/"):
    scene_geom = sp.SceneGeometry.walls_from_file(
                                file_path=file_path+base_filename)

    patches = bd.read_geometry_file(blend_file=file_path+patch_filename,
                                          wall_auto_assembly=False,
                                          blender_geom_id="Geometry")

    scene_geom._map_patches_to_walls(patches)

    npt.assert_array_equal(np.sort(np.unique(scene_geom._patches_to_wall)),
                           np.arange(len(scene_geom._walls_connectivity)))
    npt.assert_array_equal(np.bincount(scene_geom._patches_to_wall),
                           4*np.ones((len(scene_geom._walls_connectivity)),dtype=int))

@pytest.mark.parametrize('base_filename', [
    "cube.stl",
    "cube_simple.blend"])
@pytest.mark.parametrize('patch_filename', ["cube_simple.blend"])
def test_mesh_update(base_filename,
                     patch_filename,
                     file_path=os.getcwd()+"/tests/test_data/"):
    scene_geom = sp.SceneGeometry.walls_from_file(
                                file_path=file_path+base_filename)

    verts_i = scene_geom._vertices

    patches = bd.read_geometry_file(blend_file=file_path+patch_filename,
                                          wall_auto_assembly=False,
                                          blender_geom_id="Geometry")

    verts_f = scene_geom._vertices

    p_conn = scene_geom._update_scene_mesh(vertices=patches["verts"],
                                                connectivity=patches["conn"])

    for i in range(len(scene_geom._walls_connectivity)):
        npt.assert_array_equal(verts_i[scene_geom._walls_connectivity[i]],
                               verts_f[scene_geom._walls_connectivity[i]])

    assert np.amax(p_conn)>np.amax(scene_geom._walls_connectivity)


@pytest.mark.parametrize('filepath', [
    "tests/test_data/sample_walls.blend",
    ])
def test_init_comparison(filepath, sample_walls):
    pass
    # file_scene = sp.SceneGeometry.walls_from_file(filepath)
    # poly_scene = sp.SceneGeometry.from_polygon(sample_walls,
    #                                             patch_size=1)
