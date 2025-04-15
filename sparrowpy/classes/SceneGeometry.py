import numpy as np



def SceneGeometry():
    _walls_connectivity: list[int]
    _patches_connectivity: np.ndarray
    _walls_normals: np.ndarray
    _walls_up_vectors: np.ndarray
    _vertices: np.ndarray

    def __init__(
            vertices, walls_connectivity, walls_normals, walls_up_vectors):
        pass

    @classmethod
    def walls_from_mesh(cls, vertices, faces):
        # like what we are doing now
        raise NotImplementedError()

    @classmethod
    def patches_from_polygons(cls, vertices, faces):
        raise NotImplementedError()

    @classmethod
    def walls_from_file(cls, file_path):
        raise NotImplementedError()

    @classmethod
    def patches_from_file(cls, file_path):
        raise NotImplementedError()

    def walls_from_patches():
        # all the merge patches into walls or each patch, one wall
        raise NotImplementedError()

    def patches_from_walls_equal_area(patch_size):
        # we already have this... :
        raise NotImplementedError()

    def set_materials_per_wall(list_names):
        raise NotImplementedError()

    def assign_materials(dict):
        # dict with names brdfs etc... maybe even a class for brdfs
        raise NotImplementedError()

    def validate_geometry():
        # is all set, walls, patches, materials
        raise NotImplementedError()

    def plot():
        raise NotImplementedError()

    def clear_patches():
        raise NotImplementedError()

    def clear_walls():
        raise NotImplementedError()
