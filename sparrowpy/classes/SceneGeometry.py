import numpy as np



class SceneGeometry:
    _walls_connectivity: list[int]
    _patches_connectivity: np.ndarray
    _walls_normals: np.ndarray
    _walls_up_vectors: np.ndarray
    _vertices: np.ndarray

    def __init__(
            vertices, walls_connectivity, walls_normals, walls_up_vectors):
        pass

    @classmethod
    def walls_from_mesh(cls, vertices, walls_connectivity):
        """Initializes the walls from a given mesh.

        Parameters
        ----------
        vertices : array_like
            cartesian coordinates of the vertices, must be of shape
            (n_vertices, 3)
        walls_connectivity : array_like
            connectivity list of the faces, must have the length of the walls,
            the number of vertices per wall can vary.
        """
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

    def walls_from_patches(self):
        # all the merge patches into walls or each patch, one wall
        raise NotImplementedError()

    def patches_from_walls_equal_area(self, patch_size):
        # we already have this... :
        raise NotImplementedError()

    def set_material_names_per_wall(self, list_names):
        raise NotImplementedError()

    def validate_geometry(self):
        # is all set, walls, patches, materials
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()

    def clear_patches(self):
        raise NotImplementedError()

    def clear_walls(self):
        raise NotImplementedError()
