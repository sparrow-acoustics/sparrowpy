"""
Placeholder for general introduction to SceneGeometry class.
"""

import numpy as np



class SceneGeometry:
    """
    Class to handle the geometry of the scene.

    It contains the walls and patches and its orientations of the scene.
    """

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
    def patches_from_mesh(cls, vertices, patches_connectivity):
        """Initializes the patches from a given mesh.

        Parameters
        ----------
        vertices : array_like
            cartesian coordinates of the vertices, must be of shape
            (n_vertices, 3)
        patches_connectivity : array_like
            connectivity of the patches, must be of shape
            (n_walls, n_vertices_per_patch)
            n_vertices_per_patch must be constant for all patches.
        """
        raise NotImplementedError()

    @classmethod
    def walls_from_file(cls, file_path, geometry_name):
        """Initializes the walls from a given file.

        The material names are read from the file and stored in the
        _walls_material_names attribute.

        Parameters
        ----------
        file_path : str, Path
            path to the geometry file. The file must be in the format
            ``.blend``, the imported Geometry must have the name
            ``'geometry_name'``
        geometry_name : str
            name of the geometry object in the file. All other geometry objects
            will be ignored.
        """
        raise NotImplementedError()

    @classmethod
    def patches_from_file(cls, file_path, geometry_name):
        """Initializes the patches from a given file.

        The material names are read from the file and stored in the
        _walls_material_names attribute.

        Parameters
        ----------
        file_path : str, Path
            path to the geometry file. The file must be in the format
            ``.blend``, the imported Geometry must have the name
            ``'geometry_name'``
        geometry_name : str
            name of the geometry object in the file. All other geometry objects
            will be ignored.
        """
        raise NotImplementedError()

    def walls_from_patches(self):
        """Creates walls from patches."""
        # all the merge patches into walls or each patch, one wall
        raise NotImplementedError()

    def patches_from_walls_equal_area(self, patch_size):
        """
        Create patches from walls.

        The patches are created by dividing the walls into smaller patches
        of equal area.
        """
        # we already have this... :
        raise NotImplementedError()

    def set_material_names_per_wall(self, list_names):
        """
        Sets the material names for each wall.

        Parameters
        ----------
        list_names : list[str]
            list of material names for each wall.
        """
        raise NotImplementedError()

    def validate_geometry(self):
        """Validates the geometry."""
        # is all set, walls, patches, materials
        raise NotImplementedError()

    def plot(self):
        """Plots the geometry."""
        raise NotImplementedError()

    def clear_patches(self):
        """Remove the patches."""
        raise NotImplementedError()

    def clear_walls(self):
        """Remove the walls."""
        raise NotImplementedError()
