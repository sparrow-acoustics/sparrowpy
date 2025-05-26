"""
Placeholder for general introduction to SceneGeometry class.
"""

import numpy as np
import utils.blender as bd



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
    _material_name_list: list[str]
    _material_id_to_wall: np.ndarray

    def __init__(
            self,
            vertices:np.ndarray,
            walls_connectivity:list,
            walls_normals:np.ndarray,
            walls_up_vectors:np.ndarray,
            patches_connectivity:np.ndarray=None,
            materials_list:np.ndarray=None,
            materials_connectivity:np.ndarray=None):

        self._vertices = vertices
        self._walls_connectivity = walls_connectivity
        self._walls_normals = walls_normals
        self._walls_up_vectors = walls_up_vectors
        self._patches_connectivity = patches_connectivity

        if materials_list is not None:

            self._material_name_list = list(dict.fromkeys(materials_list))

            if (materials_list.shape[0]==len(self._walls_connectivity) and
                materials_connectivity is None):

                mat_conn = []
                for material in self._material_name_list:
                    mat_conn.append(
                            [k for k,mat in enumerate(materials_list)
                            if mat==material])
                self._material_id_to_wall = np.array(mat_conn)

            elif materials_connectivity is not None:
                self._material_id_to_wall = materials_connectivity


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
    def walls_from_file(cls, file_path,
                        geometry_name="Geometry",
                        auto_detect_walls = True):
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

        auto_detect_walls: bool
            If True, walls are detected an assembled automatically based on
            the model's material and face orientation [recommended].
            This overwrites the polygons in the scene, reducing the existing
            faces to fewer n-gons.
        """
        wall_data = bd.read_geometry_file(blend_file=file_path,
                                          wall_auto_assembly=auto_detect_walls,
                                          patches_from_model=False,
                                          blender_geom_id=geometry_name)

        cls(vertices = wall_data["verts"],
            walls_connectivity = wall_data["conn"],
            walls_normals = wall_data["normal"],
            walls_up_vectors = wall_data["up"],
            materials_list = wall_data["materials"])

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

    def _update_scene_mesh(self, vertices, connectivity):
        """Update scene geometry vertex list and connectivity.

        Removes redundant vertices and replaces connectivity index
        with previously existing one.

        Parameters
        ----------
        vertices: np.ndarray(n_vertices,3)
            input vertex list.
        connectivity: list
            mesh connectivity based on input vertex list.

        Returns
        -------
        conn_updated: list
            mesh connectivity based on updated scene vertex list.
        """

        if self._vertices is None:
            self._vertices = vertices
            conn_updated = connectivity

        else:
            new_ids = [0 for i in range(vertices)]
            for in_id in range(vertices):
                found_vertex = False
                for ref_id in range(self._vertices.shape[0]):
                    if np.linalg.norm(
                        vertices[in_id]-self._vertices[ref_id])<1e-3:
                        found_vertex=True
                        break
                if found_vertex:
                    new_ids[in_id]=ref_id
                else:
                    new_ids[in_id]=self._vertices.shape[0]
                    self._vertices = np.append(self._vertices,
                                               np.expand_dims(vertices[in_id]))

            conn_updated = _update_conn(conn=connectivity,new_ids=new_ids)

        return conn_updated





def _update_conn(conn,new_ids):
    """Update index in connectivity."""
    for i,poly in enumerate(conn):
        for old_id in range(new_ids):
            conn[i] = poly[poly==old_id]=new_ids[old_id]

    return conn



