"""Helper functions for handling of blender models."""
import sys
import os
from pathlib import Path
import bpy
import bmesh # type: ignore
import numpy as np
import trimesh as tm
import warnings
from mathutils import Vector

class DotDict(dict):
    """dot.notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def read_geometry_file(blend_file: Path,
                       angular_tolerance=1.,
                       max_patch_size=1.,
                       auto_walls=True,
                       auto_patches=True,
                       write_back=False):
    """Read blender file and return fine and rough mesh.

    Reads the input geometry from the blender file and reduces
    the mesh into a list of nodes (with spatial coordinates)
    and connectivity matrix (lists node indices belonging to same polygons).

    Furthermore, a rough mesh is generated in a similar fashion,
    by merging coplanar polygons into larger surface-wide polygons.

    Parameters
    ----------
    blend_file: Path
        path to blender file describing the
        scene geometry and setup

    angular_tolerance: float
        maximum angle in degree by which two patches are considered coplanar
        determines surfaces in simplified mesh

    max_patch_size: float
        maximum size of the patch edges.
        real patch size may be significantly smaller
            if max_patch_size is close to wall dimensions.

    auto_walls: bool
        flags if walls should be auto detected from the model geometry (True)
        or if each polygon in the model should become a wall (False).
        the output walls may be a rough triangularized version of the input.

    auto_patches: bool
        flags if patches should be automatically created given max size (True)
        or if each polygon in the model should become a patch (False).
        if False, the input is checked and/or corrected for consistent shape.

    Returns
    -------
    wall_data: dict
        wall vertex list ["verts"], polygon vertex mapping ["conn"],
        normals["normal"], and material names ["material"].

    patch_data: dict
        patch vertex list ["verts"], polygon vertex mapping ["conn"],
        patch-to-wall mapping ["wall_ID"]

    """
    if os.path.splitext(blend_file)[-1] == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(blend_file))
    elif os.path.splitext(blend_file)[-1] ==".stl":
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        bpy.ops.wm.stl_import(filepath=str(blend_file))
    else:
        NotImplementedError("Only .stl and .blend files are supported.")

    ensure_object_mode()
    objects = bpy.data.objects

    if os.path.splitext(blend_file)[-1] != ".blend":
        for obj in objects:
            obj.name = "Geometry"

    if "Geometry" not in objects:
        print("Geometry object not found in blend file")
        sys.exit()

    geometry = objects["Geometry"]


    # Creates file with only static geometric data of original blender file
    # without information about source and receiver
    bpy.ops.object.select_all(action="DESELECT")
    geometry.select_set(True)
    bpy.context.view_layer.objects.active = geometry


    # create bmesh from geometry
    out_mesh = bmesh.new()
    out_mesh.from_mesh(geometry.data)

    surfs = out_mesh.copy()
    surfs.transform(geometry.matrix_world)
    if auto_walls:
        # dissolve coplanar faces for simplicity
        bmesh.ops.dissolve_limit(surfs,angle_limit=angular_tolerance*np.pi/180,
                                verts=surfs.verts, edges=surfs.edges,
                                delimit={'MATERIAL'})

    if auto_patches:
        warnings.warn(                                    # noqa: B028
            RuntimeWarning(
                "A rough triangulation pass may be applied" +
                "to user-defined walls for auto patch generation."))
        bmesh.ops.triangulate(surfs, faces=list(surfs.faces),
                            quad_method="BEAUTY",
                            ngon_method="BEAUTY")
    elif not auto_patches:
        if not check_consistent_patches(list(surfs.faces)):
            warnings.warn(                                # noqa: B028
                UserWarning(
                    "User-input patches have inconsistent shapes." +
                    "\nA rough triangulation will be applied."))
            bmesh.ops.triangulate(surfs, faces=list(surfs.faces),
                            quad_method="BEAUTY",
                            ngon_method="BEAUTY")


    wall_data = generate_connectivity_wall(surfs)

    if auto_patches:
        patch_data = generate_patches(wall_data,max_patch_size=max_patch_size)
    else:
        patch_data = {"conn":   np.array(wall_data["conn"]),
                      "verts":  np.array(wall_data["verts"]),
                      "wall_ID":np.arange(len(wall_data["conn"]))}

    if write_back:
        write_to_file(blend_file, wall_data, patch_data)

    return wall_data, patch_data

def check_collections(colname):
    """Find or create blender collections."""

    collections = bpy.data.collections

    foundcol=False
    for col in collections:
        if col.name==colname:
            foundcol=True

    if not foundcol:
        colout = bpy.data.collection.new(colname)
        bpy.context.scene.collection.children.link(colout)


def write_to_file(blend_file, wall_data, patch_data):
    """Write walls, patches, materials to blender model."""
    check_collections("sparrow-model")
    ## walls
    walls = bpy.data.meshes.new(name="walls")

    verts = []
    for v in walls["verts"]:
        verts.append(Vector((v[0],v[1],v[2])))

    walls.from_pydata(verts, [], walls["conn"])

    



def ensure_object_mode():
    """Ensure Blender is in Object Mode."""
    if bpy.context.object:
        if bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

def generate_connectivity_wall(mesh: bmesh):
    """Summarize characteristics of polygons in a given mesh.

    Return a dictionary which includes a list of vertices,
    mapping (connectivity) of the vertex list to each mesh polygon,
    list of normals for each patch, and list of materials.

    Parameters
    ----------
    mesh: bmesh
        mesh extracted from blender file

    Returns
    -------
    out_mesh: dict({
                    "verts": np.ndarray(n_vertices,3),
                    "conn":  list (n_polygons, :),
                    "normal": list (n_polygons, 3),
                    "material": list(n_polygons)
                    })
        mesh in reduced data representation.
            "verts":    vertex (node) list
            "conn":     vertex to polygon mapping,
            "normal":   normal list
            "material": material list

    """
    out_mesh = {"verts": np.array([]), "conn":[], "normal":[], "material": []}

    out_mesh["verts"] = np.array([v.co for v in mesh.verts])

    for f in mesh.faces:
        if len(bpy.context.object.material_slots)!=0:
            out_mesh["material"].append(bpy.context.object.material_slots[f.material_index].name)

        line=[]

        for v in f.verts:
            line.append(v.index)
        out_mesh["conn"].append(line)

        out_mesh["normal"].append(np.array(f.normal))

    return out_mesh

def generate_patches(walls: dict, max_patch_size=1.):
    """Generate patches automatically for each wall based on max edge size.

    Parameters
    ----------
    walls: dict({
                    "verts": np.ndarray(n_wall_verts,3),
                    "conn":  list(n_walls, :),
                    "normal": list(n_walls, 3),
                    "material": list(n_walls)
                    })
        wall geometry data.
            "verts":    wall vertex (node) list
            "conn":     vertex to wall mapping,
            "normal":   normal list
            "material": material list

    max_patch_size: float
        maximum edge dimension of patches

    Returns
    -------
    patches: dict({
                    "verts": np.ndarray(n_patch_verts,3),
                    "conn":  np.ndarray(n_patches, 3),
                    "wall_id": np.ndarray(n_patches,)
                    })
        patch data in reduced data representation.
            "verts":   patch vertex (node) list
            "conn":    vertex to patch mapping,
            "wall_ID": patch to wall mapping

    """
    patches={"conn":np.array([]),
             "verts": np.array([]),
             "wall_ID": np.array([])}

    verts, facs, IDs=tm.remesh.subdivide_to_size(vertices=walls["verts"],
                                        faces=walls["conn"],
                                        max_edge=max_patch_size,
                                        return_index=True)

    patches["verts"]   = np.array(verts)
    patches["conn"]    = np.array(facs)
    patches["wall_ID"] = np.array(IDs)

    return patches


def check_consistent_patches(surflist: list):
    """Check if all patches have the same shape.

    Return True if all polygons in a given mesh
    have the same number of vertices (all triangles, all quads, etc).
    Return False otherwise.

    Parameters
    ----------
    surflist: list(bmesh.faces)
        list of all faces (polygons) in a given mesh


    Returns
    -------
    out: bool
        flags if all polygons in mesh have the same shape (True)
        or not (False).

    """
    out = True
    for i in range(1,len(surflist)):
        if len(surflist[0].verts) != len(surflist[i].verts):
            out = False
            break

    return out
