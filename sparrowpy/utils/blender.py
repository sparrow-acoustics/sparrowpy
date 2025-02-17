"""Helper functions for handling of blender models."""
import sys
import os
from pathlib import Path
import bpy
import bmesh # type: ignore
import numpy as np
from sparrowpy.radiosity_fast.visibility_helpers import point_in_polygon

class DotDict(dict):
    """dot.notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def read_geometry_file(blend_file: Path,
                       angular_tolerance=1.,
                       auto_walls=True,
                       patches_from_geometry=True):
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
    surfs = bmesh.new()
    surfs.from_mesh(geometry.data)

    # sometimes the object space is scaled/rotated inside the .blend model.
    # this preserves the geometry as the user sees it inside blender.
    surfs.transform(geometry.matrix_world)

    if auto_walls:
        # dissolve coplanar faces for simplicity's sake
        bmesh.ops.dissolve_limit(surfs,angle_limit=angular_tolerance*np.pi/180,
                                 verts=surfs.verts, edges=surfs.edges,
                                 delimit={'MATERIAL'})

    if patches_from_geometry:
        # new bmesh with patch info
        patches=bmesh.new()
        patches.from_mesh(geometry.data)
        patches.transfrom(geometry.matrix_world)
        patch_data = generate_connectivity_patch(patches, surfs)



    wall_data = generate_connectivity_wall(surfs)

    if (not patches_from_geometry) and (check_geometry(wall_data)):
        wall_data["conn"] = np.array(wall_data["conn"])
        wall_data["verts"] = np.array(wall_data["verts"])
        wall_data["normal"] = np.array(wall_data["normal"])
        wall_data["material"] = np.array(wall_data["material"])
        return wall_data


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
        else:
            out_mesh["material"].append("")

        line=[]

        for v in f.verts:
            line.append(v.index)
        out_mesh["conn"].append(line)

        out_mesh["normal"].append(np.array(f.normal))

    return out_mesh

def generate_connectivity_patch(finemesh: bmesh, broadmesh:bmesh):

    out_mesh = {"verts":np.array([]), "conn":np.array([]), "map":np.array([])}

    out_mesh["verts"] = np.array([v.co for v in finemesh.verts])
    out_mesh["map"] = np.empty((len(finemesh.faces)),dtype=int)

    for i,pface in enumerate(finemesh.faces):
        out_mesh["conn"].append([v.index for v in pface.verts])

        for j,wface in enumerate(broadmesh.faces):
            if (pface.normal==wface.normal).all():
                if pface.material_index==wface.material_index:
                    if bmesh.geometry.intersect_face_point(wface,
                                                            pface.calc_center_median()):
                        out_mesh["map"][i]=j
                        break

    out_mesh["conn"] = np.array(out_mesh["conn"])

    return out_mesh

def check_geometry(walls: dict):
    """Check if all patches have the same shape.

    Return True if all polygons in a given mesh
    have the same number of vertices (all triangles, all quads, etc).
    Return False otherwise.

    Parameters
    ----------
    walls: dict
        list of all faces (polygons) in a given mesh

    Returns
    -------
    out: bool
        flags if all polygons in mesh are regular quads (True)
        or not (False).

    """
    out=True
    for i in range(len(walls["conn"])):
        w = walls["verts"][walls["conn"][i]]
        nverts=len(w)
        for j in range(nverts):
            vec0 = w[(j+1)%nverts]-w[j]
            vec1 = w[(j+2)%nverts]-w[(j+1)%nverts]

            if (nverts != 4 or np.dot(vec0,vec1)<1e-6):
                ValueError("Walls of the model should be regular quads in shape (squares, rectangles).\n"+
                    "You can define walls by hand in the geometry model and set auto_walls=False.")
    return out
