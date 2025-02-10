"""Helper functions for handling of blender models."""
import sys
import os
from pathlib import Path
import bpy
import bmesh
import numpy as np
import trimesh as tm

class DotDict(dict):
    """dot.notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def read_geometry_file(blend_file: Path,
                       angular_tolerance=1.,
                       max_patch_size=1.):
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

    Returns
    -------
    finemesh: dict
        includes vertex list and polygon connectivity matrix.

    roughmesh: dict
        mesh with reduced number of vertices.
        coplanar polygons in the input model are dissolved to
        create large surfaces.
        includes vertex list and polygon connectivity matrix.


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

    # dissolve coplanar faces for visibility check
    surfs = out_mesh.copy()
    bmesh.ops.dissolve_limit(surfs, angle_limit=angular_tolerance*np.pi/180,
                             verts=surfs.verts, edges=surfs.edges,
                             delimit={'MATERIAL'})
    
    bmesh.ops.triangulate(surfs, faces=list(surfs.faces),
                          quad_method="BEAUTY",
                          ngon_method="BEAUTY")

    wall_data = generate_connectivity_wall(surfs)

    patch_data = generate_patches(wall_data,max_patch_size=max_patch_size)

    return wall_data, patch_data


def ensure_object_mode():
    """Ensure Blender is in Object Mode."""
    if bpy.context.object:
        if bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

def generate_connectivity_wall(mesh: bmesh):
    """Generate node list and surf connectivity matrix.

    Parameters
    ----------
    mesh: bmesh
        mesh extracted from blender file

    Returns
    -------
    out_mesh: dict({
                    "conn":  list (n_polygons, :),
                    "norm": list (n_polygons, 3),
                    "verts": np.ndarray (n_nodes,3)
                    })
        mesh in reduced data representation.

    """
    out_mesh = {"conn":[], "verts": np.array([]), "normal":[], "material": []}

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

def generate_patches(mesh: dict, max_patch_size=2):
    """Generate patches procedurally for each wall based on max edge size."""

    patches={"conn":np.array([]),
             "verts": np.array([]),
             "wall_ID": np.array([])}

    verts, facs, IDs=tm.remesh.subdivide_to_size(vertices=mesh["verts"],
                                        faces=mesh["conn"],
                                        max_edge=max_patch_size,
                                        return_index=True)

    patches["verts"]   = np.array(verts)
    patches["conn"]    = np.array(facs)
    patches["wall_ID"] = np.array(IDs)

    return patches
