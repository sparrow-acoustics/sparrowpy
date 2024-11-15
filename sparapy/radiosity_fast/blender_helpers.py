import sys
import os
from pathlib import Path
import bpy, bmesh
import numpy as np


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def read_geometry_file(blend_file: Path):

    if os.path.splitext(blend_file)[-1] == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(blend_file))
    elif os.path.splitext(blend_file)[-1] ==".stl":
        bpy.ops.wm.stl_import(filepath=str(blend_file))
    else:
        NotImplementedError("Only .stl and .blend files are supported.")

    ensure_object_mode()

    objects = bpy.data.objects


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
    bmesh.ops.dissolve_limit(surfs, angle_limit=5*np.pi/180, verts=surfs.verts, edges=surfs.edges)

    finemesh = generate_connectivity(out_mesh)
    roughmesh = generate_connectivity(surfs)

    return finemesh, roughmesh

def read_stl_file(blend_file: Path):
    

    ensure_object_mode()

    objects = bpy.data.objects


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
    bmesh.ops.dissolve_limit(surfs, angle_limit=5*np.pi/180, verts=surfs.verts, edges=surfs.edges)

    finemesh = generate_connectivity(out_mesh)
    roughmesh = generate_connectivity(surfs)

    return finemesh, roughmesh

def ensure_object_mode():
    """Ensure Blender is in Object Mode."""
    if bpy.context.object:
        if bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

def generate_connectivity(mesh: bmesh):
    """generate node list and surf connectivity matrix"""

    out_mesh = dict({"conn":[], "verts": np.array([])})

    out_mesh["verts"] = np.array([v.co for v in mesh.verts])

    for f in mesh.faces:
        line = []
        for v in f.verts:
            line.append(v.index)
        out_mesh["conn"].append(line)

    return out_mesh