"""methods for tracing and evaluating ff integration."""
import sparrowpy.form_factor.integration as intg
import sparrowpy.testing.exact_ff_solutions as ffsol
import sparrowpy.geometry as geo
import numpy as np

def _select_geometry(gtype="a",w=1,h=1,l=1):
    """Select geometry (parallel or coincident line patches)"""
    match gtype:
        case "a":
            patch_1 = geo.Polygon(
                points=[[0, 0, 0],
                        [w, 0, 0],
                        [w, 0, h],
                        [0, 0, h]],
                normal=[0, 1, 0],
                up_vector=[1, 0, 0],
            )

            patch_2 = geo.Polygon(
                points=[
                    [0, l, 0],
                    [0, l, h],
                    [w, l, h],
                    [w, l, 0]],
                normal=[0, -1, 0],
                up_vector=[1, 0, 0],
            )

            solution = ffsol.parallel_patches(w, h, l)

        case "b":
            patch_1 = geo.Polygon(
                points=[
                    [0, 0, 0],
                    [0, l, 0],
                    [0, l, h],
                    [0, 0, h]],
                normal=[1, 0, 0],
                up_vector=[1, 0, 0],
            )

            patch_2 = geo.Polygon(
                points=[[0, 0, 0],
                        [w, 0, 0],
                        [w, l, 0],
                        [0, l, 0]],
                normal=[0, 0, 1],
                up_vector=[1, 0, 0],
            )

            solution = ffsol.perpendicular_patch_coincidentline(
                            w, h, l,
                        )
        case _:
            ValueError(f"{gtype} does not correspond to any known geometries.")

    return (patch_1,patch_2),solution

