"""Test the universal form factor module."""

import pytest
import sparrowpy.geometry as geo
import numpy as np
import sparrowpy.radiosity_fast.universal_ff.univ_form_factor as form_factor
import sparrowpy.testing.exact_ff_solutions as exact_solutions
from sparrowpy.sound_object import SoundSource, Receiver
from sparrowpy.radiosity import Patches
import time
from sparrowpy.radiosity_fast import form_factor as FFac


@pytest.mark.parametrize("width", [1.])
@pytest.mark.parametrize("height", [1.,2.,3.,4])
@pytest.mark.parametrize("distance", [1.,2,3,4])
def test_parallel_facing_patches(width, height, distance):
    """Test universal form factor for equal facing parallel patches."""
    exact = exact_solutions.parallel_patches(width, height, distance)

    patch_1 = geo.Polygon(
        points=[[0, 0, 0], [width, 0, 0], [width, 0, height], [0, 0, height]],
        normal=[0, 1, 0],
        up_vector=[1, 0, 0],
    )

    patch_2 = geo.Polygon(
        points=[
            [0, distance, 0],
            [0, distance, height],
            [width, distance, height],
            [width, distance, 0],
        ],
        normal=[0, -1, 0],
        up_vector=[1, 0, 0],
    )

    univ = FFac.universal(patches_points=np.array([patch_1.pts, patch_2.pts]),
                          patches_normals=np.array(
                              [patch_1.normal, patch_2.normal]),
                          patches_areas=np.array([patch_1.area, patch_2.area]),
                          visible_patches=np.array([[0,1]]))

    rel = 100 * abs(univ[0,1] - exact) / exact

    assert  rel < 1.5


@pytest.mark.parametrize("width", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("height", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("length", [1.0])
def test_perpendicular_coincidentline_patches(width, height, length):
    """Test universal form factor for perpendicular patches sharing a side."""
    patch_1 = geo.Polygon(
        points=[
            [0, 0, 0],
            [0, length, 0],
            [0, length, height],
            [0, 0, height],
        ],
        normal=[1, 0, 0],
        up_vector=[1, 0, 0],
    )

    patch_2 = geo.Polygon(
        points=[[0, 0, 0],
                [width, 0, 0],
                [width, length, 0],
                [0, length, 0]],
        normal=[0, 0, 1],
        up_vector=[1, 0, 0],
    )

    exact = exact_solutions.perpendicular_patch_coincidentline(
        width, height, length,
    )

    univ = FFac.universal(patches_points=np.array([patch_1.pts, patch_2.pts]),
                          patches_normals=np.array(
                              [patch_1.normal, patch_2.normal]),
                          patches_areas=np.array([patch_1.area, patch_2.area]),
                          visible_patches=np.array([[0,1]]))

    rel = 100 * abs(univ[0,1] - exact) / exact

    assert rel < 2


@pytest.mark.parametrize("width1", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("width2", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("length1", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("length2", [1.0, 2.0, 3.0])
def test_perpendicular_coincidentpoint_patches(
    width1, length1, width2, length2,
):
    """Test form factor for perpendicular patches w/ common vertex."""
    patch_1 = geo.Polygon(
        points=[
            [0., 0., 0.],
            [width2, 0., 0.],
            [width2, length2, 0.],
            [0., length2, 0.],
        ],
        normal=[0, 0, 1],
        up_vector=[0, 1, 0],
    )

    patch_2 = geo.Polygon(
        points=[
            [0., 0., 0.],
            [0., 0., width1],
            [0., -length1, width1],
            [0., -length1, 0.],
        ],
        normal=[1, 0, 0],
        up_vector=[1, 0, 0],
    )

    exact = exact_solutions.perpendicular_patch_coincidentpoint(
        width1, length2, width2, length1,
    )

    univ = FFac.universal(patches_points=np.array([patch_1.pts, patch_2.pts]),
                          patches_normals=np.array(
                              [patch_1.normal, patch_2.normal]),
                          patches_areas=np.array([patch_1.area, patch_2.area]),
                          visible_patches=np.array([[0,1]]))

    rel = 100 * abs(univ[0,1] - exact) / exact

    assert rel < 5


@pytest.mark.parametrize("side", [0.1, 0.2, 0.5, 1, 2])
@pytest.mark.parametrize(
    "source",
    [
        SoundSource(
            position=np.array([1, 7, 1]),
            view=np.array([0, -1, 0]),
            up=np.array([0, 0, 1]),
        ),
    ],
)
@pytest.mark.parametrize(
    "receiver",
    [
        Receiver(
            position=np.array([1, 1, 1]),
            view=np.array([0, -1, 0]),
            up=np.array([0, 0, 1]),
        ),
    ],
)
@pytest.mark.parametrize("patchsize", [0.1])
def test_point_surface_interactions(side, source, receiver, patchsize):
    """Test source-to-patch and patch-to-receiver factor calculation."""
    sr = 1000
    c = 343

    absor_factor = 0.1

    patch_pos = geo.Polygon(
        points=[[0, 0, 0], [side, 0, 0], [side, 0, side], [0, 0, side]],
        normal=[0, 1, 0],
        up_vector=[1, 0, 0],
    )

    patch = Patches(
        polygon=patch_pos,
        max_size=patchsize * side,
        other_wall_ids=[],
        wall_id=[0],
        absorption=absor_factor,
    )

    patch.init_energy_exchange(
        0, 0.1, source, sampling_rate=sr, speed_of_sound=c,
    )

    patch = source_cast(src=source, rpatch=patch, absor=absor_factor)

    receiver_cast(receiver, patch, sr, c)


def source_cast(src, rpatch, absor):
    """Cast and test source-to-patch factor calculation."""
    t0 = time.time()
    nuss = form_factor.pt_solution(point=src.position, patch_points=rpatch.pts)
    tf_nusselt = time.time() - t0

    true = sum(rpatch.E_matrix[rpatch.E_matrix != 0])

    rel_error_nuss = abs(true - nuss * (1 - absor)) / true * 100

    assert rel_error_nuss < 1.0

    print("nusselt approach error: " + str(rel_error_nuss) + "%")
    print(
        "nusselt approach runtime: "
        + str(tf_nusselt * 1000)
        + "ms \n #################################",
    )
    return rpatch


def receiver_cast(rcv, patch, sr, c):
    """Cast and test patch-to-receiver factor calculation."""
    true_rec_energy = np.sum(
        patch.energy_at_receiver(
            receiver=rcv,
            max_order=0,
            speed_of_sound=c,
            sampling_rate=sr,
        ),
    )

    patch_energy = np.sum(patch.E_matrix)

    nuss = form_factor.pt_solution(
            point=rcv.position, patch_points=patch.pts, mode="receiver",
            ) * patch_energy

    rel_error_nuss = abs(true_rec_energy - nuss) / true_rec_energy * 100

    assert rel_error_nuss < 1.0
