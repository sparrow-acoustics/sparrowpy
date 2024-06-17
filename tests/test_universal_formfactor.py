"""Test the universal form factor module."""

import pytest
import sparapy.geometry as geo
import numpy as np
import numpy.testing as npt
import sparapy.form_factor as form_factor
import sparapy.testing.exact_ff_solutions as exact_solutions
import sparapy.radiosity as radiosity
from sparapy.sound_object import SoundSource, Receiver
from sparapy.radiosity import Patches
import time

@pytest.mark.parametrize('width', [
    1.,2.,3.
    ])
@pytest.mark.parametrize('height', [
    1.,2.,3.
    ])
@pytest.mark.parametrize('distance', [
    1.,2.,3.
    ])
def test_parallel_facing_patches(width, height, distance):
    """Test universal form factor calculation for facing parallel patches of equal dimensions"""
    exact = exact_solutions.parallel_patches(width, height, distance)

    patch_1 = geo.Polygon(points=[[0,0,0],[width, 0, 0],[width, 0, height],[0,0,height]], normal=[0,1,0], up_vector=[1,0,0])

    patch_2 = geo.Polygon(points=[[0,distance,0],[width, distance, 0],[width, distance, height],[0,distance,height]], normal=[0,-1,0], up_vector=[1,0,0])

    computed = form_factor.calc_form_factor(source_pts=patch_1.pts, receiving_pts=patch_2.pts, source_normal=patch_1.normal, receiving_normal=patch_2.normal)

    assert 100 * abs(computed-exact)/exact < .5

@pytest.mark.parametrize('width', [
    1.,2.,3.
    ])
@pytest.mark.parametrize('height', [
    1.,2.,3.
    ])
@pytest.mark.parametrize('length', [
    1.,2.,3.
    ])
def test_perpendicular_coincidentline_patches(width, height, length):
    """Test universal form factor calculation for perpendicular patches sharing a side"""

    exact = exact_solutions.perpendicular_patch_coincidentline(width,height,length)

    patch_1 = geo.Polygon(points=[[0,0,0],[width, 0, 0],[width, length, 0],[0,length,0]], normal=[0,0,1], up_vector=[1,0,0])

    patch_2 = geo.Polygon(points=[[0,0,0],[0, length, 0],[0, length, height],[0,0,height]], normal=[1,0,0], up_vector=[1,0,0])

    computed = form_factor.calc_form_factor(source_pts=patch_1.pts, receiving_pts=patch_2.pts, source_normal=patch_1.normal, receiving_normal=patch_2.normal)

    assert 100 * abs(computed-exact)/exact < 5.


@pytest.mark.parametrize('width1', [
    1.,2.,3.
    ])
@pytest.mark.parametrize('width2', [
    1.,2.,3.
    ])
@pytest.mark.parametrize('length1', [
    1.,2.,3.
    ])
@pytest.mark.parametrize('length2', [
    1.,2.,3.
    ])
def test_perpendicular_coincidentpoint_patches(width1, width2, length1, length2):
    """Test universal form factor calculation for perpendicular patches sharing a vertex"""

    exact = exact_solutions.perpendicular_patch_coincidentpoint(width1,length2,width2,length1)

    patch_1 = geo.Polygon(points=[[0,0,0],[0, length1, 0],[0, length1, width1],[0,0,width1]], normal=[1,0,0], up_vector=[0,1,0])

    patch_2 = geo.Polygon(points=[[0,length1,0],[width2, length1, 0],[width2, length1+length2, 0],[0,length1+length2,0]], normal=[0,0,1], up_vector=[1,0,0])

    computed = form_factor.calc_form_factor(source_pts=patch_1.pts, receiving_pts=patch_2.pts, source_normal=patch_1.normal, receiving_normal=patch_2.normal)

    assert 100 * abs(computed-exact)/exact < 5.

#########################################################################################################
# compare to Kang implementation

@pytest.mark.parametrize('parallel_walls', [
    [0, 1], [1, 0], [2, 3], [3, 2], [4, 5], [5, 4]
    ])
@pytest.mark.parametrize('patch_size', [
    1/3,
    0.5,
    1,
    ])
def test_kang_comparison_parallel(sample_walls, parallel_walls, patch_size):
    """Test form factor calculation for parallel walls."""
    wall_source = sample_walls[parallel_walls[0]]
    wall_receiver = sample_walls[parallel_walls[1]]

    patch_1 = radiosity.Patches(wall_source, patch_size, [1], 0)
    patch_2 = radiosity.Patches(wall_receiver, patch_size, [0], 1)
    patches = [patch_1, patch_2]
    patch_1.calculate_form_factor(patches)
    old_ff = patch_1.form_factors
    patch_1.calculate_univ_form_factor(patches)
    assert 100*abs(np.concatenate(patch_1.form_factors).sum()-np.concatenate(old_ff).sum()) / np.concatenate(old_ff).sum() < 10

@pytest.mark.parametrize('perpendicular_walls', [
    [0, 2], [0, 3], [0, 4], [0, 5],
    [1, 2], [1, 3], [1, 4], [1, 5],
    [2, 0], [2, 1], [2, 4], [2, 5],
    [3, 0], [3, 1], [3, 4], [3, 5],
    [4, 0], [4, 1], [4, 2], [4, 3],
    [5, 0], [5, 1], [5, 2], [5, 3],
    ])
@pytest.mark.parametrize('patch_size', [
    1/3,
    0.5,
    1,
    ])
def test_kang_comparison_perpendicular(
        sample_walls, perpendicular_walls, patch_size):
    """Test form factor calculation for perpendicular walls."""
    wall_source = sample_walls[perpendicular_walls[0]]
    wall_receiver = sample_walls[perpendicular_walls[1]]
    
    patch_1 = radiosity.Patches(wall_source, patch_size, [1], 0)
    patch_2 = radiosity.Patches(wall_receiver, patch_size, [0], 1)
    patches = [patch_1, patch_2]
    patch_1.calculate_form_factor(patches)
    old_ff = patch_1.form_factors
    patch_1.calculate_univ_form_factor(patches)
    assert 100*abs(np.concatenate(patch_1.form_factors).sum()-np.concatenate(old_ff).sum()) / np.concatenate(old_ff).sum() < 10



@pytest.mark.parametrize('l', [
    .1,.2,.5,1,2
    ])
@pytest.mark.parametrize('source', [
    SoundSource(position=np.array([1,7,1]), view=np.array([0,-1,0]),
            up=np.array([0,0,1])),
    ])
@pytest.mark.parametrize('receiver', [

    Receiver(position=np.array([1,1,1]), view=np.array([0,-1,0]),
            up=np.array([0,0,1])),
    ])
@pytest.mark.parametrize('patchsize', [
    .1
    ])
def test_point_surface_interactions(l, source, receiver, patchsize):

    absor_factor = .1

    patch_pos = geo.Polygon(points=[[0,0,0],[l, 0, 0],[l, 0, l],[0,0,l]], normal=[0,1,0], up_vector=[1,0,0])

    patch = Patches(polygon=patch_pos, max_size=patchsize*l, other_wall_ids=[], wall_id=[0], absorption=absor_factor)

    patch.init_energy_exchange( 0, .1, source)

    patch = source_cast(src=source, rpatch=patch, absor=absor_factor)

    receiver_cast(receiver, patch, absor_factor)


def source_cast(src, rpatch, absor):
    """Test initial energy cast from a point to a generalized patch in space
   Nusselt-analog-based option"""

    t0 = time.time()
    nuss = form_factor.pt_solution(point=src.position, patch_points=rpatch.pts)
    tf_nusselt = time.time()-t0

    true = sum(rpatch.E_matrix[rpatch.E_matrix!=0])

    rel_error_nuss = abs(true - nuss*(1-absor))/true * 100

    assert rel_error_nuss < 1.

    print("nusselt approach error: " + str(rel_error_nuss) + "%")
    print("nusselt approach runtime: " + str(tf_nusselt*1000) + "ms \n #################################")
    return rpatch
    

def receiver_cast(rcv, patch, absor):
    """Test final energy cast from a generalized patch in space to a point
    Nusselt-analog-based option"""

    true_rec_energy = np.sum(patch.energy_at_receiver(receiver=rcv, max_order=0, ir_length_s=0.1))

    patch_energy = np.sum(patch.E_matrix)

    t0 = time.time()
    nuss = form_factor.pt_solution(point=rcv.position, patch_points=patch.pts, mode='receiver') * patch_energy
    tf_nusselt = time.time()-t0

    rel_error_nuss = abs(true_rec_energy - nuss)/true_rec_energy * 100

    assert rel_error_nuss < 1.

    print("nusselt approach error: " + str(rel_error_nuss) + "%")
    print("nusselt approach runtime: " + str(tf_nusselt*1000) + "ms \n #################################")
