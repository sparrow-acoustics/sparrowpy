import sparapy.form_factor as form_factor
import pytest
import sparapy.geometry as geo
import numpy as np
import time
from sparapy.sound_object import SoundSource
from sparapy.radiosity import Patches

@pytest.mark.parametrize('l', [
    .1,.2,.5,1,2,5
    ])
@pytest.mark.parametrize('src', [
    SoundSource(position=np.array([1,1,1]), view=np.array([0,-1,0]),
            up=np.array([0,0,1]))
    ])
@pytest.mark.parametrize('patchsize', [
    .1
    ])
@pytest.mark.parametrize('absor', [
    0.1, .2, .5
    ])
def test_naive_source_cast(l, src, patchsize, absor):
    """Test universal form factor calculation for facing parallel patches of equal dimensions"""

    patch_pos = geo.Polygon(points=[[0,0,0],[l, 0, 0],[l, 0, l],[0,0,l]], normal=[0,1,0], up_vector=[1,0,0])

    rpatch = Patches(polygon=patch_pos, max_size=patchsize*l, other_wall_ids=[], wall_id=[0], absorption=absor)
    

    t0 = time.time()
    naive = form_factor.naive_pt_integration(pt=src.position, patch=patch_pos, n_samples=100)
    tf = time.time()-t0

    rpatch.nbins = 1
    rpatch.init_energy_exchange(source=src, max_order_k=0, ir_length_s=.5)

    rel_error = abs(sum(rpatch.E_matrix[rpatch.E_matrix!=0]) - naive*(1-absor))/sum(rpatch.E_matrix[rpatch.E_matrix!=0])
    
    assert tf<0.01
    assert rel_error < 1./100

@pytest.mark.parametrize('l', [
    .1,.2,.5,1,2,5
    ])
@pytest.mark.parametrize('src', [
    SoundSource(position=np.array([1,1,1]), view=np.array([0,-1,0]),
            up=np.array([0,0,1]))
    ])
@pytest.mark.parametrize('patchsize', [
    .1
    ])
@pytest.mark.parametrize('absor', [
    0.1, .2, .5
    ])
def test_stokes_source_cast(l, src, patchsize, absor):
    """Test universal form factor calculation for facing parallel patches of equal dimensions"""

    patch_pos = geo.Polygon(points=[[0,0,0],[l, 0, 0],[l, 0, l],[0,0,l]], normal=[0,1,0], up_vector=[1,0,0])

    rpatch = Patches(polygon=patch_pos, max_size=patchsize*l, other_wall_ids=[], wall_id=[0], absorption=absor)
    
    t0 = time.time()
    stokes = form_factor.stokes_pt_integration(point=np.array([src.position]), patch=patch_pos.pts, source_area=patch_pos.area)
    tf = time.time()-t0

    rpatch.nbins = 1
    rpatch.init_energy_exchange(source=src, max_order_k=0, ir_length_s=.5)

    true = sum(rpatch.E_matrix[rpatch.E_matrix!=0])

    rel_error = abs(true - stokes*(1-absor))/true

    assert tf<0.0015
    assert rel_error < .5/100

@pytest.mark.parametrize('l', [
    .1,.2,.5,1,2,3,4,5,6
    ])
@pytest.mark.parametrize('src', [
    SoundSource(position=np.array([1,7,1]), view=np.array([0,-1,0]),
            up=np.array([0,0,1]))
    ])
@pytest.mark.parametrize('patchsize', [
    .1
    ])
@pytest.mark.parametrize('absor', [
    0.1, .2, .5
    ])
def test_nusselt_source_cast(l, src, patchsize, absor):
    """Test universal form factor calculation for facing parallel patches of equal dimensions"""

    patch_pos = geo.Polygon(points=[[0,0,0],[l, 0, 0],[l, 0, l],[0,0,l]], normal=[0,1,0], up_vector=[1,0,0])

    rpatch = Patches(polygon=patch_pos, max_size=patchsize*l, other_wall_ids=[], wall_id=[0], absorption=absor)
    
    t0 = time.time()
    nuss = form_factor.nusselt_pt_solution(point=src.position, patch_points=patch_pos.pts)
    tf = time.time()-t0

    rpatch.nbins = 1
    rpatch.init_energy_exchange(source=src, max_order_k=0, ir_length_s=.5)

    true = sum(rpatch.E_matrix[rpatch.E_matrix!=0])

    rel_error = abs(true - nuss*(1-absor))/true

    assert tf<0.0015
    assert rel_error < .2/100
    

@pytest.mark.parametrize('rpatch', [
    geo.Polygon(points=[[0,0,0],[1, 0, 0],[1, 0, 1],[0,0,1]], normal=[0,1,0], up_vector=[1,0,0])
    ])
@pytest.mark.parametrize('src', [
    np.array([.5,.5,.5])
    ])
@pytest.mark.parametrize('nsamples', [
    49,64,81,100
    ])
def test_receiver(rpatch,src,nsamples,baseline):
    """Test universal form factor calculation for facing parallel patches of equal dimensions"""

    t0 = time.time()
    naive = form_factor.naive_pt_integration(pt=src, patch=rpatch, n_samples=nsamples, mode='receiver')
    tf = time.time()-t0



    assert 100 * abs(baseline-naive)/baseline < .5
    assert tf < 1/100