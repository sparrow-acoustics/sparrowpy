"""Test bdrf interpolation."""
import numpy as np
import numpy.testing as npt
import pytest
import os
import pyfar as pf

import sparapy as sp


from sparapy.radiosity_fast import geometry as geom


@pytest.mark.parametrize('posh', [
    [1.,0.,0.]])
    # [1.,1.,0.],
    # [0.,1.,0.],
    # [-1.,1.,0.],
    # [-1.,0.,0.],
    # [-1.,-1.,0.],
    # [0.,-1.,0.],
    # [1.,-1.,0.],
    # ])
@pytest.mark.parametrize('posj', [
    [1.,0.,0.],
    [1.,1.,0.],
    [0.,1.,0.],
    [-1.,1.,0.],
    [-1.,0.,0.],
    [-1.,-1.,0.],
    [0.,-1.,0.],
    [1.,-1.,0.],
    ])
def test_brdf_intp(sample_walls, posh, posj, sampling=1):
    radi = sp.DRadiosityFast.from_polygon(sample_walls, 0.2)

    data, sources, receivers = special_scattering(sampling)
    radi.set_wall_scattering(
    np.arange(len(sample_walls)), data, sources, receivers)

    posi=np.array([0.,0.,0.])

    if posh is None:
        posh = np.array([0.,-1.,1.321458765876])
    if posj is None:
        posj = np.array([1.,0.,0.])

    posh /= np.linalg.norm(posh)
    posj /= np.linalg.norm(posj)

    sc_factor=geom.get_scattering_data_dist(pos_h=posh, pos_i=posi, pos_j=posj, sources=radi._sources, receivers=radi._receivers, wall_id_i=0, scattering=radi._scattering, scattering_index=radi._scattering_index, mode="inv_dist")

    true_factor = 2*posj[0]

    err = np.abs(sc_factor-true_factor)
    rel_err = np.abs(err/true_factor) *100

    if true_factor!=0:
        assert rel_err[0] < 5
    else:
        assert err[0] < 10**-6


    


def special_scattering(sampling):
    dist = pf.samplings.sph_equal_angle(sampling)
    dist = dist[dist.z>0]
    sources = dist.copy()
    receivers = dist.copy()
    frequencies = np.array([1000.])
    data = np.ones((sources.csize, receivers.csize, frequencies.size)) 
    data = np.multiply(data[:,:,0],2*receivers.x)
    data = data[...,np.newaxis]
    return (pf.FrequencyData(data, frequencies), sources, receivers)