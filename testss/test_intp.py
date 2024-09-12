"""Test bdrf interpolation."""
import numpy as np
import numpy.testing as npt
import pytest
import os
import pyfar as pf
import sparapy as sp

import matplotlib.pyplot as plt


from sparapy.radiosity_fast import geometry as geom

def special_scattering(sampling):
    dist = pf.samplings.sph_equal_area(n_points=2*sampling)
    dist = dist[dist.z>0]
    sources = dist.copy()
    receivers = dist.copy()
    frequencies = np.array([1000.])
    data = np.ones((sources.csize, receivers.csize, frequencies.size)) 

    for i in range(sources.csize):
        for j in range(receivers.csize):
            data[i,j,0] *= np.sqrt(receivers.y[j]**2 + sources.x[i]**2)
            #data[i,j,0] *= sources.x[i]


    return (pf.FrequencyData(data, frequencies), sources, receivers)


@pytest.mark.parametrize('dist', [
    special_scattering(1000),
    ])
def test_brdf_intp_theoretical(sample_walls, posh, posj, dist):
    radi = sp.DRadiosityFast.from_polygon(sample_walls, 1)

    data, sources, receivers = dist
    radi.set_wall_scattering(
    np.arange(len(sample_walls)), data, sources, receivers)

    posi=radi.walls_center[2]

    posh = posh / np.linalg.norm(posh, axis=-1) + posi
    posj = posj / np.linalg.norm(posj, axis=-1) + posi

    sc_factor=geom.get_scattering_data_dist(pos_h=posh, pos_i=posi, pos_j=posj, 
                                            sources=radi._sources, receivers=radi._receivers, 
                                            wall_id_i=2, scattering=radi._scattering, 
                                            scattering_index=radi._scattering_index, mode="inv_dist")

    true_factor = np.sqrt((posj-posi)[1]**2 + (posh-posi)[0]**2)
    #true_factor = (posh-posi)[0]

    err = np.abs(sc_factor-true_factor)*100
    assert err[0] < 5
    

def test_brdf_intp_measured(sample_walls, mdist):
    radi = sp.DRadiosityFast.from_polygon(sample_walls, 1)

    dist, ang, sigma = mdist

    data, sources, receivers = dist
    
    if np.iscomplexobj(data.freq):
        data.freq = data.freq.real.astype(np.float64)
    
    radi.set_wall_scattering(
    np.arange(len(sample_walls)), data, sources, receivers)

    true_factors, tsour, trec = pf.io.read_sofa("testss\\brdf_examples\\brdf_s"+str(sigma)+"_10.sofa")

    if np.iscomplexobj(true_factors):
        true_factors = true_factors.freq.real.astype(np.float64)

    posi=radi.walls_center[2]

    sc_factors = np.zeros([tsour.x.shape[0],trec.x.shape[0],data.freq.shape[-1]])
    tf = np.zeros([tsour.x.shape[0],trec.x.shape[0],data.freq.shape[-1]])

    for i,posh in enumerate(tsour.cartesian):
        posh = posh / np.linalg.norm(posh, axis=-1) + posi
        for j,posj in enumerate(trec.cartesian):
            posj = posj / np.linalg.norm(posj, axis=-1) + posi

            sc_factors[i,j,:]=geom.get_scattering_data_dist(pos_h=posh, pos_i=posi, pos_j=posj, 
                                                    sources=radi._sources, receivers=radi._receivers, 
                                                    wall_id_i=2, scattering=radi._scattering, 
                                                    scattering_index=radi._scattering_index, mode="inv_dist")
            tf[i,j,:] = true_factors.freq[i,j,:].real

    err = abs(sc_factors-tf).flatten()
    mean_err = np.mean(err)
    rms_err = np.sqrt(np.mean(err**2))
    max_err = max(err)
    min_err = min(err)

    assert mean_err < 0.01
    assert rms_err < 0.01
    assert max_err < 0.05
    assert min_err < 10**-6
           
           
           
# if i%round(sources.csize/5):
# fig= plt.figure(figsize=plt.figaspect(0.5))

# ax = fig.add_subplot(1, 1, 1, projection='3d')

# # Plot the surface.
# surf = ax.plot_trisurf(data[i,0::50,0]*receivers.x[0::50], data[i,0::50,0]*receivers.y[0::50], data[i,0::50,0]*receivers.z[0::50], 
#                     linewidth=0, antialiased=False)

# surf.set_fc(plt.get_cmap('jet')(data[i,0::50,0]/max(data[i,:,0])))

# # Add a color bar which maps values to colors.
# plt.colorbar(surf, shrink=0.5, aspect=5, ax=ax)

# ax.set_zlim(0.,1.1)

# plt.savefig("tests\\test_data\\dist"+str(i)+".png")


