"""Test bdrf interpolation."""
import numpy as np
import numpy.testing as npt
import pytest
import os
import pyfar as pf
import sparapy as sp

import matplotlib.pyplot as plt
import sparapy.radiosity_fast.geometry as geo
import sparapy.plot as plot


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
    wallid=2
    data, sources, receivers = dist
    radi.set_wall_scattering(
    np.arange(len(sample_walls)), data, sources, receivers)

    posi=radi.walls_center[wallid]

    posh = posh / np.linalg.norm(posh, axis=-1) + posi
    posj = posj / np.linalg.norm(posj, axis=-1) + posi

    sc_factor=geom.get_scattering_data_dist(pos_h=posh, pos_i=posi, pos_j=posj, i_normal=radi.walls_normal[wallid],
                                            i_up=radi.walls_up_vector[wallid], sources=radi._sources, receivers=radi._receivers, 
                                            wall_id_i=wallid, scattering=radi._scattering, 
                                            scattering_index=radi._scattering_index, mode="inv_dist")

    true_factor = np.sqrt((posj-posi)[1]**2 + (posh-posi)[0]**2)
    #true_factor = (posh-posi)[0]

    err = np.abs(sc_factor-true_factor)*100
    assert err[0] < 5
    

def test_brdf_intp_measured(sample_walls, mdist):
    radi = sp.DRadiosityFast.from_polygon(sample_walls, 1)

    wallid=2

    dist, ang, sigma = mdist

    data, sources, receivers = dist
    
    if np.iscomplexobj(data.freq):
        data.freq = data.freq.real.astype(np.float64)
    
    radi.set_wall_scattering(
    np.arange(len(sample_walls)), data, sources, receivers)

    if sigma == 0:
        sig = "0"
    elif sigma ==1:
        sig = "1"
    else:
        sig = str(sigma)

    true_factors, tsour, trec = pf.io.read_sofa("testss\\brdf_examples\\brdf_s"+sig+"_5.sofa")

    if np.iscomplexobj(true_factors):
        true_factors = true_factors.freq.real.astype(np.float64)

    posi=radi.walls_center[wallid]

    sc_factors = np.zeros([tsour.x.shape[0],trec.x.shape[0],data.freq.shape[-1]])
    tf = np.zeros([tsour.x.shape[0],trec.x.shape[0],data.freq.shape[-1]])

    for i,posh in enumerate(tsour.cartesian):
        posh = posh / np.linalg.norm(posh, axis=-1) + posi
        for j,posj in enumerate(trec.cartesian):
            posj = posj / np.linalg.norm(posj, axis=-1) + posi

            sc_factors[i,j,:]=geom.get_scattering_data_dist(pos_h=posh, pos_i=posi, pos_j=posj, 
                                                    i_normal=radi.walls_normal[wallid], i_up=radi.walls_up_vector[wallid],
                                                    sources=radi._sources, receivers=radi._receivers, 
                                                    wall_id_i=2, scattering=radi._scattering, 
                                                    scattering_index=radi._scattering_index, mode="inv_dist")
            tf[i,j,:] = true_factors.freq[i,j,:].real

    err = abs(sc_factors-tf).flatten()
    mean_err = np.mean(err)
    rms_err = np.sqrt(np.mean(err**2))
    max_err = max(err)
    min_err = min(err)
    
    fig,ax = plt.subplots(1,3)
    im=ax[0].imshow(tf, cmap='jet', interpolation='nearest')
   # plt.colorbar()
    ax[0].set_title("true factors")
    im=ax[1].imshow(sc_factors, cmap='jet', interpolation='nearest')
   # plt.colorbar()
    ax[1].set_title("estimation")
    im=ax[2].imshow( abs(sc_factors-tf), cmap='jet', interpolation='nearest')
    #plt.colorbar()
    ax[2].set_title("error")
    ax[0].set_xlim([0,250])
    ax[0].set_ylim([0,250])
    ax[1].set_xlim([0,250])
    ax[1].set_ylim([0,250])
    ax[2].set_xlim([0,250])
    ax[2].set_ylim([0,250])
    
    fig.colorbar(im, ax=ax.ravel().tolist())
    
    plt.savefig(str(sigma)+"_"+str(ang)+".png")
    plt.show()

    assert mean_err < 0.01
    assert rms_err < 0.01
    assert max_err < 0.05
    assert min_err < 10**-6
 
 
@pytest.mark.parametrize('method', [
    #["nneighbor",0],
    ["inv_dist",1],
    ])
@pytest.mark.parametrize('samp', [
    30,10#,9
    ])        
def test_bdrf_energy_conservation(sample_walls, mdist, method,samp):
    radi = sp.DRadiosityFast.from_polygon(sample_walls, 1)

    wallid=2

    dist, ang, sigma = mdist

    data, sources, receivers = dist
    
    if np.iscomplexobj(data.freq):
        data.freq = data.freq.real.astype(np.float64)
    
    radi.set_wall_scattering(
    np.arange(len(sample_walls)), data, sources, receivers)

    if sigma == 0:
        sig = "0"
    elif sigma ==1:
        sig = "1"
    else:
        sig = str(sigma)

    tsour = pf.samplings.sph_equal_angle(delta_angles=samp)
    tsour= tsour[tsour.z>0]
    trec = tsour

    posi=radi.walls_center[wallid]

    sc_factors = np.zeros([tsour.x.shape[0],trec.x.shape[0],data.freq.shape[-1]])

    src = np.array([radi._sources[wallid].azimuth[:],radi._sources[wallid].elevation[:]]).transpose()
    rec = np.array([radi._receivers[wallid].azimuth[:],radi._receivers[wallid].elevation[:]]).transpose()

    for i,posh in enumerate(tsour.cartesian):
        posh = (posh + posi)
        for j,posj in enumerate(trec.cartesian):
            posj = (posj + posi)

            sc_factors[i,j,:]=geom.get_scattering_data_dist(pos_h=posh, pos_i=posi, pos_j=posj, 
                                                    i_normal=radi.walls_normal[wallid], i_up=radi.walls_up_vector[wallid],
                                                    sources=src, receivers=rec, 
                                                    wall_id_i=2, scattering=radi._scattering, 
                                                    scattering_index=radi._scattering_index, mode=method[0], order=method[1])


    src_vis_id = 0
    
    energy_in = np.mean(data.freq)
    energy_out = np.mean(sc_factors)

    rel=np.abs(energy_in-energy_out)/energy_in  

    # fig, ax0 = plt.subplots(subplot_kw={"projection": "3d"})
    # plot.brdf_3d(data=data.freq[src_vis_id,:,0].real, receivers=radi._receivers[wallid], source_pos=radi._sources[wallid][src_vis_id], ax=ax0)
    # ax0.set_title("true factors\n sig="+str(sigma)+"; ang="+str(ang))
    # plt.savefig(str(sigma)+"_"+str(ang)+"_3d_src.png")
    
    fig,ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    plot.brdf_3d(data=sc_factors[src_vis_id,:,0], receivers=trec, source_pos=tsour[src_vis_id], ax=ax1)
    ax1.set_title("estimation\n sig="+str(sigma)+"; ang="+str(ang)+"; samp="+str(samp)+"; method="+method[0])
    plt.gcf().text(0.5, 0.02, "rel error: "+str(rel)+"\n est. energy: " + str(energy_out)+"\n tru energy: " + str(energy_in), fontsize=12)
    plt.savefig(str(sigma)+"_"+str(ang)+"_"+str(samp)+"_"+method[0]+"_3d_est.png")
    
    plt.show()

    assert rel < .01
           
@pytest.mark.parametrize('elev', [
    0,5,10,20,30,45,60,75,80,85,90
    ])     
@pytest.mark.parametrize('azi', [
    0,10,30,45,60,90,105,120,135,150,176,180,200,260,270,300,310,345,350
    ])   
def test_angle_from_point(elev, azi):    
           
    elev *= np.pi/180
    azi *= np.pi/180
           
    pt = np.array([np.cos(azi)*np.cos(elev), np.sin(azi)*np.cos(elev), np.sin(elev)])
    
    angs = geo.get_relative_angles(point=pt, origin=np.array([0.,0.,0.]), normal=np.array([0.,0.,1.]), up=np.array([1.,0.,0.]))
    
    npt.assert_almost_equal(azi, angs[0])
    npt.assert_almost_equal(elev, angs[1])

           
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


