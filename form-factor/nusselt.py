import numpy as np
from elmt import elmt
from integrate_line import poly_estimation, poly_integration
import matplotlib.pyplot as plt
import sampling
import geom

PI = np.pi


def nusselt(Pi, Pj, nsamples=2, random=False, sphRadius=1):
    if random:
        p0_array = sampling.sample_random(Pi,nsamples, random)
    else:
        p0_array = sampling.sample_regular(Pi,nsamples, random)

    out = 0
    n = 2

    segs, conn = sampling.sample_border(Pj,n_div=n)

    sphPts = np.empty_like( segs )
    plnPts = np.empty( shape=(len(segs),2) )

    for p0 in p0_array:
        curved_area = 0
 
        # project patch j vertices onto unit hemisphere
        for ii in range(len(segs)):
            sphPts[ii] = ((segs[ii]-p0)*sphRadius/np.linalg.norm(segs[ii]-p0))

        # project shpere points onto patch i plane
        for ii in range(len(sphPts)):
            plnPts[ii,:] = np.inner(geom.rotation_matrix(Pi.n),sphPts[ii])[:-1]

        projPj = elmt(plnPts[0::n])
            
        for seg in conn:

            if np.inner( plnPts[seg[-1]], plnPts[seg[0]] ) >= 0: 
                curved_area += area_under_curve(plnPts[seg],n=n)
            else:
                mpoint = sphPts[seg[0]] + (sphPts[seg[-1]] - sphPts[seg[0]]) / 2
                marc = mpoint*sphRadius/np.linalg.norm(mpoint)

                mpoint = np.inner(geom.rotation_matrix(Pi.n),mpoint)[:-1]
                marc = np.inner(geom.rotation_matrix(Pi.n),marc)[:-1]

                linArea = np.linalg.norm(plnPts[seg[-1]] - plnPts[seg[0]])*np.linalg.norm(mpoint-marc)/2
                
                a = sphPts[seg[0]] + (sphPts[seg[1]] - sphPts[seg[0]]) / 2
                a = a*sphRadius/np.linalg.norm(a)
                a = np.inner(geom.rotation_matrix(Pi.n),a)[:-1]

                b = sphPts[seg[1]] + (sphPts[seg[-1]] - sphPts[seg[1]]) / 2
                b = b*sphRadius/np.linalg.norm(b)
                b = np.inner(geom.rotation_matrix(Pi.n),b)[:-1]

                
                left =  area_under_curve(np.array([plnPts[seg[0]],a,marc]),n=n)
                right = area_under_curve(np.array([marc,b,plnPts[seg[-1]]]),n=n)
                curved_area += linArea * np.sign(left) + left + right

        out += (projPj.A + curved_area) / (sphRadius**2 * PI) * (Pi.A/len(p0_array))
       
    return out

def area_under_curve(ps, n=2):

    f  = ps[-1] - ps[0]

    rot_mat = np.array([[f[0],f[1]],[-f[1],f[0]]])/np.linalg.norm(f)

    
    ff = np.inner(rot_mat,f)

    x = np.array([0])
    y = np.array([0])

    for k in range(1,n):

        c = ps[k] - ps[0]
        cc = np.inner(rot_mat,c)
        
        x = np.append(x,cc[0])
        y = np.append(y,cc[1])

    x = np.append(x,ff[0])
    y = np.append(y,ff[1])

    coefs = poly_estimation(x,y)
    area = poly_integration(coefs,x)

    return area