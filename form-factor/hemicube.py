import numpy as np
from elmt import elmt
from math import pi as PI
from sampling import sample_regular
import geom

class hemicube:

    def __init__(self):

        self.top   = elmt([[-1.,-1.,1.],[1.,-1.,1.],[1.,1.,1.],[-1.,1.,1.]])
        self.north = elmt([[1.,1.,0.],[-1.,1.,0.],[-1.,1.,1.],[1.,1.,1.]])
        self.west  = elmt([[-1.,1.,0.],[-1.,-1.,0.],[-1.,-1.,1.],[-1.,1.,1.]])
        self.south = elmt([[-1.,-1.,0.],[1.,-1.,0.],[1.,-1.,1.],[-1.,-1.,1.]])
        self.east  = elmt([[1.,-1.,0.],[1.,1.,0.],[1.,1.,1.],[1.,-1.,1.]])

        self.surf = np.array([self.top,self.north,self.west,self.south,self.east])



def hemicube_form_function(p=np.array([0,0,0]), is_sidesurf=False):

    fifi = 1/(PI*(np.sum(np.square(p)))**2)

    if is_sidesurf:
        fifi *= p[2]

    return fifi



def hemicube_estimation(ELi, ELj, nsamples = 3, resolution=10, plot=False):
    hh = hemicube()
    

    ffactor = 0
    aA=[]

    outpointlist = sample_regular(el=ELi, npoints=nsamples, plot=False)

    for outpoint in outpointlist:

        elj = elmt(geom.universal_transform(outpoint, ELi.n, ELj.pt))

        for id,surface in enumerate(hh.surf):

            plist = sample_regular(el=surface,npoints=resolution,plot=False)

            aA.append(surface.A/len(plist))

            for pt in plist:

                intsec = geom.vec_plane_intersection(v0=pt, n=elj.n, pn=elj.o)

                if (intsec is not None) and geom.point_in_polygon(intsec,elj):
                    ffactor += hemicube_form_function(p=pt, is_sidesurf=bool(id)) * surface.A / len(plist) * ELi.A / len(outpointlist)

        dA = np.mean(aA)



    return ffactor

#hemicube_estimation(ELi, ELj,resolution=50,plot=True)
#print('kkk')