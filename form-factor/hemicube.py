#from form_factor import elmt
import numpy as np
import matplotlib.pyplot as plt
from elmt import elmt
from math import pi as PI
from sampling import sample_regular
import geom







# f,a=plt.subplots(1,2,subplot_kw={'projection': '3d'})

# a[0].plot(ELi.pt[:,0], ELi.pt[:,1], ELi.pt[:,2], 'k--')
# a[0].plot(ELj.pt[:,0], ELj.pt[:,1], ELj.pt[:,2], 'g')


# a[1].plot(elj.pt[:,0], elj.pt[:,1], elj.pt[:,2], 'b')
# plt.show()

# print("end")



## HIPERCUBE DEFINITION

class hemicube:

    def __init__(self):

        self.top   = elmt([[-1.,-1.,1.],[1.,-1.,1.],[1.,1.,1.],[-1.,1.,1.]])
        self.north = elmt([[1.,1.,0.],[-1.,1.,0.],[-1.,1.,1.],[1.,1.,1.]])
        self.west  = elmt([[-1.,1.,0.],[-1.,-1.,0.],[-1.,-1.,1.],[-1.,1.,1.]])
        self.south = elmt([[-1.,-1.,0.],[1.,-1.,0.],[1.,-1.,1.],[-1.,-1.,1.]])
        self.east  = elmt([[1.,-1.,0.],[1.,1.,0.],[1.,1.,1.],[1.,-1.,1.]])

        self.surf = np.array([self.top,self.north,self.west,self.south,self.east])



def hemicube_form_factor(p=np.array([0,0,0]), is_sidesurf=False):

    fifi = 1/(PI*(np.sum(np.square(p)))**2)

    if is_sidesurf:
        fifi *= p[2]

    return fifi

####################################
ELi = elmt([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
ELj = elmt([[0.,3.,1.2],[3.,3.,.6],[3.,0.,1.2]])



def hemicube_estimation(ELi, ELj, nsamples = 3, resolution=10, plot=False):
    hh = hemicube()
    

    ffactor = 0
    aA=[]

    outpointlist = sample_regular(el=ELi, npoints=nsamples, plot=False)

    for outpoint in outpointlist:

        elj = elmt(geom.universal_transform(outpoint, ELi.n, ELj.pt))

        if plot:
            fig=plt.figure()
            ax=fig.add_subplot(projection='3d')
            ax.axis('equal')

            ax=plot_elmt(el=elj,ax=ax,style='r-')

        for id,surface in enumerate(hh.surf):

            plist = sample_regular(el=surface,npoints=resolution,plot=False)

            aA.append(surface.A/len(plist))

            if plot:
                plot_elmt(el=surface,ax=ax)

            for pt in plist:

                if plot:
                    ax.plot(pt[0],pt[1],pt[2],'r.')

                intsec = geom.vec_plane_intersection(v0=pt, n=elj.n, pn=elj.o)

                if (intsec is not None) and geom.point_in_polygon(intsec,elj):
                    ffactor += hemicube_form_factor(p=pt, is_sidesurf=bool(id)) * surface.A / len(plist) * ELi.A / len(outpointlist)

                    if plot:
                        #ax.plot(pt[0],pt[1],pt[2],'o', color=[0,1,0])
                        ax.plot(intsec[0],intsec[1],intsec[2],'b.')
                        ax.plot([0,intsec[0]],[0,intsec[1]],[0,intsec[2]],'g-')


        dA = np.mean(aA)

    if plot:
        ax.set_xlim([-3,3])
        ax.set_ylim([-3,3])
        ax.set_zlim([0,3])
        plt.show()

    return ffactor#,dA

#hemicube_estimation(ELi, ELj,resolution=50,plot=True)
#print('kkk')