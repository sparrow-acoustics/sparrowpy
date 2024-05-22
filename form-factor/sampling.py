import numpy as np
import matplotlib.pyplot as plt
from elmt import elmt

def sample_random(el=elmt([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]), npoints=10):

    npoints = npoints**2

    ptlist=np.zeros((npoints,3)) 
    
    u = el.pt[1]-el.pt[0]
    v = el.pt[-1]-el.pt[0]

    if len(el.pt)==3 or np.inner(u,v)==0:

        for i in range(npoints):
            s = np.random.uniform()
            t = np.random.uniform()

            inside = s+t <= 1

            if len(el.pt)==3 and not inside:
                s = 1-s
                t = 1-t
            
            ptlist[i] = s*u + t*v + el.pt[0]

    return np.unique(ptlist, axis=0)

def sample_regular(el=elmt([[-1.,-1.,1.],[1.,-1.,1.],[1.,1.,1.],[-1.,1.,1.]]), npoints=10):
    
    
    u = el.pt[1]-el.pt[0]
    v = el.pt[-1]-el.pt[0]

    npointsz=int(np.linalg.norm(v)/np.linalg.norm(u)*npoints)
    if npointsz==0:
        npointsz = 1
    ptlist=np.empty((npoints,npointsz,3)) 

    tt = np.linspace(0,1,npoints, endpoint=False)
    sstep =  1/(npoints*2)
    tt += sstep

    tz = np.linspace(0,1,npointsz, endpoint=False)
    sstepz =  1/(npointsz*2)
    tz += sstepz

    if len(el.pt)==3 or np.inner(u,v)==0:

        for i,s in enumerate(tt):

            for j,t in enumerate(tz):

                inside = s+t <= 1

                if not(len(el.pt)==3 and not inside):
                    ptlist[i,j] = s*u + t*v + el.pt[0]

    return np.unique(ptlist.reshape(-1,3), axis=0)


def sample_border(el,n_div=2):

    seg = np.empty((len(el.pt)*n_div,len(el.pt[0])))
    conn=[[[] for i in range(n_div+1)] for j in range(len(el.pt))]

    for i in range(len(el.pt)):

        conn[i][0]= (i*n_div)%(n_div*len(el.pt))
        conn[i][-1]= (i*n_div+n_div)%(n_div*len(el.pt))

        for ii in range(0,n_div):

            seg[i*n_div+ii,:]= (el.pt[i] + ii*(el.pt[(i+1)%len(el.pt)]-el.pt[i])/n_div)

            conn[i][ii]=(i*n_div+ii)%(n_div*len(el.pt))


    return seg,conn