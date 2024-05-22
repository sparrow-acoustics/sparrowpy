import numpy as np
import matplotlib.pyplot as plt
from math import pi as PI
from elmt import elmt
from hemicube import hemicube_estimation as hc
import time
from nusselt import nusselt
import geom
import sampling

from integrate_line import analytical_coincident_line_solution, analytical_coincident_point_solution, poly_estimation,poly_integration

import exact_solutions as exact 



def sample_points(el, stepabs):

    P = []

    step1=stepabs/np.linalg.norm(el.pt[1]-el.pt[0])

    step2=stepabs/np.linalg.norm(el.pt[-1]-el.pt[0])

    r1 = np.arange(0.,1.,step1)
    if 1%step1 == 0:
        r1 = np.append(r1,1.)

    for rr1 in r1:

        r2 = np.arange(0.,1.,step2)

        if 1%step2 == 0:
            r2 = np.append(r2,1.)

        for rr2 in r2:
            P.append(el.pt[0] + rr1 * (el.pt[1]-el.pt[0]) + rr2 * (el.pt[2]-el.pt[0]))

    return np.unique(np.array(P), axis=0)

# def sample_regular(el=elmt([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]), npoints=1, plot=False):
#     ptlist=np.empty((npoints,npoints,3)) 
    
#     u = el.pt[1]-el.pt[0]
#     v = el.pt[-1]-el.pt[0]

#     tt = np.linspace(0,1,npoints, endpoint=False)
#     sstep =  1/(npoints*2)
#     tt += sstep

#     if len(el.pt)==3 or np.inner(u,v)==0:

#         for i in range(npoints):

#             s = tt[i]

#             for j in range(npoints):
                
#                 t = tt[j]

#             inside = s+t <= 1

#             if not(len(el.pt)==3 and not inside):
#                 ptlist[i,j] = s*u + t*v + el.pt[0]

#     if plot:
#         fff,aaa=plt.subplots()
#         px=[]
#         py=[]
#         for j in range(len(el.pt)+1):
#             px.append(el.pt[j%len(el.pt),0])
#             py.append(el.pt[j%len(el.pt),1])
#         aaa.plot(px,py,'b-')

#         for pt in ptlist:
#             aaa.plot(pt[0],pt[1],'ro')

#         plt.show()


#     return ptlist




def form_function(p0,p1,n0,n1):
    cos0=geom.vec_cos(p1-p0, n0)
    cos1=geom.vec_cos(p0-p1,n1)
    rsq = np.square(np.linalg.norm(p0-p1))

    if rsq != 0:
        return(cos0*cos1/(rsq*PI))
    else:
        return 0
    

def naive_integration(base_el,out_el,samplestep, random=False):

    if random:
        samples0 = sampling.sample_random(base_el,samplestep)
        samples1 = sampling.sample_random(out_el,samplestep)
    else:
        samples0 = sampling.sample_regular(base_el,samplestep)
        samples1 = sampling.sample_regular(out_el,samplestep)

    

    # ff = plt.figure()
    # ax = ff.add_subplot(projection='3d')

    # ax.scatter(samples0.transpose()[0],samples0.transpose()[1],samples0.transpose()[2])
    # ax.scatter(samples1.transpose()[0],samples1.transpose()[1],samples1.transpose()[2])
    # ax.grid()
    # plt.show()

    int_accum=0.

    for basept in samples0:
        for outpt in samples1:
            int_accum+= form_function(basept, outpt, base_el.n, out_el.n)*(base_el.A/len(samples0))*(out_el.A/len(samples1))
  

    return int_accum/base_el.A

def stokes_ffunction(p0,p1):
    n = np.linalg.norm(p1-p0)

    if n>0:
        return np.log(n)
    else:
        return None


def singularity_check(p0,p1):
    s0 = {tuple(row) for row in p0}
    s1 = {tuple(row) for row in p1}

    for el in s1:
        if el in s0:
            return True

    return False

def stokes_integration(eli, elj, approx_order=2):

    si, coni = sampling.sample_border(eli, npoints=approx_order+1)
    sj, conj = sampling.sample_border(elj, npoints=approx_order+1)

    if singularity_check(si,sj):
        return float('nan')

    form_mat = [[[] for j in range(len(sj))] for i in range(len(si))]

    for j,pj in enumerate(sj):
        for i,pi in enumerate(si):
            form_mat[i][j] = stokes_ffunction(pi,pj)

    # here, we essentially do a bi-quadratic integral per dimension (x,y,z)
    outer_integral = 0
    inner_integral = np.zeros(shape=[len(si),len(sj[0])])

    for dim in range(len(sj[0])):  
        for i in range(len(si)):              # for each point in patch i boundary
            for segj in conj:                 # for each segment segj in patch j boundary
                
                # dim values for each segment points
                xj = sj[segj][:,dim]          

                if xj[-1]-xj[0]!=0:
                    # estimate quadratic coefficients (like for interpolation) of function which describes the stuff
                    quadfactors = poly_estimation(xj,[form_mat[i][segj[k]] for k in range(len(segj))] ) 
                    # integrate "by hand"
                    inner_integral[i][dim] += poly_integration(quadfactors,xj) 

        for segi in coni:                     # for each segment segi in patch i boundary
            # dim values for each segment points
            xi = si[segi][:,dim]

            if xi[-1]-xi[0]!=0:
                # estimate quadratic coefficients (like for interpolation) of function which describes the stuff
                quadfactors = poly_estimation(xi,[inner_integral[segi[k]][dim] for k in range(len(segi))] ) 
                # integrate quadrature "by hand"
                outer_integral += poly_integration(quadfactors,xi)


    return outer_integral/(2*PI*eli.A)


def monte_carlo(eli, elj, raysppoint, npts):
    import matplotlib.pylab as pllt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection # New import

    eli0 = elmt(geom.universal_transform(eli.o, eli.n, pt_list = eli.pt))
    elj0 = elmt(geom.universal_transform(eli.o, eli.n, pt_list = elj.pt))

    counter=0

    # f = pllt.figure()

    # a = f.add_subplot(111,projection='3d')

    # vertices = [(elj0.pt[i,0],elj0.pt[i,1],elj0.pt[i,2]) for i in range(len(elj0.pt))]

    # a.add_collection3d(Poly3DCollection(verts=[vertices],color='orange', alpha=0.5))
    # a.set_aspect('equalxy')

    ff=0

    plist = sampling.sample_regular(eli0, npts) 

    for point in plist:
        for i in range(raysppoint):

            phi   = np.random.uniform()*2*PI
            theta = np.random.uniform()*PI/2

            r = point + np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            intersection_point = geom.vec_plane_intersection(v0=r, p0=np.array([0,0,0]), n=elj0.n, pn=elj0.o)

            #pllt.plot([0.0,r[0]],[0.0,r[1]],[0.0,r[2]], 'red')
            #pllt.plot([r[0], intersection_point[0]],[r[1], intersection_point[1]], [r[2], intersection_point[2]], 'blue')
            #pllt.plot([intersection_point[0]],[intersection_point[1]], [intersection_point[2]], 'gx')
            

            if geom.point_in_polygon(pt=intersection_point, el=elj0):
                #pllt.plot([intersection_point[0]],[intersection_point[1]], [intersection_point[2]], 'ro')
                #ff += form_function(p0=np.array([0,0,0]),p1=elj0.o, n0=np.array([0,0,1]), n1=elj0.n)
                counter+=1

    #pllt.show()
    if counter !=0:
        return counter/(len(plist)*raysppoint)
    else:
        return 0


    
        

def plot_comparisons(el0, elements):

    for i,el in enumerate(elements):

        f,a = plt.subplots()

        ff_n_array=[]
        t_array=[]

        tt_stokes = []
        tf_stokes = []

        ff_cnuss = []
        t_cnuss = []


        if i == 0:
            tit = "Parallel patches"
        if i == 1:
            tit = "Perpendicular patches -- connected edge"
        if i == 2:
            tit = "Perpendicular patches -- connected vertex"
        if i == 3:
            tit = "Perpendicular patches -- disconnected"
        if i > 3:
            tit = "Random patch"

        print("\n########################################\n"+tit + "\n")
       
        samplingsteps = [3,6,10,15,20]

        for step in samplingsteps:

            t0 = time.time()
            temp= nusselt(el0,el, nsamples=step)
            t_cnuss.append(time.time()-t0)
            ff_cnuss.append(temp)


        for order in [1,2,3,4,5,6]:

            t0 = time.time()
            ff_stokes = stokes_integration(el0, el, approx_order=order)
            t_stokes = time.time()-t0

            tt_stokes.append(t_stokes)
            tf_stokes.append(ff_stokes)

        samplingsteps = [3,6,10,15,20]
        #ppp = 10
        for step in samplingsteps:

            print("# points: " + str(step))

            t0 = time.time()

            ff_naive = naive_integration(el0,el,step)

            t1 = time.time()-t0

            ff_n_array.append(ff_naive)
            t_array.append(t1)

        # for step in samplingsteps:

        #     print("# points: " + str(step))
            
        #     t0 = time.time()

        #     ff_naive = naive_integration(el0,el,step, random=True)

        #     t1 = time.time()-t0

        #     ff_nr_array.append(ff_naive)
        #     t_rand_array.append(t1)


            # print("hemicube : "+str(ppp) +" points")
            # t0 = time.time()
            # ff_hc.append(hc(el0, el, resolution=ppp , nsamples=step,plot=False))
            # t_hc.append(time.time()-t0)

        if i == 0:
            true_solution = exact.parallel_patches(1.,1.,1.)
        elif i == 1:
            true_solution = exact.perpendicular_patch_coincidentline(1.,1.,1.)
        elif i == 2:
            true_solution = exact.perpendicular_patch_coincidentpoint(1.,1.,1.,1.)
        elif i == 3:
            true_solution = exact.perpendicular_patch_floating(1.,1.,0.,1.,1.)
        else:
            true_solution = ff_naive

        print("True solution: "+ str(true_solution))


        a.plot(t_array, ( ( ff_n_array - true_solution)**2)**.5 / abs(true_solution), 'b-', label='double even quadrature integration')
        #a[i].plot(t_rand_array, ( ( ff_nr_array - true_solution)**2)**.5 / true_solution, 'b--', label='monte carlo integration')
        a.plot(t_cnuss, ( ( ff_cnuss - true_solution)**2)**.5 / abs(true_solution), 'r-', label='nusselt + even quadrature integration' )
        a.plot(tt_stokes, ( (tf_stokes - true_solution)**2)**.5 / abs(true_solution), 'g*-', label='stokes integration' )
        #a[i].plot(t_hc*np.ones_like(ff_hc), ( ( ff_hc - true_solution)**2)**.5 / true_solution, 'g-', label='hemicube' )
        a.legend()
        a.set_yscale('log')
        a.set_xscale('log')
        a.set_xlabel("computation runtime [s]")
        a.set_ylabel("relative Form Factor error (L2)")
        a.set_title(tit)
        a.grid()

        # a[1,i].plot(np.array(samplingsteps)**2 * 2, t_array, label='naive integration')
        # a[1,i].plot(np.array(samplingsteps)**2, t_cnuss, 'k--', label='nusselt curved' )
        # #a[1,i].plot([1,2000], np.array([1,1])*t_stokes, 'b--', label='quad approximation stokes' )
        # a[1,i].plot(npnp, t_mc, 'orange', label='monte carlo' )
        # a[1,i].plot(pointnr, t_hc, 'magenta', label='hemicube' )
        # a[1,i].legend()
        # a[1,i].set_yscale('log')
        # a[1,i].set_xscale('log')
        # a[1,i].set_xlabel("# patch sampling points")
        # a[1,i].set_ylabel("runtime [s]")
        # a[1,i].set_title(tit)

        # a[0,i].grid()
        # a[1,i].grid()

        plt.savefig("C:\\Users\\fatela\\Desktop\\temp\\patch"+str(i))

        print("yo")
    
el0 = elmt([[0.,0.,0.],[0.,1.,0.],[0.,1.,1.],[0.,0.,1.]])

el1 = elmt([[1.,0.,0.],[1.,0.,1.],[1.,1.,1.],[1.,1.,0.]])

el2 = elmt([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])

el3 = elmt([[0.,1.,0.],[1.,1.,0.],[1.,2.,0.],[0.,2.,0.]])

el4 = elmt([[0.,0.,-1.],[1.,0.,-1.],[1.,1.,-1.],[0.,1.,-1.]])

el5 = elmt([[1.,0.,0.],[2.,0.,1.],[2.,1.,1.],[1.,1.,0.]])

el6 = elmt([[1.,0.,0.],[2.,0.,1.],[2.,1.,1.],])



# f = plt.figure()
# a = f.add_subplot(projection='3d')

# a.plot(el0.pt[:,0],el0.pt[:,1], el0.pt[:,2], 'r-')

# a.plot(el4.pt[:,0],el4.pt[:,1], el4.pt[:,2], 'b-')
# a.grid()

# a.set_aspect('equal')

# plt.show()


plot_comparisons(el0,[el1, el2, el3, el4, el5, el6])

print("hehe")

# print(stokes_integration(el0, el1)-stokes_integration(el1, el0))


#print(naive_stokes_integration(el0, el1))
#print(naive_nusselt(el0, el1))