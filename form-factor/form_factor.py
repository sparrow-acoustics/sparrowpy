import numpy as np
import matplotlib.pyplot as plt
from math import pi as PI
from elmt import elmt
import time
from nusselt import nusselt
import geom
import sampling
from integrate_line import poly_estimation,poly_integration
import exact_solutions as exact 
from geometry import Polygon as polyg

#######################################################################################
### form functions
#######################################################################################

def ffunction(p0,p1,n0,n1):
    cos0=geom.vec_cos(p1-p0, n0)
    cos1=geom.vec_cos(p0-p1,n1)
    rsq = np.square(np.linalg.norm(p0-p1))

    if rsq != 0:
        return(cos0*cos1/(rsq*PI))
    else:
        return 0
    
def stokes_ffunction(p0,p1):
    n = np.linalg.norm(p1-p0)

    if n>0:
        return np.log(n)
    else: # handle singularity (likely unused)
        return None

#######################################################################################
### integration
#######################################################################################

def naive_integration(patch_i: polyg, patch_j: polyg, n_samples=4, random=False):
    """
    calculate an estimation of the form factor between two patches 
    by computationally integrating the form function over the two patch surfaces

    The method is called naïve because it consists on the simplest approximation:
    the form function value is merely multiplied by the area of finite sub-elements of each patch's surface.

    Parameters
    ----------
    patch_i : geometry.Polygon
        radiance receiving patch

    patch_j : geometry.Polygon
        radiance emitting patch

    n_samples : int
        number of surface function samples on each patch 
        TO DO: convert to sample density factor (large and small patches have approx. same resolution)

    random: bool
        determines whether the form function is sampled at regular intervals (False) or randomly (uniform distribution)
        over the patches' surfaces.

    """

    if random:
        surfsampling = sampling.sample_random
    else:
        surfsampling = sampling.sample_regular

    int_accum = 0. # initialize integral accumulator variable

    i_samples = surfsampling(patch_i, npoints=n_samples)
    j_samples = surfsampling(patch_j, npoints=n_samples)

    for i_pt in i_samples:
        for j_pt in j_samples:
            int_accum+= ffunction(i_pt, j_pt, patch_i.normal, patch_j.normal)*(patch_i.A/len(i_samples))*(patch_j.A/len(j_samples))
  

    return int_accum/patch_i.A


def stokes_integration(patch_i: polyg, patch_j: polyg, approx_order=2):
    """
    calculate an estimation of the form factor between two patches 
    by computationally integrating a modified form function over the two patch boundaries.
    The modified form function follows Stokes' theorem.

    The modified form function integral is calculated using a polynomial approximation based on sampled values.
    
    Parameters
    ----------
    patch_i : geometry.Polygon
        radiance receiving patch

    patch_j : geometry.Polygon
        radiance emitting patch

    approx_order: int
        determines the order of the polynomial integration order. 
        also determines the number of samples in each patch's boundary.

    """

    i_bpoints, i_conn = sampling.sample_border(patch_i, npoints=approx_order+1)
    j_bpoints, j_conn = sampling.sample_border(patch_j, npoints=approx_order+1)

    if singularity_check(i_bpoints,j_bpoints): 
        return float('nan')


    # first compute and store form function sample values
    form_mat = [[[] for j in range(len(j_bpoints))] for i in range(len(i_bpoints))]

    for j,pj in enumerate(j_bpoints):
        for i,pi in enumerate(i_bpoints):
            form_mat[i][j] = stokes_ffunction(pi,pj)


    # double polynomial integration (per dimension (x,y,z))
    outer_integral = 0
    inner_integral = np.zeros(shape=[len(i_bpoints),len(j_bpoints[0])])

    for dim in range(len(j_bpoints[0])):                                # for each dimension


        # integrate form function over each point on patch i boundary

        for i in range(len(i_bpoints)):                                 # for each point in patch i boundary
            for segj in j_conn:                                         # for each segment segj in patch j boundary
                
                xj = j_bpoints[segj][:,dim]          

                if xj[-1]-xj[0]!=0:
                    quadfactors = poly_estimation(xj,[form_mat[i][segj[k]] for k in range(len(segj))] ) # compute polynomial coefficients of approx form function over boundary segment xj
                    inner_integral[i][dim] += poly_integration(quadfactors,xj)                          # analytical integration of the approx polynomial


        # integrate previously computed integral over each boundary segment of patch i

        for segi in i_conn:                     # for each segment segi in patch i boundary

            xi = i_bpoints[segi][:,dim]

            if xi[-1]-xi[0]!=0:
                quadfactors = poly_estimation(xi,[inner_integral[segi[k]][dim] for k in range(len(segi))] ) 
                outer_integral += poly_integration(quadfactors,xi)


    return outer_integral/(2*PI*patch_i.A)

#######################################################################################
### helper
#######################################################################################

def singularity_check(p0,p1):
    """
    returns true if two patches have any common points
    """

    s0 = {tuple(row) for row in p0}
    s1 = {tuple(row) for row in p1}

    for el in s1:
        if el in s0:
            return True

    return False

















        

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
            tit = "Random patch -> " + str(len(el.pts)) + " sides"

        print("\n########################################\n"+tit + "\n")
       
        samplingsteps = [2**2,5**2,7**2,10**2,14**2,20**2]

        for step in samplingsteps:

            t0 = time.time()
            temp= nusselt(el0,el, nsamples=step)
            t_cnuss.append(time.time()-t0)
            ff_cnuss.append(temp)


        for order in [2,3,4,5]:

            t0 = time.time()
            ff_stokes = stokes_integration(el0, el, approx_order=order)
            t_stokes = time.time()-t0

            tt_stokes.append(t_stokes)
            tf_stokes.append(ff_stokes)


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
        #elif i == 3:
        #    true_solution = exact.perpendicular_patch_floating(1.,1.,1.,0.,1.)
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

        plt.savefig("C:\\Users\\fatela\\Desktop\\temp\\patch"+str(i+1))
        #plt.show()
        print("yo")
    
el0 = polyg(points=np.array([[0.,0.,0.],[0.,1.,0.],[0.,1.,1.],[0.,0.,1.]]), up_vector=np.array([0.,0.,1.]), normal=np.array([1.,0.,0.]))

el1 = polyg(points=np.array([[1.,0.,0.],[1.,0.,1.],[1.,1.,1.],[1.,1.,0.]]), up_vector=np.array([0.,0.,1.]), normal=np.array([-1.,0.,0.]))

el2 = polyg(points=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]]), up_vector=np.array([-1.,0.,0.]), normal=np.array([0.,0.,1.]))

el3 = polyg(points=np.array([[0.,1.,0.],[1.,1.,0.],[1.,2.,0.],[0.,2.,0.]]), up_vector=np.array([-1.,0.,0.]), normal=np.array([0.,0.,1.]))

el4 = polyg(points=np.array([[0.,0.,-1.],[1.,0.,-1.],[1.,1.,-1.],[0.,1.,-1.]]), up_vector=np.array([-1.,0.,0.]), normal=np.array([0.,0.,1.]))


ell5 = elmt([[1.,0.,0.],[2.,0.,1.],[2.,1.,1.],[1.,1.,0.]])

ell6 = elmt([[4.,0.,0.],[4.,0.,2.],[3.,2.,1.]])

el5 = polyg(points=ell5.pt, up_vector=ell5.pt[0]-ell5.pt[-1], normal=ell5.n)

el6 = polyg(points=ell6.pt, up_vector=ell6.pt[0]-ell6.pt[-1], normal=ell6.n)



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