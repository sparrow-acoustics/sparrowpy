from form_factor import naive_integration, nusselt_integration, stokes_integration, form_factor
import matplotlib.pyplot as plt
import time
import exact_solutions as exact
import numpy as np
from geometry import Polygon as polyg
from elmt import elmt

SCALE = .2

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
       
        samplingsteps = [2**2,5**2,10**2,15**2]

        for step in samplingsteps:

            t0 = time.time()
            temp= nusselt_integration(el0,el, nsamples=step)
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


        t0 = time.time()
        opt= form_factor(el,el0)
        t_opt=time.time()-t0

        true_solution = ff_naive

        if i == 0:
            true_solution = exact.parallel_patches(SCALE*1.,SCALE*1.,SCALE*1.)
        elif i == 1:
            true_solution = exact.perpendicular_patch_coincidentline(SCALE*1.,SCALE*1.,SCALE*1.)
        elif i == 2:
            true_solution = exact.perpendicular_patch_coincidentpoint(SCALE*1.,SCALE*1.,SCALE*1.,SCALE*1.)
        #elif i == 3:
        #    true_solution = exact.perpendicular_patch_floating(1.,1.,1.,0.,1.)
        else:
            true_solution = ff_naive

        

        print("True solution: "+ str(true_solution))


        a.plot(t_array, ( ( ff_n_array - true_solution)**2)**.5 / abs(true_solution), 'b-', label='double even quadrature integration')
        a.plot(t_cnuss, ( ( ff_cnuss - true_solution)**2)**.5 / abs(true_solution), 'r-', label='nusselt + even quadrature integration' )
        a.plot(tt_stokes, ( (tf_stokes - true_solution)**2)**.5 / abs(true_solution), 'g*-', label='stokes integration' )
        a.plot(t_opt, ( (opt - true_solution)**2)**.5 / abs(true_solution), 'r*', label='"best" approach' )
        a.legend()
        a.set_yscale('log')
        a.set_xscale('log')
        a.set_xlabel("computation runtime [s]")
        a.set_ylabel("relative Form Factor error (L2)")
        a.set_title(tit)
        a.grid()

        plt.savefig("C:\\Users\\Work profile\\Desktop\\temp\\patch"+str(i+1)+".png")



    
el0 = polyg(points=np.array([[0.,0.,0.],[0.,1.,0.],[0.,1.,1.],[0.,0.,1.]])*SCALE, up_vector=np.array([0.,0.,1.]), normal=np.array([1.,0.,0.]))

el1 = polyg(points=np.array([[1.,0.,0.],[1.,0.,1.],[1.,1.,1.],[1.,1.,0.]])*SCALE, up_vector=np.array([0.,0.,1.]), normal=np.array([-1.,0.,0.]))

el2 = polyg(points=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])*SCALE, up_vector=np.array([-1.,0.,0.]), normal=np.array([0.,0.,1.]))

el3 = polyg(points=np.array([[0.,1.,0.],[1.,1.,0.],[1.,2.,0.],[0.,2.,0.]])*SCALE, up_vector=np.array([-1.,0.,0.]), normal=np.array([0.,0.,1.]))

el4 = polyg(points=np.array([[0.,0.,-1.],[1.,0.,-1.],[1.,1.,-1.],[0.,1.,-1.]])*SCALE, up_vector=np.array([-1.,0.,0.]), normal=np.array([0.,0.,1.]))


ell5 = elmt([[1.,0.,0.],[4.,0.,1.],[4.,1.,1.],[1.,1.,0.]])

ell6 = elmt([[2.,0.,0.],[2.,0.,2.],[3.,2.,2.]])

el5 = polyg(points=ell5.pt*SCALE, up_vector=ell5.pt[0]-ell5.pt[-1], normal=ell5.n)

el6 = polyg(points=ell6.pt*SCALE, up_vector=ell6.pt[0]-ell6.pt[-1], normal=ell6.n)





plot_comparisons(el0,[el1, el2, el3, el4, el5, el6])

