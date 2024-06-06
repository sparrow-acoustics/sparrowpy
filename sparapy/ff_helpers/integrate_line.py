import numpy as np
import time
import matplotlib.pyplot as plt

def sample(p0,p1,nsamples=3):

    samples = np.empty([nsamples,len(p0)])

    step = np.empty([len(p0),1])

    for i in range(len(p0)):
        samples[:,i],step[i] = np.linspace(p0[i],p1[i],num=nsamples, retstep=True)

        
    return samples,step


def integrate(samplesx, samplesy = None,stepx=.1,stepy=None, order=1):

    if samplesy is None:
        samplesy = samplesx

    if stepy is None:
        stepy=stepx

    out = 0    

    for dim in range(len(stepx)):

        outer_integral = 0

        for samplex in samplesx:

            inner_integral = 0

            for sampley in samplesy:
                inner_integral += func(samplex,sampley)*stepy[dim]

            outer_integral += inner_integral*stepx[dim]

        out += outer_integral

    return out



def  func(x,y):
    if np.linalg.norm(x-y) == 0:
        return 0
    
    return np.log(np.linalg.norm(x-y))


def poly_estimation(x,y) -> np.ndarray:

    xmat = np.empty(shape=[len(x),len(x)])

    if x[-1]-x[0]==0:
        return np.zeros([len(x,1)])
    else:
        for i,xi in enumerate(x):
            for o in range(len(x)):
                xmat[i,len(x)-1-o] = xi**o
        
        b = np.dot(np.linalg.inv(xmat), y)

        return b

def poly_integration(c,x):
    def integ(b,y):

        out = 0

        for i in range(len(b)):

            out += b[i] * y**(len(b)-i) / (len(b)-i)

        return out

    a0 = integ(c,x[0])
    a1 = integ(c,x[-1])

    # fig,a = plt.subplots()

    # a.plot(np.linspace(x[0], x[-1], 50),integ(c,np.linspace(x[0], x[-1], 50)))
    # a.fill_between(np.linspace(x[0], x[-1], 50), quad(c,np.linspace(x[0], x[-1], 50)), alpha=.5)
    # a.plot(x,quad(c,x),'r*')

    # plt.show()

    return a1-a0


def analytical_coincident_line_solution(a,b):

    int_inner = [[[],[]],[[],[]]]
    int_outer = [[],[]]

    for i,x in enumerate([a,b]):
        for j,y in enumerate([a,b]):
            if x != y:
                int_inner[j] = (x-y)**2 * (2*np.log(np.abs(a-b)) - 3) / 4
            else:
                int_inner[j] = 0

        int_outer[i]= int_inner[1] - int_inner[0]
    
    return int_outer[1] - int_outer[0]

def analytical_coincident_point_solutiona(segi=np.array([[0,0,0],[1,0,0]]),segj=np.array([[2,0,0],[1,0,0]])):
    
    seti = {tuple(row) for row in segi}

    for i,el in enumerate(segj):
        if tuple(el) in seti:
            if i != 0:
                segj = np.flipud(segj)
            if tuple(segi[i]) == tuple(segj[i]):
                segi = np.flipud(segi)


    b = np.linalg.norm(segi[-1]-segi[0])
    c = np.inner(segi[-1]-segi[0],segj[-1]-segj[0])/b
    e = np.linalg.norm(np.cross(segi[-1]-segi[0],segj[-1]-segj[0])/b)


    int_inner = [[[],[]],[[],[]]]
    int_outer = [[],[]]

    m = e/c

    if segj[0,0]!=segj[1,0]:

        for i,x in enumerate([0,b]):
            for j,y in enumerate([b,b+c]):

                if x!=y:
                    AA = ( y**2 - y*( y - 2*x ) * np.log( abs( (m**2 + 1 ) *y**2 -2*x*y + x**2 ) ) ) / 4

                    AB = -(3*m**2 +1) * x**2 * np.log( abs( (m**2 +1)*y**2 -2*x*y + x**2 ) ) / ( 4 * ( m**2 + 1 )**2)

                    if m*x!=0:
                        AC = m**2 * x * abs(x) * abs(m) * np.arctan( ( ( m**2 + 1 ) * y - x) / ( abs(x) * abs(m) ) ) / ( ( m**2 + 1 )**2 )
                    else:
                        AC = 0

                    AD= - ( 2 * m**2 + 1 ) * x * y / ( 2 * ( m**2 + 1 )**2 )

                    A = ( AA + AB + AC + AD )

                    BA = x**2*m**2*np.log((m**2 +1)*y**2 -2*x*y + x**2)

                    if m*x!=0:
                        BB = (1 - m) * (m + 1) * x * abs(x) * abs(m) * np.arctan( ((m**2 + 1)*y - x) / (abs(x)*abs(m)) )
                    else:
                        BB = 0
                    if m*y != 0:
                        BC = - m * y**2 * np.arctan( (y - x) / (m*y) ) / 2
                    else:
                        BC = 0

                    BD = x * y * m**2 / ( 2 * (m**2 + 1) )

                    Bk = 1 / ( 2 * ( m**2 + 1 )**2 )

                    B = (BA+BB)*Bk + BC + BD

                    C = -x*y

                    int_inner[i][j] = A+B+C

                else:
                    int_inner[i][j] = 0

            int_outer[i]= int_inner[i][1] - int_inner[i][0]
        
        return int_outer[1] - int_outer[0]
    
    else:
        return 0

def analytical_coincident_point_solutionb(segi=np.array([[0,0,0],[1,0,0]]),segj=np.array([[2,4,0],[1,0,0]])):
    
    seti = {tuple(row) for row in segi}

    for i,el in enumerate(segj):
        if tuple(el) in seti:
            if i != 0:
                segj = np.flipud(segj)
            if tuple(segi[i]) == tuple(segj[i]):
                segi = np.flipud(segi)


    b = np.linalg.norm(segi[-1]-segi[0])
    c = np.inner(segi[-1]-segi[0],segj[-1]-segj[0])/b
    e = np.linalg.norm(np.cross(segi[-1]-segi[0],segj[-1]-segj[0])/b)


    int_inner = [[[],[]],[[],[]]]
    int_outer = [[],[]]

    m = e/c
    l = (m**2+1)

    if segj[0,0]!=segj[1,0]:

        for i,x in enumerate([0,b]):
            for j,y in enumerate([b,b+c]):

                r = np.sqrt(( x - y )**2 + ( m * (y-b))**2 )

                if x!=y:
                    AA =  (x**2 - 2*(l*y - b*m)*x + (3*m**2+1)*y**2 - 4*b*m**2*y + b**2*m**2) * np.log(r)

                    if m*(y-b)!=0:
                        AB = - 4 * m**3 * (y-b)**2 * np.arctan( ( x - y ) / ( m * (y-b) ) )
                    else:
                        AB = 0
                    
                    AC = - x**2 + 2 * ( 2 * m**2 * y + y - 2 * b * m**2 ) * x 

                    Ak = -1/2
                    
                    A = ( AA + AB + AC )*Ak

                    

                    if x-b!=0:
                        BA = (x-b)*abs((x-b)) * np.arctan( ( ( x - l * y + b*m**2 ) / abs( (x-b) * m ) ) )
                    else:
                        BA = 0

                    if r != 0:
                        BB = - (y-b)**2 * abs(m) * np.log(r)
                    else:
                        BB = 0

                    if m*(y-b) != 0:
                        BC = ( m**2 - 1 )*(y-b)*abs(y-b)*np.arctan((x-y)/abs((y-b)*m))
                    else:
                        BC = 0

                    BD = -x*(y-b)*abs(m)

                    Bk = -abs(m)

                    B = (BA + BB + BC + BD)*Bk

                    C = -2*l*x*y

                    K = 1/(2*l)

                    int_inner[i][j] = (A+B+C)*K

                else:
                    int_inner[i][j] = 0

            int_outer[i]= int_inner[i][1] - int_inner[i][0]
        
        return int_outer[1] - int_outer[0]
    
    else:
        return 0
    

def analytical_coincident_point_solution(segi=np.array([[0,0,0],[1,0,0]]),segj=np.array([[2,1,0],[1,0,0]])):
    
    seti = {tuple(row) for row in segi}

    for i,el in enumerate(segj):
        if tuple(el) in seti:
            if i != 0:
                segj = np.flipud(segj)
            if tuple(segi[i]) == tuple(segj[i]):
                segi = np.flipud(segi)


    b = np.linalg.norm(segi[-1]-segi[0])
    c = np.inner(segi[-1]-segi[0],segj[-1]-segj[0])/b
    e = np.linalg.norm(np.cross(segi[-1]-segi[0],segj[-1]-segj[0])/b)


    int_inner = [[[],[]],[[],[]]]
    int_outer = [[],[]]

    m = e/c

    if m==0:
        return 0

    if segj[0,0]!=segj[1,0]:

        for i,x in enumerate([0,b]):
            for j,y in enumerate([0,c]):

                
                
                AD = - ( 2*m**2 + 1 ) * x * y / ( 2*( m**2 + 1 ) )

                
                BD = m**2*x*y / ( 2*m**2 + 1 )

                if m*x!=0:
                    AB = - ( 3*m**2 + 1 ) * x**2 * np.log(abs( (m**2+1)*y**2 - 2*x*y + x**2 )) / ( 4*( m**2 + 1 )**2 )
                    AC = m**2 * x *abs(m*x) * np.arctan(( (m**2+1)*y-x ) / abs(m*x)) / (( m**2 + 1 )**2)

                    BA = m**2*x**2*np.log(abs( (m**2+1)*y**2 - 2*x*y + x**2 )) / ( 2*( m**2 + 1 )**2 )
                    BB = (1-m)*(m+1) * x *abs(m*x) * np.arctan(( (m**2+1)*y-x ) / abs(m*x)) / (2*( m**2 + 1 )**2)

                else:
                    AB = 0
                    AC = 0

                    BA = 0
                    BB = 0

                if m*y!=0:
                    AA = (y**2 - y*( y - 2*x )*np.log(abs( (m**2+1)*y**2 - 2*x*y + x**2 )))/4

                    BC = - m * y**2 * np.arctan( (y-x) / (m*y) ) / 2

                else: 

                    AA = 0

                    BC = 0

                A = AA+AB+AC+AD
                B = BA+BB+BC+BD
                C = -x*y

                int_inner[i][j] = ( A + B + C )


            int_outer[i]= int_inner[i][1] - int_inner[i][0]
        
        return int_outer[1] - int_outer[0]
    
    else:
        return 0

def main():

    a = np.array([0])
    b = np.array([10])

    t0=time.time()
    an = abs(analytical_coincident_line_solution(a[0],b[0]))
    tan = time.time()-t0
    steplist = [3,5,10,20,50,100,200,500]

    outvals = []
    runtimes = []
    gruntimes = []

    for samplenr in steplist:

        print(samplenr)

        t0 = time.time()

        sampling,steps = sample(a,b,nsamples=samplenr)

        t1 = time.time()



        outvals.append(integrate(samplesx=sampling,stepx=steps))

        tf = time.time()

        runtimes.append(tf-t1)
        gruntimes.append(tf-t0)




    fig,ax= plt.subplots(2,1)

    ax[0].plot(steplist,outvals, label="computational solution")
    ax[0].plot(steplist,np.ones_like(steplist)*an, 'k--', label="analytical solution")
    #ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].set_xlabel("# sampling points")
    ax[0].set_title("integral output")
    ax[0].legend()

    ax[1].plot(steplist,runtimes,label="integration")
    ax[1].plot(steplist,gruntimes,label="sampling + integration")
    ax[1].plot(steplist,np.ones_like(steplist)*tan, 'k--', label="analytical solution")
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_xlabel("# sampling points")
    ax[1].set_ylabel("time [s]")
    ax[1].set_title("runtimes")
    ax[1].legend()

    ax[0].grid()
    ax[1].grid()


    plt.show()

    print(an-analytical_coincident_line_solution(b,a))


def main2():

    b = np.array([[1,0,0],[2,4,0]])
    a = np.array([[0,0,0],[1,0,0]])

    t0=time.time()
    an = abs(analytical_coincident_point_solution(a,b))
    tan = time.time()-t0
    steplist = [3,5,10,20,50,100,200]

    outvals = []
    runtimes = []
    gruntimes = []

    for samplenr in steplist:

        print(samplenr)

        t0 = time.time()

        aa,stepsa = sample(a[0],a[1],nsamples=samplenr)
        bb,stepsb = sample(b[0],b[1],nsamples=samplenr)


        t1 = time.time()

        outvals.append(integrate(samplesx=aa,samplesy=bb,stepx=stepsa,stepy=stepsb))

        tf = time.time()

        runtimes.append(tf-t1)
        gruntimes.append(tf-t0)




    fig,ax= plt.subplots(2,1)

    ax[0].plot(steplist,outvals, label="computational solution")
    ax[0].plot(steplist,np.ones_like(steplist)*an, 'k--', label="analytical solution")
    #ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].set_xlabel("# sampling points")
    ax[0].set_title("integral output")
    ax[0].legend()

    ax[1].plot(steplist,runtimes,label="integration")
    ax[1].plot(steplist,gruntimes,label="sampling + integration")
    ax[1].plot(steplist,np.ones_like(steplist)*tan, 'k--', label="analytical solution")
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_xlabel("# sampling points")
    ax[1].set_ylabel("time [s]")
    ax[1].set_title("runtimes")
    ax[1].legend()

    ax[0].grid()
    ax[1].grid()


    plt.show()

    print(an-analytical_coincident_point_solution(b,a))



#main2()