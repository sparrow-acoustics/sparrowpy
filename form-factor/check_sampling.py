import sparapy.ff_helpers.sampling as sampling
import matplotlib.pyplot as plt
from elmt import elmt


def plot_elmt(el,ax=plt.gca,style='b-'):
    px=[]
    py=[]
    pz=[]
    for j in range(len(el.pt)+1):
        px.append(el.pt[j%len(el.pt),0])
        py.append(el.pt[j%len(el.pt),1])
        pz.append(el.pt[j%len(el.pt),2])

    ax.plot(px,py,pz,style)
    return ax

def plot_points(ptlist, el):
        
    fff=plt.figure()
    aaa=fff.add_subplot(111,projection='3d')
    aaa.axis('equal')
    px=[]
    py=[]
    pz=[]
    aaa = plot_elmt(el=el,ax=aaa)

    for pt in ptlist:
        aaa.plot(pt[0],pt[1],pt[2],'ro')

    plt.show()


patch = elmt([[1,0,0],[0,1,0],[0,0,1]])

plot_points(sampling.sample_random(el=patch), el=patch)

plot_points(sampling.sample_regular(el=patch), el=patch)
print("hee hee")