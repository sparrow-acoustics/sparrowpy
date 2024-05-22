import numpy as np

class elmt:

    def normal(self):
        self.n = np.cross(self.pt[1]-self.pt[0], self.pt[2]-self.pt[0])/np.linalg.norm(np.cross(self.pt[1]-self.pt[0], self.pt[2]-self.pt[0]))

    def area(self):
        if len(self.pt) == 3:
            self.A  = .5*np.linalg.norm(np.cross(self.pt[1]-self.pt[0], self.pt[2]-self.pt[0]))

        if len(self.pt) == 4:
            self.A  = .5*np.linalg.norm(np.cross(self.pt[3]-self.pt[2], self.pt[0]-self.pt[2])) + .5*np.linalg.norm(np.cross(self.pt[1]-self.pt[0], self.pt[2]-self.pt[0]))
    
    def centroid(self):
        self.o = np.sum(self.pt,axis=0)/len(self.pt)

    def __init__(self, pts) -> None:

        self.pt = np.array(pts, dtype=float)

        self.normal()
        self.area()
        self.centroid()

        pass