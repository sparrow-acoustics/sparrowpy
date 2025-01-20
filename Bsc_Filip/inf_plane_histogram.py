"""functions to generate energy histogram of "infinite" plane scenario."""
from sparrowpy.testing.stub_utils import get_histogram
import matplotlib.pyplot as plt
import numpy as np

def plot_nice_example():
    src=np.array([1.,1.,1.])
    rec=np.array([-1.,-1.,1.])
    sr=500
    
    histogram = get_histogram(
        source_pos=src,
        receiver_pos=rec,
        h2ps_ratio=.5,sampling_rate=sr)
    
    plt.figure()
    plt.plot(np.arange(histogram.shape[1])/sr,histogram[0], "*")
    plt.grid()
    plt.title("infinite plane histogram")
    plt.xlabel("time [s]")
    plt.ylabel("energy coefficients")
    plt.savefig("Bsc_Filip/test_inf_plane.png")
    
# routine if file is run as standalone program
if __name__=="__main__":
    
    plot_nice_example()
    