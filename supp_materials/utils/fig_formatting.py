"""Basic formatting functions for plot generation."""
import matplotlib.pyplot as plt
import os

OUT_DIR=os.path.join(os.getcwd(),"figures")

def create_fig(figtype="stubby"):
    """Generate a figure for plotting."""
    font={
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": "Helvetica",
        "font.size": 10,
    }

    match figtype:
        case "stubby":
            size=(3,2)

        case "big":
            size=(5,3)

    plt.rcParams.update(font)

    figure,ax = plt.subplots(figsize=size)
    plt.grid()
    return figure, ax

def export_fig(fig, filename, out_dir=OUT_DIR, fformat=".png"):
    """Export a nice looking figure."""
    fig.savefig(os.path.join(out_dir,filename+fformat), bbox_inches='tight')
