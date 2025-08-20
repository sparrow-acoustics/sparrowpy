"""Basic formatting functions for plot generation."""
import matplotlib.pyplot as plt

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
