import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_colormap(data):
    """
    Helper function to plot data with associated colormap.
    """
    np.random.seed(19680801)
    viridis = cm.get_cmap('viridis', 256)
    n = len(data.items())

    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    for [ax, data] in zip(axs.flat, data.values()):
        psm = ax.pcolormesh(data, cmap=viridis,
                            rasterized=True,
                            vmin=0, vmax=0.1)
    fig.colorbar(psm, ax=ax)

    return fig
