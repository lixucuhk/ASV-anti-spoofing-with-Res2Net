"""Script for data visualization.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def plot_data(data, path, ylabel="y"):
    """Plot data.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(
        data.transpose((-1, 0)),
        aspect='auto',
        cmap='hot',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'frame'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close("all")
