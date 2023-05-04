"""This script plots figure 2"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import palettable

def get_fig_template(plt):
    """This function sets some plt defaults and returns blue and red color
    palettes

        Input
            plt     : Matplotlib instance

        Output
            plt     : update Matplotlib instance
            colors  : blue and red color palettes

    """

    # plt default parameters
    rcParams = {
        'text.usetex': 'True',
        'axes.spines.top': 'False',
        'axes.spines.right': 'False',
        'yaxis.labellocation': 'bottom'
    }
    plt.rcParams.update(rcParams)

    # tab20 = plt.get_cmap('tab20')
    # tab20b = plt.get_cmap('tab20b')
    # tab20c = plt.get_cmap('tab20c')
    # col_exp = [tab20c(11), tab20c(8)]
    # col_A = [tab20c(0), tab20b(18), tab20(18)]
    col_C = ['#ECCBAE',
             '#D69C4E',
             # wesanderson.color_palettes['Hotel Chevalier'][0][3],
             '#B6854D']

    viridis_20 = palettable.matplotlib.Viridis_20.colors
    # magma_20 = palettable.matplotlib.Magma_20.colors
    col_exp = [
        [value / 255 for value in list_]
        for list_ in [viridis_20[4], viridis_20[1]]]
    col_A = [
        [value / 255 for value in list_]
        for list_ in [viridis_20[18], viridis_20[14], viridis_20[9]]]
    return plt, col_exp, col_A, col_C  # , col_A, reds


def config_axes(ax, y_label=None, y_lim=None, title=None, x_label=None,
                x_lim=None, xticks=None, xticklabels=None, yticks=None,
                ytickslabels=None, title_font=18, ticksize=13,
                axix_label_size=14):
    """Set basic setting for plot axes"""
    ax.grid(True, axis='y', linewidth=.3, color=[.9, .9, .9])
    if title is not None:
        ax.set_title(title, size=title_font)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=axix_label_size, loc='center')
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=axix_label_size)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, fontsize=ticksize)
    if yticks is not None:
        ax.set_yticks(yticks)
    if ytickslabels is not None:
        ax.set_yticklabels(ytickslabels, fontsize=ticksize)

# Specify directories and filenames
fig_fn = Path('figure_2_test_LaTeX_manu.png')

# Initialize figure
plt, col_exp, col_A, col_C = get_fig_template(plt)
plt.rcParams.update({'text.usetex': 'True'})  # for testing LaTeX
fig = plt.figure(figsize=(20, 6))
gs = gridspec.GridSpec(1, 5)
ax = {}
size_shape = 14
agent_colors = col_A + col_C  # Specify agent colors

# ------Trial-by-trial/round-wise average choice rates------------------

x = np.array([1, 2, 3])
x.sort()
for i, gen_model in enumerate(["C1", "C2", "C3"]):
    ax[i] = plt.subplot(gs[0, i])
    y = np.array([1, 2, 3])
    e = np.array([0.1, 0.2, 0.33])
    ax[i].errorbar(x, y, alpha=0.7, markersize=4,
                   fmt='o', linestyle=None, clip_on=False,
                   label=f"{gen_model}", yerr=e)
    ax[i].legend(loc='upper right')
    config_axes(ax[i], y_label="tau_est", x_label="tau_gen",
                xticks=np.linspace(0.25, 3, 10),
                yticks=np.round(np.linspace(0.25, 3, 10), 1))
    ax[i].set_xticklabels(np.round(np.linspace(0.25, 3, 10), 1), fontsize=10)
    ax[i].set_xticklabels(np.round(np.linspace(0.25, 3, 10), 1), fontsize=10)

# Print subject level descriptive figure
fig.tight_layout()
fig.savefig(fig_fn, dpi=200, format='png')
