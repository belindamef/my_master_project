import numpy as np
import wesanderson
import string
import palettable
from utilities.config import Paths
from matplotlib import pyplot as plt
import os


class VeryPlotter:

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.rcParams = None

    def get_pyplot_object(self):
        """This function sets some plt defaults and returns blue and red color
        palettes

            Input
                plt     : Matplotlib instance

            Output
                plt     : update Matplotlib instance
                colors  : blue and red color palettes

        """

        # plt default parameters
        self.rcParams = {
            'text.usetex': 'True',
            'axes.spines.top': 'False',
            'axes.spines.right': 'False',
            'yaxis.labellocation': 'bottom'
        }
        plt.rcParams.update(self.rcParams)
        return plt

    def get_exp_group_colors(self):
        viridis_20 = palettable.matplotlib.Viridis_20.colors

        col_exp = [
            [value / 255 for value in list_]
            for list_ in [viridis_20[4], viridis_20[1]]]

        return col_exp


    def get_agent_colors(self, control_color="orange"):

        viridis_20 = palettable.matplotlib.Viridis_20.colors
        col_A = [
            [value / 255 for value in list_]
            for list_ in [viridis_20[3], viridis_20[19], viridis_20[14]]]
        
        if control_color == "organge":
            col_C = [wesanderson.color_palettes['Darjeeling Limited'][1][0],
                    wesanderson.color_palettes['Darjeeling Limited'][1][2],
                    # wesanderson.color_palettes['Hotel Chevalier'][0][3],
                    wesanderson.color_palettes['Isle of Dogs'][1][2]]
        elif control_color == "grey":
            col_C = ['0.35', '0.6', '0.85']
        
        return col_A, col_C

    def define_a3_colors(self):
        color_indices = np.flip(np.round(np.linspace(3, 19, 11)))
        color_indices = np.round(color_indices)
        viridis_20 = palettable.matplotlib.Viridis_20.colors

        a3_viridis_colors = [viridis_20[int(i)] for i in color_indices]
        a3_colors = [
            [value / 255 for value in list_]
            for list_ in a3_viridis_colors]

        return a3_colors


    def config_axes(self, ax, y_label=None, y_lim=None, title=None, x_label=None,
                    x_lim=None, xticks=None, xticklabels=None, yticks=None,
                    ytickslabels=None, title_font=18,
                    title_color=None, ticksize=13,
                    axix_label_size=14):
        """Set basic setting for plot axes"""
        ax.grid(True, axis='y', linewidth=.3, color=[.9, .9, .9])
        if title is not None:
            if title_color is None:
                title_color = "black"
            ax.set_title(title, size=title_font,
                         fontdict={'color': title_color})
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



    def plot_bar(self, ax, x, height, colors, bar_width=0.6, errorbar_size=10,
                yerr=None, labels=None):
        """Plot bars with error bar if given"""
        yerr[np.isnan(yerr)]=0
        return ax.bar(x=x, height=height, yerr=yerr,
                    width=bar_width,
                    color=colors, zorder=0,
                    clip_on=False,
                    error_kw=dict(ecolor='gray', lw=2, capsize=errorbar_size,
                                    capthick=0.9, elinewidth=0.9),
                    label=labels)


    def plot_bar_scatter(self, ax, data, color, bar_width):
        """Plot scatters over bar with half bar_width scatter range"""
        scatter_width = bar_width * (3 / 4)

        # Sort single data points to scatter
        unique, counts = np.unique(data, return_counts=True)
        y_counts_dic = dict(zip(unique, counts))
        max_y_number = max(y_counts_dic.values())
        y_x_pos = []
        y_values = []
        for y_value, y_count in y_counts_dic.items():
            if y_count == 1:
                positions = [0]
            else:
                positions = (np.linspace(0, (y_count
                                            * scatter_width
                                            / max_y_number),
                                        y_count)
                            - y_count * scatter_width / max_y_number / 2)
            y_x_pos.extend(positions)
            y_values.extend(y_count * [y_value])

        ax.scatter(y_x_pos, y_values, alpha=0.4, s=6, color=color, zorder=1,
                clip_on=False)

    def add_letters(self, ax):
        """Add letters to subplots"""
        for key, value in ax.items():
            value.text(-0.05, 1.25, string.ascii_lowercase[key],
                    transform=value.transAxes,
                    size=30, weight='bold')

    def save_figure(self, fig, figure_filename: str):
        fig.tight_layout()
        fn = os.path.join(self.paths.figures, figure_filename)
        fig.savefig(f"{fn}.png", dpi=200, format='png')
        fig.savefig(f"{fn}.pdf", dpi=200, format='pdf')
