from email.base64mime import header_length
from turtle import width
import numpy as np
import pandas as pd
import wesanderson
import string
import palettable
from utilities.config import Paths
from utilities.task import TaskNGridParameters
from dataclasses import dataclass
import os
from matplotlib import pyplot, colors, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

@dataclass
class PlotCustomParams:
    # fontsizes
    standard_fs = 17
    legend_fs = standard_fs
    axis_label_fs = standard_fs
    axis_tick_fs = standard_fs
    axis_title_fs = standard_fs

    # marker
    marker_shape = 'o'
    marker_sz = 5
    transp_lvl = 0.7

    # errorbar_lines
    err_bar_linestyle = '-'
    err_bar_linewidth = 1

    # control agent lines
    c_agent_linestyle = '-'
    c_agent_linewidth = 0.8
    c_agent_std_transp_lvl = 0.2

    # ticks
    tau_ticks = np.round(np.linspace(0, 0.5, 3), 2)
    lambda_ticks = np.round(np.linspace(0.1, 1, 10), 1)
    n_tr_ticks = np.linspace(0, 10, 5)

    def define_tau_tick_labels(self, max_tau_value: float,
                               n_values: int = 3):
        self.tau_ticks = np.round(np.linspace(0, max_tau_value, n_values), 2)

    def define_lambda_tick_labels(self, max_lambda_value: float,
                                  n_values: int = 3):
        self.lambda_ticks = np.round(
            np.linspace(0, max_lambda_value, n_values), 2)


class VeryPlotter:

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.rcParams = None
        self.color_dict = {}
        self.plt = pyplot

    def define_run_commands(self) -> dict:
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
        # plt.rcParams.update(self.rcParams)
        return self.rcParams

    def get_exp_group_colors(self):
        viridis_20 = palettable.matplotlib.Viridis_20.colors

        col_exp = [
            [value / 255 for value in list_]
            for list_ in [viridis_20[4], viridis_20[1]]]

        return col_exp


    def get_agent_colors(self, control_color="orange") -> dict:

        viridis_20 = palettable.matplotlib.Viridis_20.colors
        col_A = [
            [value / 255 for value in list_]
            for list_ in [viridis_20[3], viridis_20[19], viridis_20[14]]]
        
        if control_color == "orange":
            col_C = [wesanderson.color_palettes['Darjeeling Limited'][1][0],
                    wesanderson.color_palettes['Darjeeling Limited'][1][2],
                    # wesanderson.color_palettes['Hotel Chevalier'][0][3],
                    wesanderson.color_palettes['Isle of Dogs'][1][2]]
        elif control_color == "grey":
            col_C = ['0.35', '0.6', '0.85']
        
        color_dict = {"C1": col_C[0],
                      "C2": col_C[1],
                      "C3": col_C[2],
                      "A1": col_A[0],
                      "A2": col_A[1],
                      "A3": col_A[2]}
        return color_dict

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

    def save_figure(self, fig, figure_filename: str,
                    pdf: bool = True, png: bool = False):
        fig.tight_layout()
        fn = os.path.join(self.paths.figures, figure_filename)

        if pdf:
            fig.savefig(f"{fn}.pdf", dpi=200, format='pdf')

        if png:
            fig.savefig(f"{fn}.png", dpi=200, format='png')

    def plot_heat_maps_of_belief_states(self, task_params: TaskNGridParameters,
                                        beh_data: pd.DataFrame):
        """Function to plot model variables over trials

        Args:
            task_params (TaskNGridParameters): _description_
            beh_data (pd.DataFrame): _description_
        """

        # ---- Prepare data ----------------------------------------
        dim = task_params.dim
        n_nodes = task_params.n_nodes
        n_plotable_trials = beh_data["s1_t"].count()
        model_components = ["s1_t", "s2_t", "o_t",
                            "marg_s1_b_t", "marg_s2_b_t",
                            "a_t"]
        n_rows = len(model_components)

        data = {}
        for component in model_components:
            data[component] = np.full((n_nodes,
                                       n_plotable_trials + 1), np.nan)

            for trial_col in range(n_plotable_trials):
                if component in ["s1_t", "s2_t"]:
                    data[component][:, trial_col] = np.array(
                        [1 if node == beh_data[component][trial_col] - 1 else 0
                            for node in range(n_nodes)]
                        )

                elif component == "o_t":
                    data[component][:, trial_col] = beh_data["o_t"][trial_col][1:5]
                    if beh_data["o_t"][trial_col][0] == 1:
                        data[component][:, trial_col][
                            (beh_data["s2_t"][trial_col] - 1)] = 3

                elif component in ["marg_s1_b_t", "marg_s2_b_t"]:
                    data[component][:, trial_col] = beh_data[component][trial_col]

                elif component == "a_t":

                    # plot curren position along in a grid
                    data[component][:, trial_col] = np.array(
                        [1 if node == beh_data["s1_t"][trial_col] - 1 else 0
                            for node in range(n_nodes)]
                        )


        # ------Preapre figure-----------------------------------------------
        # Get viridis colormap
        viridis_cmap = cm.get_cmap("viridis")
        rc_params = self.define_run_commands()
        self.plt.rcParams.update(rc_params)

        fig, axs = self.plt.subplots(
            n_rows,
            n_plotable_trials,
            sharex=True, sharey=True,
            layout="constrained"
            #figsize=(16, 9)
            )

        fig.suptitle('Belief State Update')

        y_labels = {
            "s1_t": r"$s^1_t$",
            "s2_t": r"$s^2_t$",
            "o_t": r"$o_t$",
            "marg_s1_b_t": r"$p(s^1_t\vert o_t)$",
            "marg_s2_b_t": r"$p(s^2_t\vert o_t)$",
            "a_t": r"$a_t$"
            }

        cmaps = {
            "s1_t": colors.ListedColormap(["black", "grey"]),
            "s2_t": colors.ListedColormap(["black", "green"]),
            "o_t": colors.ListedColormap(["black", "grey", "lightblue", "green"]),
            "marg_s1_b_t": colors.ListedColormap([viridis_cmap(0),
                                                  viridis_cmap(256)]),
            "marg_s2_b_t": "viridis",
            "a_t": colors.ListedColormap(["black", "grey"])
            }

        v_range = {
            "s1_t": [0, 1],
            "s2_t": [0, 1],
            "o_t": [0, 3],
            "marg_s1_b_t": [0, 1],
            "marg_s2_b_t": [0, 1],
            "a_t": [0, 1]
        }

        # Set up the colorbar for "o_t"
        cmap_ticks = {
            "s1_t": np.linspace(0, 1, 2),
            "s2_t": np.linspace(0, 1, 2),
            "o_t": np.linspace(0, 3, 4),  # np.linspace(0, 3, 4),
            "marg_s1_b_t": np.linspace(0, 1, 2),
            "marg_s2_b_t": np.linspace(0, 1, 2),
            "a_t": np.linspace(0, 1, 2)
        }

        # ------ Create images -----------------------------
        images = []

        for row, component in enumerate(model_components):
            for trial_col in range(n_plotable_trials):

                images.append(
                    axs[row, trial_col].imshow(
                        data[component][:, trial_col].reshape(dim, dim),
                        cmap=cmaps[component],
                        vmin=v_range[component][0],
                        vmax=v_range[component][1]
                        )
                    )

                axs[row, trial_col].set_yticklabels([])  # Remove y-axis ticks
                axs[row, trial_col].set_xticklabels([])  # Remove y-axis ticks

                axs[row, trial_col].set_ylabel(y_labels[component],
                                               loc="center",
                                               rotation="horizontal",
                                               labelpad=20)
                axs[row, trial_col].label_outer()
                axs[row, trial_col].set_xticks(np.arange(-0.5, dim, 1), minor=True)
                axs[row, trial_col].set_yticks(np.arange(-0.5, dim, 1), minor=True)
                axs[row, trial_col].set_xticks([], major=True)
                axs[row, trial_col].set_yticks([], major=True)
                axs[row, trial_col].grid(
                    which="minor",
                    color='grey',
                    linestyle='-',
                    linewidth=0.1)

                if component == "a_t":
                    a_t = beh_data["a_t"][trial_col]

                    if not np.isnan(a_t):

                        s1_t_field = np.where(data["s1_t"][:, trial_col].reshape(dim, dim) == 1)
                        s1_t_x = int(s1_t_field[1])  # column --> x axis
                        s1_t_y = int(s1_t_field[0])  # row --> y axis

                        # Get the extent (x0, x1, y0, y1) in data coordinates
                        extent = images[row].get_extent()  # returns the image extent as tuple (left, right, bottom, top).


                        # TODO: translate to other dimensions

                        # Extract x and y coordinates
                        x_coords = np.linspace(extent[0] + 1,  # left
                                               extent[1],  # right
                                               dim)
                        y_coords = np.linspace(extent[3] + 1,  # top
                                               extent[2],  # bottom,
                                               dim)

                        # define arrow direction
                        s1_new = beh_data["s1_t"][trial_col] + a_t
                        s1_new_one_hot = np.array(
                            [1 if node == s1_new - 1 else 0
                                for node in range(n_nodes)]
                            )
                        
                        s1_new_field = np.where(s1_new_one_hot.reshape(dim, dim) == 1)
                        s1_new_x = int(s1_new_field[1])  # column --> x axis
                        s1_new_y = int(s1_new_field[0])  # row --> y axis

                        # calculate arrow "direction" as difference between s_{t + 1} and s_t
                        dx = (x_coords[s1_new_x] - 0.5) - (x_coords[s1_t_x] - 0.5)
                        dy = (y_coords[s1_new_y] - 0.5) - (y_coords[s1_t_y] - 0.5)

                        # Draw arrow
                        axs[row, trial_col].arrow(
                            x_coords[s1_t_x] - 0.5,
                            y_coords[s1_t_y] - 0.5,
                            dx,
                            dy,
                            color="lightgreen",
                            width=0.002,
                            length_includes_head=True,
                            head_width=0.3,
                            head_length=0.25)


        # ------ Add colorbar -------------------

        for row, model_components in enumerate(model_components):
            divider = make_axes_locatable(axs[row, -1])
            cax = divider.append_axes("right", size="13%", pad=0.3)

            ticks = cmap_ticks[model_components]

            fig.colorbar(
                images[(row + 1) * n_plotable_trials - 1],
                cax,
                # location='bottom',
                orientation='vertical',
                ticks=ticks
                # norm=norm
                )

            # if model_components == "o_t":
            # #     norm = colors.BoundaryNorm(np.arange(len(ticks) + 1) - 0.5, cmaps["o_t"].N)
            #     colorbar.set_ticks(np.array([0.5, 1.5, 2.5, 3.5]))
            #     colorbar.set_ticklabels(ticks)
            # # else:
            # #     norm = None

        # Adjust layout for colorbar
        #fig.subplots_adjust(right=0.85)  # Increase the right margin as needed

        fig_fn = f"belief_update_{task_params.n_nodes}-nodes_{task_params.n_hides}-hides"
        self.save_figure(fig=fig, figure_filename=fig_fn)
