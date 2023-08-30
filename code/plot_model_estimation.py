#!/usr/bin/env python3
"""This script plots model fitting performance results: BICs"""
import math
import numpy as np
from matplotlib import gridspec
from utilities.config import DirectoryManager, DataHandler
from utilities.very_plotter_new import VeryPlotter, PlotCustomParams


def plot_model_estimation_results(
        exp_label: str, vers: str, save_file: bool = True,
        plt_params: PlotCustomParams = PlotCustomParams()):

    # Prepare data
    dir_mgr = DirectoryManager()
    dir_mgr.define_model_est_results_path(exp_label=exp_label,
                                          version=VERSION_NO)
    data_loader = DataHandler(dir_mgr.paths, exp_label=exp_label)
    all_bics_df = data_loader.load_data_in_one_folder(
        folder_path=dir_mgr.paths.this_model_est_results_dir
        )

    bic_of_analizing_models = [
        col_name for col_name in all_bics_df.columns if "BIC" in col_name]
    analyzing_models = [
        bic_model[-2:] for bic_model in bic_of_analizing_models
        ]
    bic_group_average = all_bics_df.agg(
            ["mean", "std"])

    bic_min_for_yaxis = int(math.floor(
        np.nanmin(bic_group_average.loc[:]/ 100.0)) * 100)

    bic_max_for_yaxis = int(math.ceil(
        np.nanmax(bic_group_average.loc[:]/ 100.0)) * 100)
    
    # Preapre figure
    plotter = VeryPlotter(paths=dir_mgr.paths)
    plt = plotter.get_pyplot_object()
    col_agents, col_controls = plotter.get_agent_colors(
        control_color="grey")
    color_dict = {"C1": col_controls[0],
                  "C2": col_controls[1],
                  "C3": col_controls[2],
                  "A1": col_agents[0],
                  "A2": col_agents[1],
                  "A3": plot_model_recov_resultscol_agents[2]}
    axes = {}
    fig = plt.figure(figsize=(16, 9), layout="constrained")
    gridspecstrum = gridspec.GridSpec(1, 1)

# ------ Figure A --------------------------------
    i = 1

    axes[i] = plt.subplot(gridspecstrum[0, 0])
    x_ticks = range(1, len(analyzing_models)+1)
    x_tick_labels = []
    colors = []

    j = 0
    for analyzing_model in analyzing_models:

        x_tick_labels.append(analyzing_model)

        colors.append(color_dict[analyzing_model])
        axes[i].errorbar(j+1,
                         bic_group_average.loc[
                             "mean"][f"BIC_{analyzing_model}"],
                         alpha=0.7, markersize=9,
                         color=color_dict[analyzing_model],
                         fmt='o', linestyle=None, clip_on=False,
                         label=analyzing_model,
                         yerr=bic_group_average.loc[
                             "std"][f"BIC_{analyzing_model}"]
                            )
        
        value = bic_group_average.loc[
                         'mean'][f'BIC_{analyzing_model}']
        axes[i].text(j+1.07, value + 2, s=f"{round(value, ndigits=2)}",
                     fontsize=20)
        j += 1

    plotter.config_axes(ax=axes[i],
                        title="Model recovery with experimental data",
                        # title_color=color_dict[f"{gen_agent}"],
                        xticks=x_ticks,
                        xticklabels=x_tick_labels,
                        yticks=np.round(
                            np.linspace(
                                bic_min_for_yaxis,
                                bic_max_for_yaxis,
                                5),
                            decimals=2),
                        ytickslabels=np.round(
                            np.linspace(
                                bic_min_for_yaxis,
                                bic_max_for_yaxis,
                                5),
                            decimals=2),
                        title_font=26,
                        axix_label_size=24,
                        ticksize=18,
                        x_label="Analyzing agent"
                            )

    plotter.config_axes(ax=axes[1], y_label="BIC", axix_label_size=22)
    fig.align_ylabels(axs=list(axes.values()))

    plotter.save_figure(fig=fig, figure_filename=FIGURE_FILENAME)

if __name__ == "__main__":

    EXP_LABEL = "exp_msc"
    VERSION_NO = "test_1"
    FIGURE_FILENAME = f"figure_model_est_{VERSION_NO}"
    N_BLOCKS = 3

    plot_model_estimation_results()
