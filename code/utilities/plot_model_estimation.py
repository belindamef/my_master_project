#!/usr/bin/env python3
"""This script plots model fitting for experimental data
performance results: BICs"""
import math
import numpy as np
from matplotlib import pyplot
from utilities.config import DirectoryManager, DataHandler
from utilities.very_plotter_new import VeryPlotter, PlotCustomParams


def plot_model_fitting_perf(
        exp_label: str, vers: str, save_file: bool = True,
        plt_params: PlotCustomParams = PlotCustomParams()):

    # Get and prepare data
    dir_mgr = DirectoryManager()
    dir_mgr.define_model_est_results_path(exp_label=exp_label,
                                          version=vers)
    data_loader = DataHandler(dir_mgr.paths, exp_label=EXP_LABEL)
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
        np.nanmin(bic_group_average)/100)*100)
    bic_max_for_yaxis = int(math.ceil(
        np.nanmax(bic_group_average/100))*100)
    bic_y_tick = np.round(np.linspace(bic_min_for_yaxis, bic_max_for_yaxis, 5),
                          decimals=2)

    # Preapre figure
    plotter = VeryPlotter(paths=dir_mgr.paths)
    plt = pyplot
    color_dict = plotter.get_agent_colors(control_color="grey")

    rc_params = plotter.define_run_commands()
    plt = pyplot
    plt.rcParams.update(rc_params)
    fig, ax = plt.subplots(figsize=(16, 9),
                           layout="constrained")

# ------ Figure A --------------------------------

    x_ticks = range(1, len(analyzing_models)+1)
    x_tick_labels = []
    colors = []

    j = 0
    for analyzing_model in analyzing_models:

        x_tick_labels.append(analyzing_model)
        colors.append(color_dict[analyzing_model])

        ax.errorbar(
            x=j+1,
            y=bic_group_average.loc[
                "mean"][f"BIC_{analyzing_model}"],
            yerr=bic_group_average.loc[
                "std"][f"BIC_{analyzing_model}"],
            fmt=plt_params.marker_shape,
            alpha=plt_params.transp_lvl, markersize=plt_params.marker_sz,
            color=color_dict[analyzing_model],
            label=analyzing_model
                    )

        # Add value as text to the bar plot
        value = bic_group_average.loc[
                         'mean'][f'BIC_{analyzing_model}']
        ax.text(j+1.07, value + 2, s=f"{round(value, ndigits=2)}",
                fontsize=plt_params.standard_fs)
        j += 1

    plotter.config_axes(
        ax=ax,
        x_label="Analyzing agent",
        y_label="BIC",
        axix_label_size=plt_params.axis_label_fs,
        xticks=x_ticks,
        xticklabels=x_tick_labels,
        yticks=bic_y_tick,
        ytickslabels=bic_y_tick,
        ticksize=plt_params.axis_tick_fs)

    plotter.save_figure(fig=fig, figure_filename=FIGURE_FILENAME)


if __name__ == "__main__":

    EXP_LABEL = "exp_msc"
    VERSION = "test_1"
    FIGURE_FILENAME = f"figure_model_est_{VERSION}"
    N_BLOCKS = 3

    plot_model_fitting_perf(exp_label=EXP_LABEL, vers=VERSION, save_file=True)
