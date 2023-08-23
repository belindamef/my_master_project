"""This script plots model fitting performance results: BICs"""
import math
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot
from code.utilities.very_plotter import VeryPlotter
import pandas as pd
from utilities.config import DirectoryManager, DataLoader


def find_central_value(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)

    if n % 2 == 1:
        central_index = n // 2
        central_value = sorted_lst[central_index]
    else:
        middle_right = n // 2
        middle_left = middle_right - 1
        central_value = (sorted_lst[middle_left] + sorted_lst[middle_right]) / 2

    return central_value


def find_value_closest_to_half_max(lst):
    max_val = max(lst)
    half_max = max_val / 2

    closest_val = None
    min_distance = float('inf')

    for val in lst:
        distance = abs(val - half_max)
        if distance < min_distance:
            closest_val = val
            min_distance = distance

    return closest_val


def pick_values(lst):
    if len(lst) < 3:
        raise ValueError("Input list must contain at least 3 values")

    min_val = min(lst)
    max_val = max(lst)

    # Pick one value from the remaining list
    middle = find_value_closest_to_half_max(lst)

    return [min_val, middle, max_val]


def main():
    # Prepare data
    dir_mgr = DirectoryManager()
    dir_mgr.define_val_results_path(dir_label=EXP_LABEL, version=VERSION_NO)
    data_loader = DataLoader(dir_mgr.paths, exp_label=EXP_LABEL)
    all_bics_df = data_loader.load_data_in_one_folder(
        folder_path=dir_mgr.paths.this_model_fit_results_dir
        )

    bic_of_analizing_models = [
        col_name for col_name in all_bics_df.columns if "BIC" in col_name]
    analyzing_models = [
        bic_model[-2:] for bic_model in bic_of_analizing_models
        ]
    bic_group_averages = all_bics_df.groupby(
        ["participant"], dropna=False)[bic_of_analizing_models].agg(
            ["mean", "std"])

    bic_min_for_yaxis = int(math.floor(
        np.nanmin(bic_group_averages.loc[:]/ 100.0)) * 100)

    bic_max_for_yaxis = int(math.ceil(
        np.nanmax(bic_group_averages.loc[:]/ 100.0)) * 100)
    
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
                  "A3": col_agents[2]}
    axes = {}
    fig = plt.figure(figsize=(26, 9), layout="constrained")
    gridspecstrum = gridspec.GridSpec(1, 1)

# ------ Figure A --------------------------------
    i = 1

    axes[i] = plt.subplot(gridspecstrum[0, i])
    x_ticks = range(1, len(analyzing_models)+1)
    x_tick_labels = []
    colors = []

    j = 0
    for analyzing_model in analyzing_models:

        x_tick_labels.append(analyzing_model)

        colors.append(color_dict[analyzing_model])
        axes[i].errorbar(j+1,
                         np.mean(bic_group_averages[
                             f"BIC_{analyzing_model}", "mean"]),
                         alpha=0.7, markersize=7,
                         color=color_dict[analyzing_model],
                         fmt='o', linestyle=None, clip_on=False,
                         label=analyzing_model,
                         yerr=np.mean(bic_group_averages[
                             f"BIC_{analyzing_model}", "std"])
                        )
        j += 1

    plotter.config_axes(ax=axes[i],
                        title="participant",
                        #title_color=color_dict[f"{gen_agent}"],
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
                        title_font=20,
                        axix_label_size=22,
                        ticksize=18,
                        x_label="Analyzing agent"
                        )

    plotter.config_axes(ax=axes[0], y_label="BIC", axix_label_size=22)
    fig.align_ylabels(axs=list(axes.values()))

    plotter.save_figure(fig=fig, figure_filename=FIGURE_FILENAME)

if __name__ == "__main__":

    EXP_LABEL = "exp_msc"
    VERSION_NO = "test_1"
    FIGURE_FILENAME = f"figure_model_fit_{VERSION_NO}"
    N_BLOCKS = 3

    main()
