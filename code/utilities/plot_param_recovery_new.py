#!/usr/bin/env python3
"""This script plots figure 2"""
import os
import glob
import re
import argparse
import numpy as np
from matplotlib import pyplot
from very_plotter_new import VeryPlotter
from config import DirectoryManager, DataHandler


def extract_number(f):
    s = re.findall("\d+$", f)
    return (int(s[0]) if s else -1, f)


def find_most_recent_data_dir(val_results_paths: str) -> str:
    all_dirs = glob.glob(os.path.join(val_results_paths, "*"))
    most_recent_data = max(all_dirs, key=extract_number)
    return most_recent_data


def main(exp_label: str, vers: str, save_file: bool = True):

    # Get and Prepare data
    dir_mgr = DirectoryManager()
    dir_mgr.define_model_recov_results_path(exp_label=exp_label,
                                            version=vers)
    data_loader = DataHandler(dir_mgr.paths, exp_label=exp_label)
    all_val_results = data_loader.load_data_in_one_folder(
        folder_path=dir_mgr.paths.this_model_recov_results_dir
        )

    agent_gen_models = all_val_results.agent.unique().tolist()
    agent_gen_models.sort()
    bayesian_gen_agents = [agent for agent in agent_gen_models if "A" in agent]

    mle_grp_avrgs = all_val_results.groupby(
        ["agent", "tau_gen"])[["tau_mle", "lambda_mle"]].agg(
            ["mean", "std"])

    # Select group averages for different lambda_gen_values<<
    mle_grp_avrgs_diff_lambdas = all_val_results.groupby(
        ["agent", "tau_gen", "lambda_gen"])[["tau_mle", "lambda_mle"]].agg(
            ["mean", "std"])

    # Prepare figure
    plotter = VeryPlotter(paths=dir_mgr.paths)
    col_agents, col_controls = plotter.get_agent_colors()
    color_dict = {"C1": col_controls[0],
                  "C2": col_controls[1],
                  "C3": col_controls[2],
                  "A1": col_agents[0],
                  "A2": col_agents[1],
                  "A3": col_agents[2]}

    rc_params = plotter.define_run_commands()
    plt = pyplot
    plt.rcParams.update(rc_params)
    fig, axs = plt.subplots(nrows=2, ncols=3,
                            figsize=(15, 10))

    # fontsizes
    standard_fs = 14
    legend_fs = standard_fs
    axis_label_fs = standard_fs
    axis_tick_fs = 12

    # marker
    marker_shape = 'o'
    marker_sz = 6
    transp_lvl = 0.7

    # other parameters
    tau_ticks = np.round(np.linspace(0, 0.5, 3), 2)
    lambda_ticks = np.round(np.linspace(0.1, 1, 10), 1)

    # ------First row------------------
    row = 0
    for column, gen_model in enumerate(bayesian_gen_agents):
        this_ax = axs[row, column]

        x_values = mle_grp_avrgs.loc[gen_model].index.unique(
            level="tau_gen").values
        y_values = mle_grp_avrgs.loc[gen_model]["tau_mle"]["mean"].values
        stds = mle_grp_avrgs.loc[gen_model]["tau_mle"]["std"].values

        this_ax.errorbar(
            x_values, y_values, yerr=stds,
            fmt=marker_shape, color=color_dict[gen_model],
            alpha=transp_lvl, markersize=marker_sz,
            label=f"{gen_model}")

        this_ax.legend(loc='upper left', fontsize=legend_fs)

        plotter.config_axes(
            this_ax,
            y_label=r"$\hat{\tau}$",
            x_label=r"$\tau$",
            xticks=tau_ticks,
            xticklabels=tau_ticks,
            yticks=tau_ticks,
            ytickslabels=tau_ticks,
            ticksize=axis_tick_fs,
            axix_label_size=axis_label_fs
            )

    # ------Second row------------------
    row = 1
    if "A3" in bayesian_gen_agents:
        gen_model = "A3"
        tau_values = mle_grp_avrgs.loc["A3"].index.unique(
            level="tau_gen").values
        if len(tau_values) > 3:
            number_of_elements_wanted = 3
        else:
            number_of_elements_wanted = len(tau_values)
        indices_tau_selection = np.round(np.linspace(
            0, len(tau_values) - 1, number_of_elements_wanted)
            ).astype(int)
        taus_for_plt = tau_values[indices_tau_selection]

        for column, tau_i in enumerate(taus_for_plt):
            this_ax = axs[row, column]
            x_values = mle_grp_avrgs_diff_lambdas.index.get_level_values(
                "lambda_gen").tolist()
            y_values = mle_grp_avrgs_diff_lambdas.loc[
                "A3", tau_i]["lambda_mle"]["mean"].values
            stds = mle_grp_avrgs_diff_lambdas.loc[
                "A3", tau_i]["lambda_mle"]["std"].values

            this_ax.errorbar(
                x_values, y_values, yerr=stds,
                fmt=marker_shape, color=color_dict[gen_model],
                alpha=transp_lvl, markersize=marker_sz,
                label=f"{gen_model}, " + r"$\tau = $" + f"{tau_i}")

            this_ax.legend(loc='upper left', fontsize=legend_fs)

            plotter.config_axes(
                this_ax,
                y_label=r"$\hat{\lambda}$",
                x_label=r"$\lambda$",
                xticks=lambda_ticks,
                xticklabels=lambda_ticks,
                yticks=lambda_ticks,
                ytickslabels=lambda_ticks,
                ticksize=axis_tick_fs,
                axix_label_size=axis_label_fs)

    if save_file:
        plotter.save_figure(fig=fig, figure_filename=FIGURE_FILENAME)


if __name__ == "__main__":

    EXP_LABEL = "exp_msc"
    VERSION_NO = "test_hr_1_test_hr_1"
    FIGURE_FILENAME = f"figure_param_recov_{VERSION_NO}"
    N_BLOCKS = 3

    parser = argparse.ArgumentParser(description='Plotting')
    parser.add_argument('--dont_save_file', action="store_false",
                        default=True)
    parser.add_argument('--show_plot', action="store_true",default=False)
    args = parser.parse_args()

    save_file = args.dont_save_file
    show_plot = args.show_plot

    main(save_file=save_file, exp_label=EXP_LABEL, vers=VERSION_NO)
