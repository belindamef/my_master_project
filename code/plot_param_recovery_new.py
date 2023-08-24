"""This script plots figure 2"""
import os
import glob
import numpy as np
import re
from matplotlib import gridspec
from utilities.very_plotter import VeryPlotter
from utilities.config import DirectoryManager, DataHandler
import pandas as pd
import utilities.config as config


def extract_number(f):
    s = re.findall("\d+$", f)
    return (int(s[0]) if s else -1, f)


def find_most_recent_data_dir(val_results_paths: str) -> str:
    all_dirs = glob.glob(os.path.join(val_results_paths, "*"))
    most_recent_data = max(all_dirs, key=extract_number)
    return most_recent_data


def main():

# Specify directories and filenames
    # Prepare data
    dir_mgr = DirectoryManager()
    dir_mgr.define_model_recov_results_path(dir_label=EXP_LABEL,
                                            version=VERSION_NO)
    data_loader = DataHandler(dir_mgr.paths, exp_label=EXP_LABEL)
    all_val_results = data_loader.load_data_in_one_folder(
        folder_path=dir_mgr.paths.this_model_recov_results_dir
        )

    agent_gen_models = all_val_results.agent.unique().tolist()
    agent_gen_models.sort()

    control_gen_agents = [agent for agent in agent_gen_models if "C" in agent]
    Bayesian_gen_agents = [agent for agent in agent_gen_models if "A" in agent]
    n_agents = len(agent_gen_models)

    mle_group_averages = all_val_results.groupby(
        ["agent", "tau_gen"])[["tau_mle", "lambda_mle"]].agg(
            ["mean", "std"])

    # Prepare figure
    plotter = VeryPlotter(paths=dir_mgr.paths)
    plt = plotter.get_pyplot_object()
    col_agents, col_controls = plotter.get_agent_colors()
    fig = plt.figure(figsize=(15, 10))
    gridspecstrum = gridspec.GridSpec(2, 3)
    axes = {}
    agent_colors = col_agents + col_controls  # Specify agent colors
    color_dict = {"C1": col_controls[0],
                  "C2": col_controls[1],
                  "C3": col_controls[2],
                  "A1": col_agents[0],
                  "A2": col_agents[1],
                  "A3": col_agents[2]}
    # ------Trial-by-trial/round-wise average choice rates------------------

    # Select group averages for fixed lambda to 0.5
    # TODO: Choose to either select it here or during data generation
    # mle_group_averages_fixed_lambda = mle_group_averages.loc[:, :, 0.5]
    mle_group_averages_fixed_lambda = mle_group_averages

    for i, gen_model in enumerate(Bayesian_gen_agents):
        axes[i] = plt.subplot(gridspecstrum[0, i])
        x = mle_group_averages_fixed_lambda.loc[gen_model].index.unique(level="tau_gen").values
        # x.sort()
        y = mle_group_averages_fixed_lambda.loc[gen_model]["tau_mle"]["mean"].values
        e = mle_group_averages_fixed_lambda.loc[gen_model]["tau_mle"]["std"].values
        axes[i].errorbar(x, y, alpha=0.7, markersize=6, color=color_dict[gen_model],
                    fmt='o', linestyle=None, clip_on=False,
                    label=f"{gen_model}, lambda = 0.5", yerr=e)
        axes[i].legend(loc='upper left', fontsize=14)
        plotter.config_axes(axes[i], y_label="tau_est", x_label="tau_gen",
                                xticks=np.linspace(0, 0.5, 3),
                                yticks=np.round(np.linspace(0, 0.5, 3), 1))
        axes[i].set_xticklabels(np.round(np.linspace(0, 0.5, 3), 1), fontsize=14)


    # Select group averages for fixed tau to 0.1
    mle_group_averages_diff_lambda = all_val_results.groupby(
        ["agent", "tau_gen", "lambda_gen"])[["tau_mle", "lambda_mle"]].agg(
            ["mean", "std"])  # TODO: dirty naming

    #mle_group_averages_fixed_tau = mle_group_averages_diff_lambda.loc[:, 0.1, :]  # TODO: dirty naming

    if "A3" in Bayesian_gen_agents:
        tau_values = mle_group_averages.loc["A3"].index.unique(
            level="tau_gen").values
        if len(tau_values) > 3:
            NUM_ELEM = 3
        else:
            NUM_ELEM = len(tau_values)
        idx = np.round(np.linspace(0, len(tau_values) - 1, NUM_ELEM)).astype(int)
        taus_for_plt = tau_values[idx]

        for i, tau_i in enumerate(taus_for_plt):
            axes[i] = plt.subplot(gridspecstrum[1, i])
            x = mle_group_averages_diff_lambda.loc["A3", tau_i, :].index.values.tolist()  # TODO: more elegant way!
            # x.sort()
            y = mle_group_averages_diff_lambda.loc["A3", tau_i]["lambda_mle"]["mean"].values
            e = mle_group_averages_diff_lambda.loc["A3", tau_i]["lambda_mle"]["std"].values
            axes[i].errorbar(x, y, alpha=0.7, markersize=6, color=color_dict["A3"],
                        fmt='o', linestyle=None, clip_on=False,
                        label=f"A3, tau = {tau_i}", yerr=e)
            axes[i].legend(loc='upper left', fontsize=14)
            plotter.config_axes(axes[i], y_label="lambda_est",
                                    x_label="lambda_gen",
                                    xticks=np.linspace(0.1, 1, 10),
                                    yticks=np.round(np.linspace(0.1, 1, 10), 1))
            axes[i].set_xticklabels(np.round(np.linspace(0.1, 1, 10), 1),
                                fontsize=14)

    plotter.save_figure(fig=fig, figure_filename=FIGURE_FILENAME)


if __name__ == "__main__":

    EXP_LABEL = "exp_msc"
    VERSION_NO = "test_parallel_1"
    FIGURE_FILENAME = f"figure_param_recov_{VERSION_NO}"
    N_BLOCKS = 3

    main()
