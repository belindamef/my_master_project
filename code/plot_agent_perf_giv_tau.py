"""This script plots simulated agent performance"""

import os
import numpy as np
import pandas as pd
from utilities.very_plotter_new import VeryPlotter
from utilities.config import DirectoryManager, DataLoader
from matplotlib import gridspec
import palettable


def main():
    dir_mgr = DirectoryManager()
    dir_mgr.define_raw_beh_data_out_path(data_type="sim",
                                         out_dir_label=EXP_LABEL,
                                         make_dir=False)
    dir_mgr.define_processed_data_path(data_type="sim",
                                       dir_label=EXP_LABEL,
                                       make_dir=True)
    dir_mgr.define_descr_stats_path(data_type="sim",
                                    dir_label=EXP_LABEL,
                                    make_dir=True)
    dir_mgr.define_stats_filenames()

    data_loader = DataLoader(dir_mgr.paths, EXP_LABEL)
    subj_lvl_stats_df = data_loader.load_sim_subj_lvl_stats()

    # Prepare group level stats
    agent_models = subj_lvl_stats_df.agent.unique().tolist()
    agent_models.sort()

    agent_performance_group_averages = subj_lvl_stats_df.groupby(
        ["agent", "tau_gen", "lambda_gen"], dropna=False)["mean_tr_over_blocks"].agg(
        ["mean", "std"])

    tau_gen_values = agent_performance_group_averages.index.unique(
        level="tau_gen").values
    tau_gen_values = tau_gen_values[~np.isnan(tau_gen_values)]
    lambda_gen_values = agent_performance_group_averages.index.unique(
        level="lambda_gen").values
    lambda_gen_values = lambda_gen_values[~np.isnan(lambda_gen_values)]
    lambda_gen_values.sort()

    agent_n_tr_means = {}
    agent_n_tr_stds = {}

    for agent in agent_models:
        agent_n_tr_means[agent] = agent_performance_group_averages.loc[
            agent]["mean"].values
        agent_n_tr_stds[agent] = agent_performance_group_averages.loc[
            agent]["std"].values
    # ----------------------------------------------------------
    #       Prepare figure
    # ----------------------------------------------------------
    plotter = VeryPlotter(paths=dir_mgr.paths)
    plt = plotter.get_pyplot_object()

    col_agents, col_controls = plotter.get_agent_colors(
        control_color="grey")
    axes = {}

    fig = plt.figure(figsize=(16, 20), layout="constrained")
    gridspecstrum = gridspec.GridSpec(2, 1)
    axes[0] = plt.subplot(gridspecstrum[0, 0])
    axes[1] = plt.subplot(gridspecstrum[1, 0])

    agent_colors = col_agents + col_controls  # Specify agent colors

    # ------Figure A----------------------------------------–
    color_index = 0

    # A1 and A2
    for bayesian_agent in ["A1", "A2"]:
        tau_gen_values = agent_performance_group_averages.loc[
            bayesian_agent, :, :].index.unique(level="tau_gen").values
        tau_gen_values = tau_gen_values[~np.isnan(tau_gen_values)]

        means = agent_n_tr_means[bayesian_agent]
        stds = agent_n_tr_stds[bayesian_agent]

        axes[0].errorbar(
            tau_gen_values, means, alpha=0.7, markersize=5,
            color=agent_colors[color_index], fmt='o',
            linestyle='-', linewidth=1,
            label=bayesian_agent,
            clip_on=False, yerr=stds)

        color_index += 1

    # Control agents
    color_index = 3
    for control_agent in ["C1", "C2", "C3"]:

        tau_gen_values = agent_performance_group_averages.index.unique(
            level="tau_gen").values
        tau_gen_values = tau_gen_values[~np.isnan(tau_gen_values)]
        tau_gen_values.sort()
        
        means = agent_n_tr_means[control_agent]
        means = list(means) * len(tau_gen_values)
        std = agent_n_tr_stds[control_agent]

        # axes[0].hlines(y=mean, xmin=min(tau_gen_values),
        #           xmax=max(tau_gen_values), linewidth=1,
        #           color=agent_colors[color_index])

        axes[0].plot(tau_gen_values, means,
                     color=agent_colors[color_index],
                     linestyle='-', linewidth=0.8, label=control_agent)
        axes[0].fill_between(tau_gen_values, means-std, means+std,
                             color=agent_colors[color_index],
                             alpha=0.2)

    plotter.config_axes(
        axes[0], y_label=r"\textit{N} treasures",
        x_label=r"$\tau$",
        # title=f"A3_lambda = {lambda_gen}",
        title_font=20,
        axix_label_size=28,
        ticksize=18,
        xticks=np.round(np.arange(0, max(tau_gen_values) + 0.05, 0.05), 2),
        xticklabels=np.round(np.arange(
            0, max(tau_gen_values) + 0.05, 0.05), 2),
        yticks=np.arange(0, 11, 2),
        ytickslabels=np.arange(0, 11, 2),
        y_lim=(0, 10))

    axes[0].legend(loc='upper right', fontsize=15)

    # ------Figure B----------------------------------------–

    color_indices = np.flip(np.round(np.linspace(3, 19, 11)))
    color_indices = np.round(color_indices)
    viridis_20 = palettable.matplotlib.Viridis_20.colors
    a3_viridis_colors = [viridis_20[int(i)] for i in color_indices]
    a3_colors = [
        [value / 255 for value in list_]
        for list_ in a3_viridis_colors]

    for index, lambda_gen in enumerate(lambda_gen_values):
        agent_n_tr_means["A3"] = agent_performance_group_averages.loc[
            :, :, lambda_gen].loc["A3"]["mean"].values
        agent_n_tr_stds["A3"] = agent_performance_group_averages.loc[
            :, :, lambda_gen].loc["A3"]["std"].values

        axes[1] = plt.subplot(gridspecstrum[1, 0])

        tau_gen_values = agent_performance_group_averages.loc[
            "A3", :, lambda_gen].index.unique(level="tau_gen").values

        tau_gen_values = tau_gen_values[~np.isnan(tau_gen_values)]
        tau_gen_values.sort()
        means = agent_n_tr_means["A3"]
        stds = agent_n_tr_stds["A3"]

        axes[1].errorbar(
            tau_gen_values, means, alpha=0.7, markersize=5,
            color=a3_colors[index],
            fmt='o',
            linestyle='-', linewidth=1,
            label=r"$\lambda$ = " + f"{lambda_gen}",
            clip_on=False, yerr=stds)

    tau_gen_values = agent_performance_group_averages.loc[
        "A3", :, :].index.unique(level="tau_gen").values
    axes[1].legend(loc="upper right", fontsize=18)
        # axex[1].set_title(label=r"A3 $\lambda$ = " + f"{lambda_gen}",
        #                     loc="left", fontsize=20)

    plotter.config_axes(axes[1], y_label=r"\textit{N} treasures",
                                x_label=r"$\tau$",
                                # title=f"A3_lambda = {lambda_gen}",
                                title_font=20,
                                axix_label_size=28,
                                ticksize=18,
                                xticks=np.round(np.arange(0, max(tau_gen_values) + 0.05, 0.05), 2),
                                xticklabels=np.round(np.arange(0, max(tau_gen_values)+0.05, 0.05), 2),
                                yticks=np.arange(0, 11, 2),
                                ytickslabels=np.arange(0, 11, 2),
                                y_lim=(0, 10))

    # Print subject level descriptive figure
    fig.tight_layout()
    fig_fn = os.path.join(dir_mgr.paths.figures, FIGURE_FILENAME)
    fig.savefig(f"{fig_fn}.png", dpi=200, format='png')
    fig.savefig(f"{fig_fn}.pdf", dpi=200, format='pdf')


if __name__ == "__main__":

    EXP_LABEL = "exp_msc_50parts_new"
    FIGURE_FILENAME = "figue_1_agent_perf_50parts_new"
    N_BLOCKS = 3

    main()
