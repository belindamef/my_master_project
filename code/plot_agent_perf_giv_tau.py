"""This script plots figure 1"""

import os
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot
from utilities import very_plotter
from utilities.config import DirectoryManager, Paths


class DataLoader:
    def __init__(self, paths: Paths, exp_label):
        self.paths = paths
        self.exp_label = exp_label

    def load_sim_subj_lvl_stats(self) -> pd.DataFrame:
        subj_lvl_stats_df = pd.read_pickle(
            f"{self.paths.subj_lvl_descr_stats_fn}.pkl")
        return subj_lvl_stats_df


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
        ["agent", "tau_gen", "lambda_gen"])["mean_tr_over_blocks"].agg(
        ["mean", "std"])

    tau_gen_values = agent_performance_group_averages.index.unique(
        level="tau_gen").values
    lambda_gen_values = agent_performance_group_averages.index.unique(
        level="lambda_gen").values

    width = 0.014
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
    # Create general figure components

    # Initialize figure
    plt, col_exp, col_agents, col_controls = very_plotter.get_fig_template(
        pyplot)
    axes = {}

    fig = plt.figure(figsize=(16, 13), layout="constrained")
    gridspecstrum = gridspec.GridSpec(5, 1)

    agent_colors = col_agents + col_controls  # Specify agent colors

    # ------Treasure discovery----------------------------------------–

    for index, lambda_gen in enumerate(lambda_gen_values):

        agent_n_tr_means["A3"] = agent_performance_group_averages.loc[
            :, :, lambda_gen].loc["A3"]["mean"].values
        agent_n_tr_stds["A3"] = agent_performance_group_averages.loc[
            :, :, lambda_gen].loc["A3"]["std"].values

        axes[index] = plt.subplot(gridspecstrum[index, 0])
        this_ax = axes[index]

        multiplier = 0
        color_index = 0
        for agent, means in agent_n_tr_means.items():

            stds = agent_n_tr_stds[agent]
            offset = width * multiplier

            very_plotter.plot_bar(
                ax=this_ax, x=tau_gen_values+offset,
                height=means, bar_width=width,
                colors=agent_colors[color_index], yerr=stds,
                labels=agent, errorbar_size=0.4)

            multiplier += 1
            color_index += 1

        very_plotter.config_axes(this_ax, y_label=r"\textit{N} treasures",
                                 x_label=r"$\tau$",
                                 # title=f"A3_lambda = {lambda_gen}",
                                 title_font=20,
                                 axix_label_size=20,
                                 ticksize=16,
                                 xticks=tau_gen_values + width,
                                 xticklabels=tau_gen_values,
                                 yticks=np.arange(0, 11, 2),
                                 ytickslabels=np.arange(0, 11, 2),
                                 y_lim=(0, 10))

        this_ax.set_title(label=r"A3 $\lambda$ = " + f"{lambda_gen}",
                          loc="left", fontsize=20)

    axes[0].legend(loc='upper right', fontsize=15)

    # Print subject level descriptive figure
    fig.tight_layout()
    fig_fn = os.path.join(dir_mgr.paths.figures, FIGURE_FILENAME)
    fig.savefig(f"{fig_fn}.png", dpi=200, format='png')
    fig.savefig(f"{fig_fn}.pdf", dpi=200, format='pdf')


if __name__ == "__main__":

    EXP_LABEL = "exp_msc"
    FIGURE_FILENAME = "figue_1_agent_perf"
    N_BLOCKS = 3

    main()
