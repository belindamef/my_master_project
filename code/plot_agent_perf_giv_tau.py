"""This script plots figure 1"""

import os
import glob
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
        ["agent", "tau_gen", "lambda_gen"])["n_tr"].agg(["mean", "std"])

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
    plt, col_exp, col_A, col_C = very_plotter.get_fig_template(pyplot)
    axes = {}

    fig = plt.figure(figsize=(16, 13), layout="constrained")
    gridspecstrum = gridspec.GridSpec(5, 1)


    agent_colors = col_A + col_C  # Specify agent colors

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

    # # Bayesian agents
    # very_plotter.plot_bar(
    #     ax=this_ax, x=[1, 1.5, 2],
    #     height=grp_lvl_stats_bw_sim_agents[block]['mean_tr_over_subs'].values,
    #     colors=col_A, bar_width=half_bar_width)

    # # Control agents
    # very_plotter.plot_bar(
    #     ax=this_ax, x=[2.5, 3, 3.5],
    #     height=grp_lvl_stats_bw_sim_controls[block]['mean_tr_over_subs'].values,
    #     colors=col_C, bar_width=half_bar_width,
    #     yerr=grp_lvl_stats_bw_sim_controls[block]['std_tr_over_subs'],
    #     errorbar_size=3)

    # # Configure axis
    # very_plotter.config_axes(
    #     this_ax, y_label="Number of Treasures", y_lim=[0, n_rounds],
    #     xticks=[1, 1.5, 2, 2.5, 3, 3.5],
    #     xticklabels=['A1', 'A2', 'A3', 'C1', 'C2', 'C3'],
    #     yticks=np.linspace(0, n_rounds, 6),
    #     # ytickslabels=np.around(np.linspace(0, n_rounds, 6), 2))
    #     ytickslabels=[0, 2, 4, 6, 8, 10])

    # # Add letter and title
    # very_plotter.config_axes(
    #     axes[0], title="Task performance\n " + r"\textit{exp. run 1}")
    # very_plotter.config_axes(axes[1], title=" \n " + r"\textit{exp. run 2}")
    # very_plotter.config_axes(axes[2], title=" \n " + r"\textit{exp. run 3}")
    # axes[0].text(-0.15, 1.25, 'a', transform=axes[0].transAxes, size=32, weight='bold')

    # # ------Average choice rates--------------------------------------------
    # for block in range(n_blocks):
    #     axes[block] = plt.subplot(gridspecstrum[block, 2:4])
    #     this_ax = axes[block]
    #     block += 1

    #     # Bayesian agents
    #     very_plotter.plot_bar(
    #         ax=this_ax, x=[1, 1.5, 2],
    #         height=grp_lvl_stats_bw_sim_agents[block]['mean_drills_over_subs'].values,
    #         colors=col_A, bar_width=half_bar_width)

    #     # Control agents
    #     very_plotter.plot_bar(
    #         ax=this_ax, x=[2.5, 3, 3.5],
    #         height=grp_lvl_stats_bw_sim_controls[block]['mean_drills_over_subs'].values,
    #         colors=col_C, bar_width=half_bar_width,
    #         yerr=grp_lvl_stats_bw_sim_controls[block]['std_drills_over_subs'].values,
    #         errorbar_size=3)

    #     # Configure axis
    #     very_plotter.config_axes(
    #         this_ax, y_label="\%", y_lim=[0, 1],
    #         xticks=[1, 1.5, 2, 2.5, 3, 3.5],
    #         xticklabels=['A1', 'A2', 'A3', 'C1', 'C2', 'C3'],
    #         yticks=np.linspace(0, 1, 6),
    #         # ytickslabels=int(np.linspace(0, 100, 6), 0))
    #         # ytickslabels=np.linspace(0, 100, 6))
    #         ytickslabels=[0, 20, 40, 60, 80, 100])

    # # Add letter and title
    # very_plotter.config_axes(
    #     axes[0], title="Informative choice rates\n " + r"\textit{exp. run 1}")
    # very_plotter.config_axes(axes[1], title="\n " + r"\textit{exp. run 2}")
    # very_plotter.config_axes(axes[2], title="\n " + r"\textit{exp. run 3}")
    # # very_plotter.add_letters({1: ax[0]})

    # # ------Trial-by-trial/round-wise average choice rates------------------
    # for block in range(n_blocks):
    #     block += 1
    #     axes[block] = plt.subplot(gridspecstrum[block - 1, 4:10])

    #     # Bayesian agents
    #     i = 0
    #     for agent, ev_thisagent_df in ev_sim_bw[block].groupby('agent'):  # TODO hier geändert!! wie weiter?
    #         if 'C' in agent:  # Scip control agents
    #             continue
    #         ev_thisagent_gbround = ev_thisagent_df.groupby('round_')
    #         vlines = [(round_ * 12 - 11) for round_, ev_df in ev_thisagent_gbround]
    #         x = [((round_ * 12) - 5.5) for round_, ev_df in ev_thisagent_gbround]
    #         y = [np.mean(ev_df['action_type_num'])
    #             for round_, ev_df in ev_thisagent_gbround]
    #         e = [np.std(ev_df['action_type_num'])
    #             for round_, ev_df in ev_thisagent_gbround]
    #         this_ax.errorbar(
    #             x, y, alpha=0.7, markersize=4, color=agent_colors[i], fmt='o',
    #             linestyle='-', linewidth=0.8, clip_on=False,
    #             label=f"{agent}'s round average")
    #         i += 1

    #     # ------Configure axis------
    #     # Add vertical lines
    #     this_ax.vlines(
    #         vlines, colors=[.9, .9, .9], linewidth=.4, ymin=0, ymax=1)
    #     vlines.append(120)  # Add last boundary, to have 12 xticklabels
    #     very_plotter.config_axes(
    #         this_ax, x_lim=[0, 120], x_label='Trial', xticks=vlines,
    #         xticklabels=np.around((np.linspace(1, 120, 11))).astype(int),
    #         y_label="\%", y_lim=[0, 1], yticks=np.linspace(0, 1.0, 6),
    #         # ytickslabels=np.around(np.linspace(0, 1.0, 6), 2))
    #         ytickslabels=[0, 20, 40, 60, 80, 100])

    # # Add letter, title and legend
    # axes[1].legend(
    #     bbox_to_anchor=(1.30, 1), loc='upper right', borderaxespad=0, fontsize=13)

    # very_plotter.config_axes(
    #     axes[1],
    #     title="Trial-by-trial/roundwise informative choice rates\n "
    #         + r"\textit{exp. run 1}")
    # very_plotter.config_axes(axes[2], title="\n " + r"\textit{exp. run 2}")
    # very_plotter.config_axes(axes[3], title="\n " + r"\textit{exp. run 3}")
    # # Add letter
    # # very_plotter.add_letters({2: ax[1]})
    # axes[1].text(-0.01, 1.25, 'c', transform=axes[1].transAxes, size=32, weight='bold')


    # Print subject level descriptive figure
    fig.tight_layout()
    fig_fn = os.path.join(dir_mgr.paths.figures, FIGURE_FILENAME)
    fig.savefig(f"{fig_fn}.png", dpi=200, format='png')
    fig.savefig(f"{fig_fn}.pdf", dpi=200, format='pdf')


if __name__ == "__main__":

    EXP_LABEL = "exp_msc_test"
    FIGURE_FILENAME = "figue_1_agent_perf"
    N_BLOCKS = 3

    main()
