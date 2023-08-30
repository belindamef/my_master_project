#!/usr/bin/env python3
"""This script plots simulated agent performance"""
import numpy as np
from matplotlib import pyplot
from very_plotter_new import VeryPlotter, PlotCustomParams
from config import DirectoryManager, DataHandler


def plot_agent_perf(
        exp_label: str, vers: str, save_file: bool = True,
        plt_params: PlotCustomParams = PlotCustomParams()):

    # Get and Prepare data
    dir_mgr = DirectoryManager()
    dir_mgr.define_raw_beh_data_out_path(data_type="sim",
                                         exp_label=exp_label,
                                         version=vers)
    dir_mgr.define_processed_data_path(data_type="sim",
                                       exp_label=exp_label,
                                       vers=vers)
    dir_mgr.define_descr_stats_path(data_type="sim",
                                    exp_label=exp_label,
                                    version=vers)
    dir_mgr.define_stats_filenames()
    data_loader = DataHandler(dir_mgr.paths, exp_label=exp_label)
    subj_lvl_stats_df = data_loader.load_sim_subj_lvl_stats()

    agent_models = subj_lvl_stats_df.agent.unique().tolist()
    agent_models.sort()

    agent_performance_group_averages = subj_lvl_stats_df.groupby(
        ["agent", "tau_gen", "lambda_gen"],
        dropna=False)["mean_tr_over_blocks"].agg(
                                                 ["mean", "std"])

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

    # Prepare figure
    plotter = VeryPlotter(paths=dir_mgr.paths)
    plt = pyplot
    color_dict = plotter.get_agent_colors(control_color="grey")

    rc_params = plotter.define_run_commands()
    plt = pyplot
    plt.rcParams.update(rc_params)
    fig, axs = plt.subplots(nrows=2, ncols=1,
                            figsize=(13, 14),
                            layout="constrained")

    # Adujust axis parameters
    tau_gen_all = agent_performance_group_averages.index.get_level_values(
        level="tau_gen").unique().values
    tau_gen_all = tau_gen_all[~np.isnan(tau_gen_all)]
    tau_gen_all.sort()
    plt_params.define_tau_tick_labels(max_tau_value=max(tau_gen_all),
                                      n_values=11)
    # ------Figure A: A1, A2----------------------------------------

    row = 0
    this_ax = axs[row]

    for bayesian_agent in ["A1", "A2"]:
        tau_gen_values = agent_performance_group_averages.loc[
            bayesian_agent].index.get_level_values(
            level="tau_gen").unique().values

        means_n_tr = agent_n_tr_means[bayesian_agent]
        stds_n_tr = agent_n_tr_stds[bayesian_agent]

        this_ax.errorbar(
            x=tau_gen_values, y=means_n_tr, yerr=stds_n_tr,
            fmt=plt_params.marker_shape, color=color_dict[bayesian_agent],
            alpha=plt_params.transp_lvl, markersize=plt_params.marker_sz,
            linestyle=plt_params.err_bar_linestyle,
            linewidth=plt_params.err_bar_linewidth,
            label=bayesian_agent)


    # Control agents
    for control_agent in ["C1", "C2", "C3"]:

        means_n_tr = agent_n_tr_means[control_agent]
        means_n_tr = list(means_n_tr) * len(tau_gen_values)
        std_n_tr = agent_n_tr_stds[control_agent]

        this_ax.plot(
            tau_gen_values, means_n_tr,
            color=color_dict[control_agent],
            linestyle=plt_params.c_agent_linestyle,
            linewidth=plt_params.c_agent_linewidth,
            label=control_agent)

        this_ax.fill_between(
            tau_gen_values, means_n_tr-std_n_tr, means_n_tr+std_n_tr,
            color=color_dict[control_agent],
            alpha=plt_params.c_agent_std_transp_lvl)

    plotter.config_axes(
        this_ax,
        y_label=r"\textit{N} treasures",
        x_label=r"$\tau$",
        axix_label_size=plt_params.axis_label_fs,
        ticksize=plt_params.axis_tick_fs,
        title_font=plt_params.axis_title_fs,
        xticks=plt_params.tau_ticks,
        xticklabels=plt_params.tau_ticks,
        yticks=plt_params.n_tr_ticks,
        ytickslabels=plt_params.n_tr_ticks
        )

    this_ax.legend(loc='upper right', fontsize=plt_params.legend_fs,
                   bbox_to_anchor=(1, 1))

    # ------Figure B----------------------------------------

    row = 1
    a3_colors = plotter.define_a3_colors()

    this_ax = axs[row]

    for index, lambda_gen in enumerate(lambda_gen_values):
        agent_n_tr_means["A3"] = agent_performance_group_averages.loc[
            :, :, lambda_gen].loc["A3"]["mean"].values
        agent_n_tr_stds["A3"] = agent_performance_group_averages.loc[
            :, :, lambda_gen].loc["A3"]["std"].values

        tau_gen_values_this_lambda = agent_performance_group_averages.loc[
            "A3", :, lambda_gen].index.unique(level="tau_gen").values

        tau_gen_values_this_lambda = tau_gen_values_this_lambda[~np.isnan(tau_gen_values_this_lambda)]
        tau_gen_values_this_lambda.sort()
        means_n_tr = agent_n_tr_means["A3"]
        stds_n_tr = agent_n_tr_stds["A3"]

        this_ax.errorbar(            
            x=tau_gen_values_this_lambda, y=means_n_tr, yerr=stds_n_tr,
            fmt=plt_params.marker_shape, color=a3_colors[index],
            alpha=plt_params.transp_lvl, markersize=plt_params.marker_sz,
            linestyle=plt_params.err_bar_linestyle,
            linewidth=plt_params.err_bar_linewidth,
            label=r"$\lambda$ = " + f"{lambda_gen}")

    this_ax.legend(loc="upper right", fontsize=plt_params.legend_fs,
                   bbox_to_anchor=(1, 1.1))

    plotter.config_axes(
        this_ax,
        y_label=r"\textit{N} treasures",
        x_label=r"$\tau$",
        axix_label_size=plt_params.axis_label_fs,
        ticksize=plt_params.axis_tick_fs,
        title_font=plt_params.axis_title_fs,
        xticks=plt_params.tau_ticks,
        xticklabels=plt_params.tau_ticks,
        yticks=plt_params.n_tr_ticks,
        ytickslabels=plt_params.n_tr_ticks
        )
    # Print subject level descriptive figure
    if save_file:
        plotter.save_figure(fig=fig, figure_filename=FIGURE_FILENAME)


if __name__ == "__main__":

    EXP_LABEL = "exp_msc"
    VERSION = "50parts_new"
    FIGURE_FILENAME = f"figue_1_agent_perf_{VERSION}"

    plot_agent_perf(exp_label=EXP_LABEL, vers=VERSION, save_file=True)
