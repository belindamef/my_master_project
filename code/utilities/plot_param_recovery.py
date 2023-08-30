#!/usr/bin/env python3
"""This script plots figure 2"""
import argparse
import numpy as np
from matplotlib import pyplot
from utilities.very_plotter_new import VeryPlotter, PlotCustomParams
from utilities.config import DirectoryManager, DataHandler


def plot_param_recov_results(
        exp_label: str, vers: str, save_file: bool = True,
        plt_params: PlotCustomParams = PlotCustomParams()):

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
    color_dict = plotter.get_agent_colors()

    rc_params = plotter.define_run_commands()
    plt = pyplot
    plt.rcParams.update(rc_params)
    fig, axs = plt.subplots(nrows=2, ncols=3,
                            figsize=(13, 8), layout="constrained")

    # ------First row------------------
    row = 0
    for column, gen_model in enumerate(bayesian_gen_agents):
        this_ax = axs[row, column]

        tau_gen_values = mle_grp_avrgs.loc[gen_model].index.unique(
            level="tau_gen").values
        tau_est_values = mle_grp_avrgs.loc[gen_model]["tau_mle"]["mean"].values
        stds = mle_grp_avrgs.loc[gen_model]["tau_mle"]["std"].values

        this_ax.errorbar(
            x=tau_gen_values, y=tau_est_values, yerr=stds,
            fmt=plt_params.marker_shape, color=color_dict[gen_model],
            alpha=plt_params.transp_lvl, markersize=plt_params.marker_sz,
            label=f"{gen_model}")

        this_ax.legend(loc='upper left', fontsize=plt_params.legend_fs)

        plotter.config_axes(
            this_ax,
            y_label=r"$\hat{\tau}$",
            x_label=r"$\tau$",
            xticks=plt_params.tau_ticks,
            xticklabels=plt_params.tau_ticks,
            yticks=plt_params.tau_ticks,
            ytickslabels=plt_params.tau_ticks,
            ticksize=plt_params.axis_tick_fs,
            axix_label_size=plt_params.axis_label_fs
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

            tau_gen_values = mle_grp_avrgs_diff_lambdas.index.get_level_values(
                "lambda_gen").tolist()
            tau_est_values = mle_grp_avrgs_diff_lambdas.loc[
                "A3", tau_i]["lambda_mle"]["mean"].values
            stds = mle_grp_avrgs_diff_lambdas.loc[
                "A3", tau_i]["lambda_mle"]["std"].values

            this_ax.errorbar(
                x=tau_gen_values, y=tau_est_values, yerr=stds,
                fmt=plt_params.marker_shape, color=color_dict[gen_model],
                alpha=plt_params.transp_lvl, markersize=plt_params.marker_sz,
                label=f"{gen_model}, " + r"$\tau = $" + f"{tau_i}")

            this_ax.legend(loc='upper left', fontsize=plt_params.legend_fs)

            plotter.config_axes(
                this_ax,
                y_label=r"$\hat{\lambda}$",
                x_label=r"$\lambda$",
                xticks=plt_params.lambda_ticks,
                xticklabels=plt_params.lambda_ticks,
                yticks=plt_params.lambda_ticks,
                ytickslabels=plt_params.lambda_ticks,
                ticksize=plt_params.axis_tick_fs,
                axix_label_size=plt_params.axis_label_fs)

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

    SAVE_FILE = args.dont_save_file

    plot_param_recov_results(save_file=SAVE_FILE, exp_label=EXP_LABEL,
                             vers=VERSION_NO)
