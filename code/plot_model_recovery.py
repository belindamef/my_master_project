"""This script plots model recovery performance results: BICs"""
import os
import glob
import math
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot
from utilities.very_plotter_new import VeryPlotter
import pandas as pd
from utilities.config import DirectoryManager, DataLoader


def main():
    # Prepare data
    dir_mgr = DirectoryManager()
    dir_mgr.define_val_out_path(dir_label=EXP_LABEL, version=VERSION_NO)
    data_loader = DataLoader(dir_mgr.paths, EXP_LABEL)
    all_bics_df = data_loader.load_data_in_folder(
        folder_path=dir_mgr.paths.this_val_out_dir
        )

    agent_gen_models = all_bics_df.agent.unique().tolist()
    agent_gen_models.sort()

    control_gen_agents = [agent for agent in agent_gen_models if "C" in agent]
    Bayesian_gen_agents = [agent for agent in agent_gen_models if "A" in agent]
    n_agents = len(agent_gen_models)

    bic_of_analizing_models = [
        col_name for col_name in all_bics_df.columns if "BIC" in col_name]
    analyzing_models = [
        bic_model[-2:] for bic_model in bic_of_analizing_models
        ]
    bic_group_averages = all_bics_df.groupby(
        ["agent", "tau_gen", "lambda_gen"],
        dropna=False)[bic_of_analizing_models].agg(
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
    lambda_gen_values = np.delete(
        all_bics_df["lambda_gen"].unique(),
        np.where(np.isnan(all_bics_df.lambda_gen.unique())))
    lambda_gen_values.sort()


    color_dict = {"C1": col_controls[0],
                  "C2": col_controls[1],
                  "C3": col_controls[2],
                  "A1": col_agents[0],
                  "A2": col_agents[1],
                  "A3": col_agents[2]}
    axes = {}
    fig = plt.figure(figsize=(26, 9), layout="constrained")
    gridspecstrum = gridspec.GridSpec(2, n_agents-1)

# ------ Figure A --------------------------------
    i = 0
    for gen_agent in control_gen_agents:
        axes[i] = plt.subplot(gridspecstrum[0, i])
        x_ticks = range(1, len(analyzing_models)+1)
        x_tick_labels = []
        colors = []

        j = 0
        for analyzing_model in analyzing_models:

            x_tick_labels.append(analyzing_model)

            colors.append(color_dict[analyzing_model])
            axes[i].errorbar(j+1,
                             np.mean(bic_group_averages.loc[gen_agent][
                                       f"BIC_{analyzing_model}", "mean"]),
                             alpha=0.7, markersize=7,
                             color=color_dict[analyzing_model],
                             fmt='o', linestyle=None, clip_on=False,
                             label=analyzing_model,
                             yerr=np.mean(bic_group_averages.loc[
                                gen_agent][
                                    f"BIC_{analyzing_model}", "std"]))
            j += 1

        plotter.config_axes(ax=axes[i],
                            title=f"{gen_agent}",
                            title_color=color_dict[f"{gen_agent}"],
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
        i += 1

    i = len(control_gen_agents)

    tau_gen_values = np.delete(  # TODO: redundant?
        all_bics_df.tau_gen.unique(),
        np.where(np.isnan(all_bics_df.tau_gen.unique())))

    for gen_agent in [agent for agent in Bayesian_gen_agents if agent != "A3"]:
        axes[i] = plt.subplot(gridspecstrum[0, i])
        tau_gen_values = np.array(
            bic_group_averages.loc[gen_agent].index.get_level_values("tau_gen")
            )

        for analyzing_model in analyzing_models:
            
            bic_values_this_analyzing_agent = bic_group_averages.loc[
                gen_agent][f"BIC_{analyzing_model}", "mean"]

            stds = (np.mean(bic_group_averages.loc[
                        gen_agent][f"BIC_{analyzing_model}", "std"])
                    )
            this_agents_color = color_dict[analyzing_model]
            axes[i].errorbar(
                tau_gen_values,
                bic_values_this_analyzing_agent,
                alpha=0.7, markersize=7,
                color=this_agents_color,
                fmt='o',
                linestyle='-', linewidth=1,
                clip_on=False,
                label=f"{analyzing_model}",
                yerr=stds
                )

        plotter.config_axes(ax=axes[i],  # y_label="tau_est", x_label="tau_gen",
                            title=f"{gen_agent}",
                            title_color=color_dict[f"{gen_agent}"],
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
                            xticks=np.round(tau_gen_values, decimals=2),
                            xticklabels=np.round(tau_gen_values, decimals=2),
                            x_label=r"$\tau$",
                            title_font=20,
                            axix_label_size=22,
                            ticksize=18,
                            )
        i += 1

    axes[0].legend(loc="lower right", fontsize=14)

# ------ Figure B-------------------------------------------------
    i = 0
    if "A3" in bic_group_averages.index.get_level_values("agent"):
        for lambda_gen in lambda_gen_values:
            axes[i] = plt.subplot(gridspecstrum[1, i])
            tau_gen_values = np.array(
                bic_group_averages.loc[
                    "A3", :, lambda_gen].index.get_level_values("tau_gen"))

            for analyzing_model in analyzing_models:
                
                bic_values_this_analyzing_agent = bic_group_averages.loc[
                    "A3",: , lambda_gen][f"BIC_{analyzing_model}", "mean"]

                stds = (np.mean(bic_group_averages.loc[
                            "A3"][f"BIC_{analyzing_model}", "std"])
                        )
                this_agents_color = color_dict[analyzing_model]
                axes[i].errorbar(
                    tau_gen_values,
                    bic_values_this_analyzing_agent,
                    alpha=0.7, markersize=7,
                    color=this_agents_color,
                    fmt='o',
                    linestyle='-', linewidth=1,
                    clip_on=False,
                    label=f"{analyzing_model}",
                    yerr=stds
                    )

            plotter.config_axes(ax=axes[i],
                                title=r"A3 $\lambda= $" + f"{lambda_gen}",
                                title_color=color_dict[f"A3"],
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
                                xticks=np.round(tau_gen_values, decimals=2),
                                xticklabels=np.round(tau_gen_values, decimals=2),
                                x_label=r"$\tau$",
                                title_font=20,
                                axix_label_size=22,
                                ticksize=18,
                                )
            i += 1

    plotter.config_axes(ax=axes[0], y_label="BIC", axix_label_size=22)
    fig.align_ylabels(axs=list(axes.values()))

    plotter.save_figure(fig=fig, figure_filename=FIGURE_FILENAME)

if __name__ == "__main__":

    EXP_LABEL = "exp_msc_test_allagents"
    VERSION_NO = 1
    FIGURE_FILENAME = f"figure_model_recov_test_{VERSION_NO}"
    N_BLOCKS = 3

    main()
