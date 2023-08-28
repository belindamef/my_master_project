"""This script plots figure 2"""
import os
import glob
import numpy as np
import re
from matplotlib import gridspec
from utilities.very_plotter import VeryPlotter
import pandas as pd
import utilities.config as config


def extract_number(f):
    s = re.findall("\d+$", f)
    return (int(s[0]) if s else -1, f)


def find_most_recent_data_dir(val_results_paths: str) -> str:
    all_dirs = glob.glob(os.path.join(val_results_paths, "*"))
    most_recent_data = max(all_dirs, key=extract_number)
    return most_recent_data


# Specify directories and filenames
paths = config.Paths()
fig_fn = os.path.join(paths.figures, 'figure_param_recov_test_parallel_1.png')


data_dir = find_most_recent_data_dir(paths.model_recov_results)

all_files = glob.glob(os.path.join(data_dir, "*"))

mle_df = pd.concat(
    (pd.read_csv(f, sep="\t") for f in all_files), ignore_index=True)

agent_models = mle_df.agent.unique().tolist()
agent_models.sort()
mle_group_averages = mle_df.groupby(
    ["agent", "tau_gen"])[["tau_mle", "lambda_mle"]].agg(
        ["mean", "std"])

# # Load data
# data_fn = os.path.join(paths.sim_data, "test", "param_recovery.tsv")
# data = pd.read_csv(data_fn, sep="\t")

# ----------------------------------------------------------
#       Prepare figure
# ----------------------------------------------------------

# Initialize figure
plotter = VeryPlotter(paths=dir_mgr.paths)
plt = plotter.get_pyplot_object()
col_exp = plotter.get_exp_group_colors()
col_A, col_C = plotter.get_agent_colors()
fig = plt.figure(figsize=(25, 10))
gs = gridspec.GridSpec(2, 10)
ax = {}
SHAPE_SIZE = 14
agent_colors = col_A + col_C  # Specify agent colors

# ------Trial-by-trial/round-wise average choice rates------------------

# Select group averages for fixed lambda to 0.5
# TODO: Choose to either select it here or during data generation
# mle_group_averages_fixed_lambda = mle_group_averages.loc[:, :, 0.5]
mle_group_averages_fixed_lambda = mle_group_averages

for i, gen_model in enumerate(agent_models):
    ax[i] = plt.subplot(gs[0, (i*2):(i*2+2)])
    x = mle_group_averages_fixed_lambda.loc[gen_model].index.unique(level="tau_gen").values
    # x.sort()
    y = mle_group_averages_fixed_lambda.loc[gen_model]["tau_mle"]["mean"].values
    e = mle_group_averages_fixed_lambda.loc[gen_model]["tau_mle"]["std"].values
    ax[i].errorbar(x, y, alpha=0.7, markersize=4, color=agent_colors[i+1],
                   fmt='o', linestyle=None, clip_on=False,
                   label=f"{gen_model}, lambda = 0.5", yerr=e)
    ax[i].legend(loc='upper right')
    plotter.config_axes(ax[i], y_label="tau_est", x_label="tau_gen",
                             xticks=np.linspace(0.25, 2, 10),
                             yticks=np.round(np.linspace(0.25, 2, 10), 1))
    ax[i].set_xticklabels(np.round(np.linspace(0.25, 2, 10), 1), fontsize=10)


# Select group averages for fixed tau to 0.1
mle_group_averages_diff_lambda = mle_df.groupby(
    ["agent", "tau_gen", "lambda_gen"])[["tau_mle", "lambda_mle"]].agg(
        ["mean", "std"])  # TODO: dirty naming

mle_group_averages_fixed_tau = mle_group_averages_diff_lambda.loc[:, 0.1, :]  # TODO: dirty naming

if "A3" in agent_models:
    tau_values = mle_group_averages.loc["A3"].index.unique(
        level="tau_gen").values
    if len(tau_values) > 6:
        NUM_ELEM = 6
    else:
        NUM_ELEM = len(tau_values)
    idx = np.round(np.linspace(0, len(tau_values) - 1, NUM_ELEM)).astype(int)
    taus_for_plt = tau_values[idx]

    for i, tau_i in enumerate(tau_values):
        ax[i] = plt.subplot(gs[1, i*2:(i*2+2)])
        x = mle_group_averages_diff_lambda.loc["A3", tau_i, :].index.values.tolist()  # TODO: more elegant way!
        # x.sort()
        y = mle_group_averages_diff_lambda.loc["A3", tau_i]["lambda_mle"]["mean"].values
        e = mle_group_averages_diff_lambda.loc["A3", tau_i]["lambda_mle"]["std"].values
        ax[i].errorbar(x, y, alpha=0.7, markersize=4, color=agent_colors[i+1],
                       fmt='o', linestyle=None, clip_on=False,
                       label=f"A3, tau = {tau_i}", yerr=e)
        ax[i].legend(loc='upper right')
        plotter.config_axes(ax[i], y_label="lambda_est",
                                 x_label="lambda_gen",
                                 xticks=np.linspace(0.1, 1, 10),
                                 yticks=np.round(np.linspace(0.1, 1, 10), 1))
        ax[i].set_xticklabels(np.round(np.linspace(0.1, 1, 10), 1),
                              fontsize=10)

# Print subject level descriptive figure
fig.tight_layout()
fig.savefig(fig_fn, dpi=200, format='png')
