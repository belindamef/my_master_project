import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os
import glob
import utilities.very_plotter as very_plotter
import utilities.config as config

"""This script plots figure 2"""


# Specify directories and filenames
paths = config.Paths()
fig_fn = os.path.join(paths.figures, 'figure_2.png')

# Load data
data_fn = os.path.join(paths.sim_data, "test", "param_recovery.tsv" )
data = pd.read_csv(data_fn, sep="\t")

# ----------------------------------------------------------
#       Prepare figure
# ----------------------------------------------------------

# Initialize figure
plt, col_exp, col_A, col_C = very_plotter.get_fig_template(plt)
fig = plt.figure(figsize=(20, 6))
gs = gridspec.GridSpec(1, 5)
ax = {}
s = 14
agent_colors = col_A + col_C  # Specify agent colors

# ------Trial-by-trial/round-wise average choice rates------------------

x = data.tau_gen.values
for i, gen_model in enumerate(["C2", "C3", "A1", "A2", "A3"]):
    ax[i] = plt.subplot(gs[0,i])
    y = data[f"{gen_model}_mle_group_mean"].values
    e = data[f"{gen_model}_mle_group_sd"].values
    ax[i].errorbar(x, y, alpha=0.7, markersize=4, color=agent_colors[i+1],
                fmt='o', linestyle=None, clip_on=False,
                label=f"{gen_model}", yerr=e)
    ax[i].legend(loc='upper right')
    very_plotter.config_axes(ax[i], y_label="tau_est", x_label="tau_gen",
                             xticks=np.linspace(0.25,2,10),
                             yticks=np.round(np.linspace(0.25,2,10),1))
    ax[i].set_xticklabels(np.round(np.linspace(0.25,2,10),1), fontsize=10)
    ax[i].set_xticklabels(np.round(np.linspace(0.25, 2, 10),1), fontsize=10)

# Print subject level descriptive figure
fig.tight_layout()
fig.savefig(fig_fn, dpi=200, format='png')
