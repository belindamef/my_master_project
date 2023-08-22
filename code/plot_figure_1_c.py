import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os
import glob
from code.utilities.very_plotter import VeryPlotter

"""This script plots figure 1, only part c"""

# Get task configuration
#exp_label = str(input("Enter exp_label (e.g. 'exp_msc' or 'test'): "))
# sim_label = str(input("Enter sim_label (e.g. 'sim_exp_msc' or 'test'): "))
exp_label = "exp_msc"

dim = 5
n_blocks = 3

# Specify directories and filenames
working_dir = os.getcwd()
project_dir = os.sep.join(working_dir.split(os.sep)[:4])
data_dir = os.path.join(project_dir, 'data')
results_dir = os.path.join(project_dir, 'results')
descr_stats_dir = os.path.join(results_dir, 'descr_stats')
figures_dir = os.path.join(project_dir, 'figures')
fig_fn = os.path.join(figures_dir, 'figure_1_c_1.pdf')

exp_data_dir = os.path.join(data_dir, 'rawdata', 'exp', f'{exp_label}')
sim_data_dir = os.path.join(data_dir, 'rawdata', 'sim', f'sim_{exp_label}')
exp_proc_data_dir = os.path.join(
    data_dir, 'processed_data', 'exp', f'{exp_label}')
sim_proc_data_dir = os.path.join(
    data_dir, 'processed_data', 'sim', f'sim_{exp_label}')
sim_100_proc_data_dir = os.path.join(
    data_dir, 'processed_data', 'sim', f'sim_100_msc')

ev_exp_fn = os.path.join(exp_proc_data_dir, 'sub-all_task-th_run-all_beh')
ev_sim_fn = os.path.join(sim_proc_data_dir, 'sub-all_task-th_run-all_beh')
ev_exp_run_fn = os.path.join(exp_proc_data_dir, 'sub-all_task-th_run-')
ev_sim_run_fn = os.path.join(sim_proc_data_dir, 'sub-all_task-th_run-')

ds_exp_fn = os.path.join(descr_stats_dir, 'exp', f'{exp_label}', 'descr_stats')
ds_sim_fn = os.path.join(descr_stats_dir, 'sim', f'sim_{exp_label}',
                         'descr_stats')
ds_sim_100_fn = os.path.join(
    descr_stats_dir, 'sim', f'sim_100_msc', 'descr_stats')
grp_stats_exp_fn = os.path.join(
    descr_stats_dir, 'exp', f'{exp_label}', 'grp_lvl_stats')
grp_stats_sim_fn = os.path.join(
    descr_stats_dir, 'sim', f'sim_{exp_label}', 'grp_lvl_stats')
grp_stats_sim_100_fn = os.path.join(
    descr_stats_dir, 'sim', f'sim_100_msc', 'grp_lvl_stats')
tw_exp_fn = os.path.join(
    descr_stats_dir, 'exp', f'{exp_label}', 't_wise_stats')
tw_sim_fn = os.path.join(
    descr_stats_dir, 'sim', f'sim_{exp_label}', 't_wise_stats')
tw_sim_100_fn = os.path.join(
    descr_stats_dir, 'sim', f'sim_100_msc', 't_wise_stats')

# ----------------------------------------------------------
#       Load data
# ----------------------------------------------------------

# ------Overall data---------------------
# Experimental data
exp_ev_all_subs_df = pd.read_pickle(f'{ev_exp_fn}.pkl')

# Simulation 100 data
grp_lvl_stats_sim_100 = pd.read_pickle(f'{grp_stats_sim_100_fn}.pkl')
grp_lvl_stats_sim_100_A = grp_lvl_stats_sim_100[
    grp_lvl_stats_sim_100['sub_id'].isin(
        ['Agent A1', 'Agent A2', 'Agent A3'])]
grp_lvl_stats_sim_100_C = grp_lvl_stats_sim_100[
    grp_lvl_stats_sim_100['sub_id'].isin(
        ['Agent C1', 'Agent C2', 'Agent C3'])]
tw_sim_100_aw = {}  # trial wise stats each agent over all blocks
for agent in ['A1', 'A2', 'A3']:
    tw_sim_100_aw[agent] = pd.read_pickle(
        f'{tw_sim_100_fn}_agent-Agent {agent}.pkl')

# ------Blockwise data------------------
# Experimental data
descr_stats_exp_bw = {}
tw_exp_bw = {}
descr_stats_all_subs_bw_exp = {}
grp_lvl_stats_bw_exp = {}

# Simulation main data
ev_sim_bw = {}
grp_lvl_stats_bw_sim = {}
grp_lvl_stats_bw_sim_A = {}
grp_lvl_stats_bw_sim_C = {}

for block_ in range(n_blocks):
    this_block = block_ + 1

    # Experimental data
    descr_stats_exp_bw[this_block] = pd.read_pickle(
        f'{ds_exp_fn}_run-{this_block:02d}.pkl')
    tw_exp_bw[this_block] = pd.read_pickle(
        f'{tw_exp_fn}_run-{this_block:02d}.pkl')
    descr_stats_all_subs_bw_exp[this_block] = pd.read_pickle(
        f'{ds_exp_fn}_run-{this_block:02d}.pkl')
    grp_lvl_stats_bw_exp_both_rows = pd.read_pickle(
        f'{grp_stats_exp_fn}_run-{this_block:02d}.pkl')
    grp_lvl_stats_bw_exp[this_block] = grp_lvl_stats_bw_exp_both_rows[
        grp_lvl_stats_bw_exp_both_rows['sub_id'].isin(['experiment'])]

    # Simulation main data
    ev_sim_bw[this_block] = pd.read_pickle(
        f'{ev_sim_run_fn}{this_block:02d}_beh.pkl')
    grp_lvl_stats_bw_sim[this_block] = pd.read_pickle(
        f'{grp_stats_sim_fn}_run-{this_block:02d}.pkl')
    grp_lvl_stats_bw_sim_A[this_block] = grp_lvl_stats_bw_sim[this_block][
        grp_lvl_stats_bw_sim[this_block]['sub_id'].isin(
            ['Agent A1', 'Agent A2', 'Agent A3'])]
    grp_lvl_stats_bw_sim_C[this_block] = grp_lvl_stats_bw_sim[this_block][
        grp_lvl_stats_bw_sim[this_block]['sub_id'].isin(
            ['Agent C1', 'Agent C2', 'Agent C3'])]

# ----------------------------------------------------------
#       Prepare figure
# ----------------------------------------------------------
# Create general figure components
sub_label_beh = [s_dir[(s_dir.find('sub-') + 4):]
                 for s_dir in glob.glob(exp_data_dir + '/sub-*')]
sub_label_sim = [s_dir[(s_dir.find('sub-') + 4):]
                 for s_dir in glob.glob(sim_data_dir + '/sub-*')]
sub_label_beh.sort()
sub_label_sim.sort()

# Extract task configuration-specific beh_model components
n_blocks = np.max(exp_ev_all_subs_df['block'])
n_rounds = np.max(exp_ev_all_subs_df['round'])
n_trials = np.max(exp_ev_all_subs_df['trial']) - 1
n_tr_max = int(n_blocks * n_rounds)
n_tr_b = n_rounds

# Initialize figure
plotter = VeryPlotter(paths=dir_mgr.paths)
plt = plotter.get_pyplot_object()
col_exp = plotter.get_exp_group_colors()
col_A, col_C = plotter.get_agent_colors()

ax = {}
fig = plt.figure(figsize=(14, 4))
gs = gridspec.GridSpec(2, 1)
bar_width = 0.6
half_bar_width = bar_width / 3
s = 14
agent_colors = col_A + col_C  # Specify agent colors
linewidth = 1.5
ms=8
tick_and_axes_label_size=23
title_f=26
lettersize=26

# -----------------------------------------------------------------
#       Blockwise plots Experimental and main Simulation data
# -----------------------------------------------------------------

# ------Trial-by-trial/round-wise average choice rates------------------
block = 1
ax[0] = plt.subplot()
#ax[0] = plt.subplot(gs[0,0])
#ax[1] = plt.subplot(gs[1,0])

# Experimental group trial-by-trial choices
this_ax = ax[0]
x = tw_exp_bw[block].trial.values
y = tw_exp_bw[block].mean_drill.values
this_ax.scatter(x, y, alpha=0.2, s=50, color=col_exp[1], clip_on=False,
                #label="Experimental (trial-wise)"
                )

# Experimental group round-by-round choices
tw_exp_run_grpby_round = tw_exp_bw[block].groupby('round')
x = [((round_ * 12) - 5.5)
     for round_, tw_exp_run_thisround in tw_exp_run_grpby_round]
y = [np.mean(tw_exp_run_thisround['mean_drill'])
     for round_, tw_exp_run_thisround in tw_exp_run_grpby_round]
e = [np.std(tw_exp_run_thisround['mean_drill'])
     for round_, tw_exp_run_thisround in tw_exp_run_grpby_round]
this_ax.errorbar(
    x, y, alpha=0.7, markersize=8, color=col_exp[1], fmt='o',
    linestyle='-', linewidth=linewidth,
    label="Participant data",
    clip_on=False, yerr=e)

# Bayesian agents
i = 0
for agent, ev_thisagent_df in ev_sim_bw[block].groupby('sub_id'):
    if 'C' in agent:  # Scip control agents
        continue
    ev_thisagent_gbround = ev_thisagent_df.groupby('round')
    vlines = [(round_ * 12 - 11) for round_, ev_df in ev_thisagent_gbround]
    x = [((round_ * 12) - 5.5) for round_, ev_df in ev_thisagent_gbround]
    y = [np.mean(ev_df['action_type_num'])
         for round_, ev_df in ev_thisagent_gbround]
    e = [np.std(ev_df['action_type_num'])
         for round_, ev_df in ev_thisagent_gbround]
    this_ax.errorbar(
        x, y, alpha=0.7, markersize=ms, color=agent_colors[i], fmt='o',
        linestyle='-', linewidth=linewidth, clip_on=False,
        label=f"{agent} Simulation")
    i += 1

# ------Configure axis------
# Add vertical lines
this_ax.vlines(vlines, colors=[0.75, 0.75 , 0.75], linewidth=1.2, ymin=0,
               ymax=1, linestyles="--")
vlines.append(120)  # Add last boundary, to have 12 xticklabels
plotter.config_axes(
    this_ax,
    #title=r"\textbf{Experimental task configuration}", title_font=title_f,
    x_lim=[0, 120], x_label='Trial', xticks=vlines,
    xticklabels=np.around((np.linspace(1, 120, 11))).astype(int),
    y_label="\% Informative actions", y_lim=[0, 1],
    yticks=np.linspace(0, 1.0, 6),
    # ytickslabels=np.around(np.linspace(0, 1.0, 6), 2))
    ytickslabels=[0, 20, 40, 60, 80, 100], ticksize=tick_and_axes_label_size, axix_label_size=tick_and_axes_label_size)

this_ax.spines['bottom'].set_color('grey')
this_ax.spines['left'].set_color('grey')

# ax[0].text(0, 1.1, r'\textbf{A}', transform=this_ax.transAxes, size=lettersize,
#              weight='bold')


# Add letter, title and legend
this_ax.legend(
    bbox_to_anchor=(1.06, 0.9),
    loc='upper right',
    borderaxespad=0,
    fontsize=22)


# Trial-wise from sim 100
#
# # ------Trial-by-trial/round-wise average choice rates------------------
# i = 0
# this_ax = ax[1]
# for agent, agent_tw_df in tw_sim_100_aw.items():
#     ev_thisagent_gbround = agent_tw_df.groupby('round')
#     vlines = [(round_ * 12 - 11) for round_, ev_df in ev_thisagent_gbround]
#     x = [((round_ * 12) - 5.5) for round_, ev_df in ev_thisagent_gbround]
#     y = [np.nanmean(ev_df['p_drills'])
#          for round_, ev_df in ev_thisagent_gbround]
#     e = [np.nanstd(ev_df['p_drills'])
#          for round_, ev_df in ev_thisagent_gbround]
#     this_ax.errorbar(
#         x, y, alpha=0.7, markersize=ms, color=agent_colors[i],
#         fmt='o', linestyle='-', linewidth=linewidth, clip_on=False,
#         label=f"{agent} round average", yerr=e)
#     this_ax.vlines(vlines, colors=[.9, .9, .9], linewidth=.4, ymin=0, ymax=1)
#     i += 1
#
# # ------Configure axis------
# # Add vertical lines
# this_ax.vlines(vlines, colors=[0.75, 0.75 , 0.75], linewidth=1.2, ymin=0,
#                ymax=1, linestyles="--")
# vlines.append(120)  # Add last boundary, to have 12 xticklabels
# very_plotter.config_axes(
#     this_ax,
#     title=r"\textbf{Simulation of 100 task configurations}",
#     title_font=title_f,
#     x_lim=[0, 120], x_label='Trial', xticks=vlines,
#     xticklabels=np.around((np.linspace(1, 120, 11))).astype(int),
#     y_label="\% Informative actions", y_lim=[0, 1],
#     yticks=np.linspace(0, 1.0, 6),
#     # ytickslabels=np.around(np.linspace(0, 1.0, 6), 2))
#     ytickslabels=[0, 20, 40, 60, 80, 100], ticksize=tick_and_axes_label_size, axix_label_size=tick_and_axes_label_size)
#
# ax[1].text(0, 1.1, r'\textbf{B}', transform=this_ax.transAxes, size=lettersize,
#              weight='bold')

# Print subject level descriptive figure
fig.tight_layout()
fig.savefig(fig_fn, dpi=200, format='pdf')
