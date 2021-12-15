import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import wesanderson
import pandas as pd
import os
import glob

from utilities.data_class import Data
from utilities.data_analyses import DescrStats
from utilities.data_analyses import GroupStats
# from utilities.plot_colormap import plot_colormap
import utilities.very_plotter as very_plotter


"""This script plots figure 1"""

# Specify task configuration
exp_label = 'main'
sim_label = 'main_bu'
dim = 5
n_blocks = 3

# Specify directories and filenames
working_dir = os.getcwd()
project_dir = os.sep.join(working_dir.split(os.sep)[:4])  # Should be Users/<{$USER}>/<{$PROJECTFOLDER}>
data_dir = os.path.join(project_dir, 'data')
results_dir = os.path.join(project_dir, 'results')
descr_stats_dir = os.path.join(results_dir, 'descr_stats')
figures_dir = os.path.join(project_dir, 'figures')

exp_data_dir = os.path.join(data_dir, 'rawdata', 'exp', f'{exp_label}')
sim_data_dir = os.path.join(data_dir, 'rawdata', 'sim', f'{sim_label}')
exp_proc_data_dir = os.path.join(data_dir, 'processed_data', 'exp', f'{exp_label}')
sim_proc_data_dir = os.path.join(data_dir, 'processed_data', 'sim', f'{exp_label}')

ev_exp_fn = os.path.join(exp_proc_data_dir, 'sub-all_task-th_run-all_beh')
ev_sim_fn = os.path.join(sim_proc_data_dir, 'sub-all_task-th_run-all_beh')
ev_exp_run_fn = os.path.join(exp_proc_data_dir, 'sub-all_task-th_run-')
ev_sim_run_fn = os.path.join(sim_proc_data_dir, 'sub-all_task-th_run-')

ds_exp_fn = os.path.join(descr_stats_dir, 'exp', f'{exp_label}', 'descr_stats')
ds_sim_fn = os.path.join(descr_stats_dir, 'sim', f'{exp_label}', 'descr_stats')
grp_stats_exp_fn = os.path.join(descr_stats_dir, 'exp', f'{exp_label}', 'grp_lvl_stats')
grp_stats_sim_fn = os.path.join(descr_stats_dir, 'sim', f'{exp_label}', 'grp_lvl_stats')
tw_exp_fn = os.path.join(descr_stats_dir, 'exp', f'{exp_label}', 't_wise_stats')
tw_sim_fn = os.path.join(descr_stats_dir, 'sim', f'{exp_label}', 't_wise_stats')

# Load data
exp_ev_all_subs_df = pd.read_pickle(f'{ev_exp_fn}.pkl')
sim_ev_all_subs_df = pd.read_pickle(f'{ev_sim_fn}.pkl')
descr_stats_exp_df = pd.read_pickle(f'{ds_exp_fn}.pkl')
descr_stats_sim_df = pd.read_pickle(f'{ds_sim_fn}.pkl')
exp_grp_stats_df = pd.read_pickle(f'{grp_stats_exp_fn}.pkl')
sim_grp_stats_df = pd.read_pickle(f'{grp_stats_sim_fn}.pkl')

ev_exp_run = {}
ev_sim_run = {}
ds_exp_run = {}
ds_sim_run = {}
ds_sim_A_run = {}
ds_sim_C_run = {}
tw_exp_run = {}
for block_ in range(n_blocks):
    this_block = block_ + 1
    ev_exp_run[this_block] = pd.read_pickle(f'{ev_exp_run_fn}{this_block:02d}_beh.pkl')
    ev_sim_run[this_block] = pd.read_pickle(f'{ev_sim_run_fn}{this_block:02d}_beh.pkl')
    ds_exp_run[this_block] = pd.read_pickle(f'{ds_exp_fn}_run-{this_block:02d}.pkl')
    ds_sim_run[this_block] = pd.read_pickle(f'{ds_sim_fn}_run-{this_block:02d}.pkl')
    ds_sim_A_run[this_block] = ds_sim_run[this_block][ds_sim_run[this_block]['sub_id'].isin(['A1', 'A2', 'A3'])]
    ds_sim_C_run[this_block] = ds_sim_run[this_block][ds_sim_run[this_block]['sub_id'].isin(['C1', 'C2', 'C3'])]
    tw_exp_run[this_block] = pd.read_pickle(f'{tw_exp_fn}_run-{this_block:02d}.pkl')

# Create general figure components
sub_label_beh = [s_dir[(s_dir.find('sub-') + 4):] for s_dir in glob.glob(exp_data_dir + '/sub-*')]
sub_label_sim = [s_dir[(s_dir.find('sub-') + 4):] for s_dir in glob.glob(sim_data_dir + '/sub-*')]
sub_label_beh.sort()
sub_label_sim.sort()

# Select single subject data only
ds_os_beh_df = descr_stats_exp_df[descr_stats_exp_df['sub_id'].isin(sub_label_beh)]  # descr stats only subs
ds_oa_sim_df = descr_stats_sim_df[descr_stats_sim_df['sub_id'].isin(sub_label_sim)]  # descr stats only agents
ds_A_sim_df = descr_stats_sim_df[descr_stats_sim_df['sub_id'].isin(['A1', 'A2', 'A3'])]
ds_C_sim_df = descr_stats_sim_df[descr_stats_sim_df['sub_id'].isin(['C1', 'C2', 'C3'])]
# Extract task config specific model components
n_blocks = np.max(exp_ev_all_subs_df['block'])
n_rounds = np.max(exp_ev_all_subs_df['round'])
n_trials = np.max(exp_ev_all_subs_df['trial']) - 1
n_tr_max = int(n_blocks * n_rounds)
n_tr_b = n_rounds

# Initialize figure
fig_fn = os.path.join(figures_dir, 'figure_1.png')
plt, col_exp, col_A, col_C = very_plotter.get_fig_template(plt)
ax = {}
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(4, 8)
bar_width = 0.6
half_bar_width = bar_width / 3

# ------Plot subject level treasures discovery--------------------------------
for block in range(n_blocks):
    ax[block] = plt.subplot(gs[block, 0:2])
    this_ax = ax[block]
    block += 1
    very_plotter.plot_bar(ax=this_ax, x=0,  height=np.mean(ds_exp_run[block]['n_tr'].div(n_tr_b)),
                          yerr=np.std(ds_exp_run[1]['n_tr'].div(n_tr_b)),
                          colors=col_exp[0])
    very_plotter.plot_bar(ax=this_ax, x=[1, 1.5, 2], height=ds_sim_A_run[block]['n_tr'].div(n_tr_b).values,
                          colors=col_A, bar_width=half_bar_width)
    very_plotter.plot_bar(ax=this_ax, x=[2.5, 3, 3.5], height=ds_sim_C_run[block]['n_tr'].div(n_tr_b).values,
                          colors=col_C, bar_width=half_bar_width)

    very_plotter.plot_bar_scatter(this_ax, ds_exp_run[block]['n_tr'].div(n_tr_b), color=col_exp[1], bar_width=bar_width)

    very_plotter.config_axes(this_ax, title=f"Task performance",
                             y_label="Number of Treasures", y_lim=[0, 1],
                             xticks=[0, 1, 1.5, 2, 2.5, 3, 3.5],
                             xticklabels=['Participants', 'A1', 'A2', 'A3', 'C1', 'C2', 'C3'],
                             yticks=np.linspace(0, 1.0, 6),
                             ytickslabels=np.around(np.linspace(0, 1.0, 6), 2))

# ------Plot trialwise tr_disc--------------------------------------------
s = 14
for block in range(n_blocks):
    block += 1
    ax[block] = plt.subplot(gs[block - 1, 2:8])
    this_ax = ax[block]
    tw_exp_df = ev_exp_run[block].groupby('trial_cont')
    x = tw_exp_run[block].trial.values
    y = tw_exp_run[block].p_drills.values
    this_ax.scatter(x, y, alpha=0.6, s=s, color=col_exp[1], label="Participant's \n average") #facecolors='none', edgecolors=col_exp[1])
    for i, agent in enumerate(['A1', 'A2', 'A3']):
        action_types = list(np.nan_to_num(ev_sim_run[block][ev_sim_run[block]['sub_id'] == agent].action_type.values))
        y_drills = [(1.2 + i * 0.2) if action == 'drill' else np.nan for action in action_types]
        y_steps = [(1.1 + i * 0.2) if action == 'step' else np.nan for action in action_types]
        #this_ax.scatter(x, y_steps, marker="v", s=s, edgecolors=col_A[i + 1], label=f'{agent} step', facecolors='black') #, facecolors='none', edgecolors=col_A[i])
        #this_ax.scatter(x, y_drills, marker="v", s=s, color=col_A[i + 1], label=f'{agent} drill')
        this_ax.scatter(x, y_steps, marker="o", s=s, edgecolors=col_A[i], label=f'{agent} step', facecolors='none') #, facecolors='none', edgecolors=col_A[i])
        this_ax.scatter(x, y_drills, marker="o", s=s, color=col_A[i], label=f'{agent} drill')
    for i, agent in enumerate(['C1', 'C2', 'C3']):
        #yellows_ = [col_C[0], col_C[2]]
        yellows_ = col_C
        action_types = list(np.nan_to_num(ev_sim_run[block][ev_sim_run[block]['sub_id'] == agent].action_type.values))
        y_drills = [(1.8 + i * 0.2) if action == 'drill' else np.nan for action in action_types]
        y_steps = [(1.7 + i * 0.2) if action == 'step' else np.nan for action in action_types]
        #this_ax.scatter(x, y_steps, marker="o", s=s, edgecolors=yellows_[i], label=f'{agent} step', facecolors='black')
        #this_ax.scatter(x, y_drills, marker="o", s=s, facecolors='none', edgecolors=yellows_[i], label=f'{agent} drill')
        this_ax.scatter(x, y_steps, marker="v", s=s, edgecolors=yellows_[i], label=f'{agent} step', facecolors='none')
        this_ax.scatter(x, y_drills, marker="v", s=s, color=yellows_[i], label=f'{agent} drill')

    very_plotter.config_axes(this_ax,
                             x_lim=[0, 140], x_label='Trial', xticks=(np.linspace(1, 120, 11)),
                             xticklabels=np.around((np.linspace(1, 120, 11))).astype(int),
                             y_label="\% Informative", y_lim=[0, 2],
                             yticks=np.linspace(0, 1.0, 6),
                             ytickslabels=np.around(np.linspace(0, 1.0, 6), 2))
ax[1].legend(loc='center right')
    # y_labels = [item.get_text() for item in this_ax.get_yticklabels()]
    # y_labels[-1] = ''
    # this_ax.set_yticklabels(y_labels)
very_plotter.config_axes(ax[1], title="Action choices")

# ------Plot subject level %drills--------------------------------------------
ax[4] = plt.subplot(gs[3, 0:2])
this_ax = ax[4]
very_plotter.plot_bar(ax=this_ax, x=0, height=np.mean(ds_os_beh_df['p_drills']), yerr=np.std(ds_os_beh_df['p_drills']),
                      colors=col_exp[0])
very_plotter.plot_bar(ax=this_ax, x=[1, 1.5, 2], height=ds_A_sim_df['p_drills'].values,
                      colors=col_A, bar_width=half_bar_width)
very_plotter.plot_bar(ax=this_ax, x=[2.5, 3, 3.5], height=ds_C_sim_df['p_drills'].values,
                      colors=col_C, bar_width=half_bar_width)

very_plotter.plot_bar_scatter(this_ax, np.around(ds_os_beh_df['p_drills'], 2), color=col_exp[1], bar_width=bar_width)
very_plotter.config_axes(this_ax, title=f"Average choice rates",
                         y_label='\% Informative actions', y_lim=[0, 1.4],
                         xticks=[0, 1, 1.5, 2, 2.5, 3, 3.5],
                         xticklabels=['Participants', 'A1', 'A2', 'A3', 'C1', 'C2', 'C3'])

# ------Plot subject level %drills--------------------------------------------
ax[4] = plt.subplot(gs[3, 0:2])
this_ax = ax[4]
very_plotter.plot_bar(ax=this_ax, x=0, height=np.mean(ds_os_beh_df['p_drills']), yerr=np.std(ds_os_beh_df['p_drills']),
                      colors=col_exp[0])
very_plotter.plot_bar(ax=this_ax, x=[1, 1.5, 2], height=ds_A_sim_df['p_drills'].values,
                      colors=col_A, bar_width=half_bar_width)
very_plotter.plot_bar(ax=this_ax, x=[2.5, 3, 3.5], height=ds_C_sim_df['p_drills'].values,
                      colors=col_C, bar_width=half_bar_width)

very_plotter.plot_bar_scatter(this_ax, np.around(ds_os_beh_df['p_drills'], 2), color=col_exp[1], bar_width=bar_width)
very_plotter.config_axes(this_ax, title=f"Average choice rates",
                         y_label='\% Informative actions', y_lim=[0, 1.4],
                         xticks=[0, 1, 1.5, 2, 2.5, 3, 3.5],
                         xticklabels=['Participants', 'A1', 'A2', 'A3', 'C1', 'C2', 'C3'])



# Add letter to sub_id-figures
very_plotter.add_letters(ax)

# Print subject level descriptive figure
fig.tight_layout()
fig.savefig(fig_fn, dpi=200, format='png')
#
#
# ax[0] = plt.subplot(gs[0, 0:2])
# this_ax = ax[0]
# very_plotter.plot_bar(ax=this_ax, x=0, height=np.mean(ds_os_beh_df['n_tr'].div(n_tr_max)),
#                       yerr=np.std(ds_os_beh_df['n_tr'].div(n_tr_max)),
#                       colors=greens[0])
# very_plotter.plot_bar(ax=this_ax, x=[1, 1.5, 2], height=ds_A_sim_df['n_tr'].div(n_tr_max).values,
#                       colors=blues, bar_width=half_bar_width)
# very_plotter.plot_bar(ax=this_ax, x=[2.5, 3, 3.5], height=ds_C_sim_df['n_tr'].div(n_tr_max).values,
#                       colors=yellows, bar_width=half_bar_width)
#
# very_plotter.plot_bar_scatter(this_ax, ds_os_beh_df['n_tr'].div(n_tr_max), color=greens[1], bar_width=bar_width)
#
# very_plotter.config_axes(this_ax, title=f"Participant and Agent's task performance",
#                          y_label="\% Treasures", y_lim=[0, 1],
#                          xticks=[0, 1, 1.5, 2, 2.5, 3, 3.5],
#                          xticklabels=['Participants', 'A1', 'A2', 'A3', 'C1', 'C2', 'C3'])
