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
exp_events_all_subs_fn = os.path.join(exp_proc_data_dir, 'sub-all_task-th_run-all_beh')
sim_events_all_subs_fn = os.path.join(sim_proc_data_dir, 'sub-all_task-th_run-all_beh')
exp_run_ev_all_subs_fn = os.path.join(exp_proc_data_dir, 'sub-all_task-th_run-')
sim_run_ev_all_subs_fn = os.path.join(sim_proc_data_dir, 'sub-all_task-th_run-')

exp_descr_stats_fn = os.path.join(descr_stats_dir, 'exp', f'{exp_label}', 'descr_stats')
sim_descr_stats_fn = os.path.join(descr_stats_dir, 'sim', f'{exp_label}', 'descr_stats')
exp_grp_stats_fn = os.path.join(descr_stats_dir, 'exp', f'{exp_label}', 'grp_lvl_stats')
sim_grp_stats_fn = os.path.join(descr_stats_dir, 'sim', f'{exp_label}', 'grp_lvl_stats')

# Load data
exp_ev_all_subs_df = pd.read_pickle(f'{exp_events_all_subs_fn}.pkl')
sim_ev_all_subs_df = pd.read_pickle(f'{sim_events_all_subs_fn}.pkl')
descr_stats_exp_df = pd.read_pickle(f'{exp_descr_stats_fn}.pkl')
descr_stats_sim_df = pd.read_pickle(f'{sim_descr_stats_fn}.pkl')
exp_grp_stats_df = pd.read_pickle(f'{exp_grp_stats_fn}.pkl')
sim_grp_stats_df = pd.read_pickle(f'{sim_grp_stats_fn}.pkl')
exp_run_ev = {}
sim_run_ev = {}
exp_ds_run = {}
sim_ds_run = {}
for block_ in range(n_blocks):
    this_block = block_ + 1
    exp_run_ev[this_block] = pd.read_pickle(f'{exp_run_ev_all_subs_fn}{this_block:02d}_beh.pkl')
    sim_run_ev[this_block] = pd.read_pickle(f'{sim_run_ev_all_subs_fn}{this_block:02d}_beh.pkl')
    exp_ds_run[this_block] = pd.read_pickle(f'{exp_descr_stats_fn}_run-{this_block:02d}.pkl')
    sim_ds_run[this_block] = pd.read_pickle(f'{sim_descr_stats_fn}_run-{this_block:02d}.pkl')

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

# Initialize figure
fig_fn = os.path.join(figures_dir, 'figure_1.png')
plt, greens, blues, yellows = very_plotter.get_fig_template(plt)
ax = {}
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(4, 7)
bar_width = 0.6
half_bar_width = bar_width / 2

# ------Plot subject level treasures discovery--------------------------------
ax[0] = plt.subplot(gs[0, 0:2])
this_ax = ax[0]
very_plotter.plot_bar(ax=this_ax, x=0, height=np.mean(exp_ds_run[1]['n_tr'].div(n_tr_max)),
                      yerr=np.std(ds_os_beh_df['n_tr'].div(n_tr_max)),
                      colors=greens[0])
very_plotter.plot_bar(ax=this_ax, x=[1, 1.5, 2], height=ds_A_sim_df['n_tr'].div(n_tr_max).values,
                      colors=blues, bar_width=half_bar_width)
very_plotter.plot_bar(ax=this_ax, x=[2.5, 3, 3.5], height=ds_C_sim_df['n_tr'].div(n_tr_max).values,
                      colors=yellows, bar_width=half_bar_width)

very_plotter.plot_bar_scatter(this_ax, ds_os_beh_df['n_tr'].div(n_tr_max), color=greens[1], bar_width=bar_width)

very_plotter.config_axes(this_ax, title=f"Participant and Agent's task performance",
                         y_label="\% Treasures", y_lim=[0, 1],
                         xticks=[0, 1, 1.5, 2, 2.5, 3, 3.5],
                         xticklabels=['Participants', 'A1', 'A2', 'A3', 'C1', 'C2', 'C3'])


# ------Plot trialwise tr_disc--------------------------------------------
ax[2] = plt.subplot(gs[0, 2:6])
#ax[2].scatter()

# ------Plot subject level %drills--------------------------------------------
ax[4] = plt.subplot(gs[3, 0:2])
this_ax = ax[4]
very_plotter.plot_bar(ax=this_ax, x=0, height=np.mean(ds_os_beh_df['p_drills']), yerr=np.std(ds_os_beh_df['p_drills']),
                      colors=greens[0])
very_plotter.plot_bar(ax=this_ax, x=[1, 1.5, 2], height=ds_A_sim_df['p_drills'].values,
                      colors=blues, bar_width=half_bar_width)
very_plotter.plot_bar(ax=this_ax, x=[2.5, 3, 3.5], height=ds_C_sim_df['p_drills'].values,
                      colors=yellows, bar_width=half_bar_width)

very_plotter.plot_bar_scatter(this_ax, np.around(ds_os_beh_df['p_drills'], 2), color=greens[1], bar_width=bar_width)
very_plotter.config_axes(this_ax, title=f"Participant and Agent's action choices",
                         y_label='\% Informative actions', y_lim=[0, 1],
                         xticks=[0, 1, 1.5, 2, 2.5, 3, 3.5],
                         xticklabels=['Participants', 'A1', 'A2', 'A3', 'C1', 'C2', 'C3'])


# Add letter to sub_id-figures
very_plotter.add_letters(ax)

# Print subject level descriptive figure
fig.tight_layout()
fig.savefig(fig_fn, dpi=100, format='png')
