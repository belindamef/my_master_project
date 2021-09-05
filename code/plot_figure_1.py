import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import wesanderson
import pandas as pd
import os
import glob
import string
# from utilities.abm_structure import AbmStructure
from utilities.data_class import Data
from utilities.data_analyses import DescrStats
from utilities.data_analyses import GroupStats
# from utilities.plot_colormap import plot_colormap
from utilities.abm_figure_template import abm_figures


"""This script plots figure 1"""

# Specify task configuration
exp_label = 'main'
dim = 5

# Specify directories and filenames
working_dir = os.getcwd()
project_dir = os.sep.join(working_dir.split(os.sep)[:4])  # Should be Users/<{$USER}>/<{$PROJECTFOLDER}>
data_dir = os.path.join(project_dir, 'data')
results_dir = os.path.join(project_dir, 'results')
figures_dir = os.path.join(project_dir, 'figures')

beh_data_dir = os.path.join(data_dir, 'beh', f'{exp_label}')
sim_data_dir = os.path.join(data_dir, 'sim', f'{exp_label}')
beh_results_dir = os.path.join(results_dir, 'beh', f'{exp_label}')
sim_results_dir = os.path.join(results_dir, 'sim', f'{exp_label}')
beh_proc_data_dir = os.path.join(beh_results_dir, 'processed_data')
sim_proc_data_dir = os.path.join(sim_results_dir, 'processed_data')
beh_events_all_subs_fn = os.path.join(beh_results_dir, 'events_all_subs')
sim_events_all_subs_fn = os.path.join(sim_results_dir, 'events_all_subs')
beh_descr_stats_fn = os.path.join(beh_results_dir, 'descr_stats')
sim_descr_stats_fn = os.path.join(sim_results_dir, 'descr_stats')
beh_grp_stats_fn = os.path.join(beh_results_dir, 'grp_lvl_stats')
sim_grp_stats_fn = os.path.join(sim_results_dir, 'grp_lvl_stats')

# Load data
beh_ev_all_subs_df = pd.read_pickle(f'{beh_events_all_subs_fn}.pkl')
sim_ev_all_subs_df = pd.read_pickle(f'{sim_events_all_subs_fn}.pkl')
descr_stats_beh_df = pd.read_pickle(f'{beh_descr_stats_fn}.pkl')
descr_stats_sim_df = pd.read_pickle(f'{sim_descr_stats_fn}.pkl')
beh_grp_stats_df = pd.read_pickle(f'{beh_grp_stats_fn}.pkl')
sim_grp_stats_df = pd.read_pickle(f'{sim_grp_stats_fn}.pkl')

# Create general figure components
sub_label_beh = [s_dir[(s_dir.find('sub-') + 4):] for s_dir in glob.glob(beh_data_dir + '/sub-*')]
sub_label_sim = [s_dir[(s_dir.find('sub-') + 4):] for s_dir in glob.glob(sim_data_dir + '/sub-*')]
sub_label_beh.sort()
sub_label_sim.sort()

# Select single subject data only
ds_os_beh_df = descr_stats_beh_df[descr_stats_beh_df['subject'].isin(sub_label_beh)]  # descr stats only subs
ds_oa_sim_df = descr_stats_sim_df[descr_stats_sim_df['subject'].isin(sub_label_sim)]  # descr stats only agents
ds_A_sim_df = descr_stats_sim_df[descr_stats_sim_df['subject'].isin(['A1', 'A2', 'A3'])]
ds_C_sim_df = descr_stats_sim_df[descr_stats_sim_df['subject'].isin(['C1', 'C2', 'C3'])]
# Extract task config specific model components
n_blocks = np.max(beh_ev_all_subs_df['block'])
n_rounds = np.max(beh_ev_all_subs_df['round'])
n_trials = np.max(beh_ev_all_subs_df['trial']) - 1
n_tr_max = n_blocks * n_rounds

plot_set = {
    'treasure_sc': {
        'df': ds_os_beh_df,
        'cm_bar': wesanderson.color_palettes['Castello Cavalcanti'][0][0:2]
    }
}


# Initialize figure
fig_fn = os.path.join(figures_dir, 'figure_1.png')
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 6)
bar_width = 0.6
half_bar_width = bar_width / 2
ax = {}

# Plot subject level treasures discovery
#---------------------------------------------------------------------------------

# for behavioral data
ax[0] = plt.subplot(gs[0, 0:2])
ax[0].bar(0, np.mean(ds_os_beh_df['n_tr']), yerr=np.std(ds_os_beh_df['n_tr']),
          width=bar_width, capsize=10,
          color=wesanderson.color_palettes['Castello Cavalcanti'][0][3], zorder=0)
ax[0].bar([1, 1.5, 2], ds_A_sim_df['n_tr'].values,
          width=bar_width * (1/2),
          color=[wesanderson.color_palettes['Darjeeling Limited'][1][1],
                 wesanderson.color_palettes['The Life Aquatic with Steve Zissou'][0][0],
                 wesanderson.color_palettes['Darjeeling Limited'][1][3]])
ax[0].bar([2.5, 3, 3.5], ds_C_sim_df['n_tr'].values,
          width=bar_width * (1/2),
          color=[wesanderson.color_palettes['Darjeeling Limited'][1][2],
                 wesanderson.color_palettes['Isle of Dogs'][1][2],
                 wesanderson.color_palettes['Darjeeling Limited'][1][0]])
# Sort single data points to scatter
unique, counts = np.unique(ds_os_beh_df['n_tr'], return_counts=True)
y_counts_dic = dict(zip(unique, counts))
max_y_number = max(y_counts_dic.values())
y_x_pos = []
y_values = []
for y_value, y_count in y_counts_dic.items():
    if y_count == 1:
        positions = [0]
    else:
        positions = (np.linspace(0, (y_count * half_bar_width / max_y_number),
                                 y_count) - y_count * half_bar_width / max_y_number / 2)# + half_bar_width)
    y_x_pos.extend(positions)
    y_values.extend(y_count * [y_value])
ax[0].scatter(y_x_pos, y_values, s=6, color=wesanderson.color_palettes['Castello Cavalcanti'][0][1], zorder=1)
ax[0].grid(True, axis='y', linewidth=.5, color=[.9, .9, .9])
ax[0].set_ylabel('Total Number of Treasures', fontsize=16)
ax[0].set_ylim(0, 10)
ax[0].set_yticks(np.linspace(0, n_tr_max, int(n_tr_max/5) + 1))
ax[0].set_xticks([0, 1, 1.5, 2, 2.5, 3, 3.5])
ax[0].set_xticklabels(['Participants', 'A1', 'A2', 'A3', 'C1', 'C2', 'C3'], fontsize=16)
ax[0].set_title(f"Participant and Agent's task performance", size=18)

# Plot subject level %drills
#---------------------------------------------------------------------------------
# for behavioral data
ax[1] = plt.subplot(gs[1, 0:2])
ax[1].bar(0, np.mean(ds_os_beh_df['p_drills']), yerr=np.std(ds_os_beh_df['p_drills']),
          width=bar_width, capsize=10,
          color=wesanderson.color_palettes['Castello Cavalcanti'][0][3], zorder=0)
ax[1].bar([1, 1.5, 2], ds_A_sim_df['p_drills'].values,
          width=bar_width * (1/2),
          color=[wesanderson.color_palettes['Darjeeling Limited'][1][1],
                 wesanderson.color_palettes['The Life Aquatic with Steve Zissou'][0][0],
                 wesanderson.color_palettes['Darjeeling Limited'][1][3]])
ax[1].bar([2.5, 3, 3.5], ds_C_sim_df['p_drills'].values,
          width=bar_width * (1/2),
          color=[wesanderson.color_palettes['Darjeeling Limited'][1][2],
                 wesanderson.color_palettes['Isle of Dogs'][1][2],
                 wesanderson.color_palettes['Darjeeling Limited'][1][0]])
# Sort single data points to scatter
unique, counts = np.unique(np.around(ds_os_beh_df['p_drills'], 2), return_counts=True)
y_counts_dic = dict(zip(unique, counts))
max_y_number = max(y_counts_dic.values())
y_x_pos = []
y_values = []
for y_value, y_count in y_counts_dic.items():
    if y_count == 1:
        positions = [0]
    else:
        positions = (np.linspace(0, (y_count * half_bar_width / max_y_number),
                                 y_count) - y_count * half_bar_width / max_y_number / 2)# + half_bar_width)
    y_x_pos.extend(positions)
    y_values.extend(y_count * [y_value])

ax[1].scatter(y_x_pos, y_values, s=6, color=wesanderson.color_palettes['Castello Cavalcanti'][0][1], zorder=1)
ax[1].grid(True, axis='y', linewidth=.5, color=[.9, .9, .9])
ax[1].set_ylabel('Percentage drills', fontsize=16)
ax[1].set_ylim(0, 1)
ax[1].set_yticks(np.linspace(0, 1, 11))
ax[1].set_xticks([0, 1, 1.5, 2, 2.5, 3, 3.5])
ax[1].set_xticklabels(['Participants', 'A1', 'A2', 'A3', 'C1', 'C2', 'C3'], fontsize=16)
ax[1].set_title(f"Participant and Agent's action choices", size=18)



# Add letter to sub_id-figures
for key, value in ax.items():
    value.text(-0.1, 1.05, string.ascii_lowercase[key],
               transform=value.transAxes,
               size=20, weight='bold')

# Print subject level descriptive figure
fig.tight_layout()
fig.savefig(fig_fn, dpi=300, format='png')