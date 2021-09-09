import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import pandas as pd
import os
import glob
import string
# from utilities.abm_structure import AbmStructure
from utilities.data_class import Data
from utilities.data_analyses import DescrStats
from utilities.data_analyses import GroupStats
# from utilities.plot_colormap import plot_colormap
from utilities.very_plotter import get_fig_template


"""
This script runs and plots a set of descriptive statistics on the treasure hunt task

File creation
    tr_found.png  : Number of treasures found for each subject
    todo: update png that are created

"""
# Specify dataset and experiment name
dataset = 'sim'  # 'exp' or 'sim'
exp_label = 'main'
dim = 5

# Specify directories and create if not existent
working_dir = os.getcwd()
project_dir = os.sep.join(working_dir.split(os.sep)[:4])  # Should be Users/<{$USER}>/<{$PROJECTFOLDER}>
data_dir = os.path.join(project_dir, 'data')
results_dir = os.path.join(project_dir, 'results')
figures_dir = os.path.join(project_dir, 'figures')
input_data_dir = os.path.join(data_dir, 'rawdata', f'{dataset}', f'{exp_label}')
out_proc_data_dir = os.path.join(data_dir, 'processed_data', f'{dataset}', f'{exp_label}')
out_descr_stats_dir = os.path.join(results_dir, 'descr_stats', f'{dataset}', f'{exp_label}')
out_fig_dir = os.path.join(figures_dir, f'{dataset}', f'{exp_label}')
if not os.path.exists(out_proc_data_dir):
    os.makedirs(out_proc_data_dir)
if not os.path.exists(out_descr_stats_dir):
    os.makedirs(out_descr_stats_dir)
if not os.path.exists(out_fig_dir):
    os.makedirs(out_fig_dir)

# Define file names
events_all_subs_fn = os.path.join(out_proc_data_dir, f'sub-all_task-th_run-all_beh')
descr_stats_fn = os.path.join(out_descr_stats_dir, 'descr_stats')
grp_lvl_stats_fn = os.path.join(out_descr_stats_dir, 'grp_lvl_stats')
t_wise_stats_fn = os.path.join(out_descr_stats_dir, 't_wise_stats')
r_wise_stats_fn = os.path.join(out_descr_stats_dir, 'r_wise_stats')

# Initialize or Load dataframes for event and descr stats

edited_events_all_subs = False
edited_descr_stats = False
edited_grp_lvl_stats = False
edited_t_wise_stats = False
edited_r_wise_stats = False

# Initialize dataframe for events_all_subs
if os.path.exists(f'{events_all_subs_fn}.pkl'):
    events_all_subs_df = pd.read_pickle(f'{events_all_subs_fn}.pkl')
else:
    events_all_subs_df = pd.DataFrame()
    edited_events_all_subs = True

# TODO: blockwise event files NOT robust!
events_bw = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
descr_stats_bw = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
trialwise_bw = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}

if os.path.exists(f'{descr_stats_fn}.pkl'):
    descr_stats_df = pd.read_pickle(f'{descr_stats_fn}.pkl')
else:
    descr_stats_df = pd.DataFrame()  # (one row per subject)
    edited_descr_stats = True

if os.path.exists(f'{grp_lvl_stats_fn}.pkl'):
    grp_lvl_stats_df = pd.read_pickle(f'{grp_lvl_stats_fn}.pkl')
else:
    grp_lvl_stats_df = pd.DataFrame()
    edited_grp_lvl_stats = True

if os.path.exists(f'{t_wise_stats_fn}.pkl'):
    t_wise_stats_df = pd.read_pickle(f'{t_wise_stats_fn}.pkl')
else:
    t_wise_stats_df = pd.DataFrame()
    edited_t_wise_stats = True

if os.path.exists(f'{r_wise_stats_fn}.pkl'):
    r_wise_stats_df = pd.read_pickle(f'{r_wise_stats_fn}.pkl')
else:
    r_wise_stats_df = pd.DataFrame()
    edited_r_wise_stats = True

# Initialize components for belief state colormaps
# viridis = cm.get_cmap('viridis', 256)

# Create file list for all subjects
ev_file_list = glob.glob(input_data_dir + '/*/*/sub-*_task-th_beh.tsv')
ev_file_list.sort()

# Initialize id list
sub_id_list = []
# ------Loop through subjects to create events and descr stats----------------
for index, events_fn in enumerate(ev_file_list):

    # Get subject ID
    sub_id = events_fn[(events_fn.find('sub-') + 4):events_fn.find('sub-') + 6]
    sub_id_list.append(sub_id)

    # -------Process data------------------
    # Check if subject in descr_stats_df
    if events_all_subs_df.empty or (sub_id not in events_all_subs_df.sub_id.values) or \
            descr_stats_df.empty or (sub_id not in descr_stats_df.sub_id.values):

        # Check if processed data existent
        proc_data_fn = os.path.join(out_proc_data_dir, f'sub-{sub_id}_task-th_run-all_beh')
        if os.path.exists(f'{proc_data_fn}.pkl'):
            events_this_sub_df = pd.read_pickle(f'{proc_data_fn}.pkl')
            print(f'unpacking sub-{sub_id} proc_events.pkl')
        else:
            print(f'processing events data for sub-{sub_id}')
            # Get and prepare data from this subject's events and append to events_df of all subs
            data_this_sub = Data(dataset, sub_id, events_fn, dim)  # Instantiate data class object
            data_this_sub.prep_data()  # --> Adds object attribute: data.event_data
            events_this_sub_df = data_this_sub.events_df

            # ------Save processed data--------
            with open(f'{proc_data_fn}.tsv', 'w') as tsv_file:
                tsv_file.write(events_this_sub_df.to_csv(sep='\t', na_rep=np.NaN, index=False))
            events_this_sub_df.to_pickle(f'{proc_data_fn}.pkl')
            for block_, block_df in data_this_sub.events_block.items():
                # Append this subs block events to block-specific events_all_subs
                events_bw[block_] = events_bw[block_].append(block_df, ignore_index=True)
                with open(f'{out_proc_data_dir}/sub-{sub_id}_task-th_run-{block_:02d}_beh.tsv', 'w') as f:
                    f.write(block_df.to_csv(sep='\t', na_rep=np.NaN, index=False))

        # Add this subs events to events_all_subs_df
        events_all_subs_df = events_all_subs_df.append(events_this_sub_df, ignore_index=True)

        # Perform descriptive statistics for this subject and append to descr stats df of all subs
        descr_stats_this_sub = DescrStats(events_this_sub_df, dataset, subject=sub_id)
        descr_stats_this_sub_df = descr_stats_this_sub.perform_descr_stats()  # (One row for this subject)
        descr_stats_df = descr_stats_df.append(descr_stats_this_sub_df, ignore_index=True)

        # TODO: not yet robust
        for block_, block_df in data_this_sub.events_block.items():
            descr_stats_this_block = DescrStats(block_df, dataset, subject=sub_id)
            descr_stats_this_block_df = descr_stats_this_block.perform_descr_stats()
            descr_stats_bw[block_] = descr_stats_bw[block_].append(descr_stats_this_block_df, ignore_index=True)

    else:
        print(f'Skipping processing for sub-{sub_id}, already in descr_stats_df')

    # # Heat maps for belief states, if simulation data
    # # ------------------------------------------------------------------
    # if dataset == 'sim':
    #
    #     groupby_block_rounds = data_summary_sub.groupby_block_rounds
    #
    #     for block_n_round, block_n_round_df in groupby_block_rounds:
    #
    #         b1_data_dic = {}
    #         # Iterate over trials
    #         for index, row in block_n_round_df.iterrows():
    #             if type(row['b_1_tr_loc_reshaped']) != float:
    #                 b1_data_dic[index] = row['b_1_tr_loc_reshaped']
    #         fig_row_cm_fn = os.path.join(out_figures_sub_dir, f'cmp_{block_n_round}.png')
    #         # pass dictionary with trial-wise data for one round or block ??
    #         fig_row_cm = plot_colormap(b1_data_dic)
    #         #fig_row_cm.write_image(file='static/images/staff_plot.png', format='.png')
    #         fig_row_cm.savefig(fig_row_cm_fn)

# ------Evaluate group-level stats-------------------------
if grp_lvl_stats_df.empty or ('whole_sample' not in grp_lvl_stats_df.sub_id.values):
    print("Computing 'whole sample' descr stats")
    descr_stats_whole_sample = DescrStats(events_all_subs_df, dataset, subject='whole_sample')
    if dataset == 'sim':
        descr_stats_whole_sample.agent = np.nan
    descr_stats_whole_sample_df = descr_stats_whole_sample.perform_descr_stats()
    grp_lvl_stats_df = grp_lvl_stats_df.append(descr_stats_whole_sample_df, ignore_index=True)
    edited_grp_lvl_stats = True

grp_stats = GroupStats(events_all_subs_df, dataset, descr_stats_df)

if dataset == 'sim':
    agent_group_descr_stats_incomplete = False
    for agent in list(events_all_subs_df['agent'].unique()):
        if agent not in grp_lvl_stats_df.sub_id.values:
            agent_group_descr_stats_incomplete = True
            break
    if agent_group_descr_stats_incomplete:
        grp_lvl_stats_df = grp_lvl_stats_df.append(grp_stats.perform_group_descr_stats(group_by='agent'))
        #grp_lvl_stats_df = descr_stats_df.append(descr_stats_agents_df, ignore_index=True)
        edited_grp_lvl_stats = True

# if dataset == 'sim' and edited_grp_lvl_stats:
#     print('Computing group-level stats')
#     agent_list = list(events_all_subs_df.agent.unique())
#     descr_stats_all_subs_df = descr_stats_df[descr_stats_df['sub_id'].isin(sub_id_list)]
#     grp_stats.descr_df = descr_stats_all_subs_df
#     grp_lvl_stats_df = grp_lvl_stats_df.append(grp_stats.perform_grp_lvl_stats(group_by='agent'), ignore_index=True)

if edited_t_wise_stats:  # Change check for 'output_incomplete' to output-specific check
    print('Computing trialwise stats')
    t_wise_stats_df = grp_stats.eval_t_wise_stats(groupby='trial_contin')

# TODO: not yet robust!
t_wise_bw = {}
for block_, block_df in events_bw.items():
    grp_stats = GroupStats(block_df, dataset, descr_stats_bw[block_])
    t_wise_bw[block_] = grp_stats.eval_t_wise_stats(groupby='trial_cont')

if edited_r_wise_stats:
    print('Computing roundwise stats')
    descr_stats_all_subs_df = descr_stats_df[descr_stats_df['sub_id'].isin(sub_id_list)]
    grp_stats.descr_df = descr_stats_all_subs_df
    r_wise_stats_df = grp_stats.eval_r_wise_stats()

# ---------------------------------------------------------------------------------------------
# --------------Save files-------------------------
# ---------------------------------------------------------------------------------------------
if edited_events_all_subs:
    with open(f'{events_all_subs_fn}.tsv', 'w') as tsv_file:
        tsv_file.write(events_all_subs_df.to_csv(sep='\t', na_rep='n/a'))
    events_all_subs_df.to_pickle(f'{events_all_subs_fn}.pkl')
    # TODO: bw-events not robust!
    for block_, block_df in events_bw.items():
        block_df.to_pickle(f'{out_proc_data_dir}/sub-all_task-th_run-{block_:02d}_beh.pkl')
        with open(f'{out_proc_data_dir}/sub-all_task-th_run-{block_:02d}_beh.tsv', 'w') as f:
            f.write(block_df.to_csv(sep='\t', na_rep='n/a'))
if edited_descr_stats:
    with open(f'{descr_stats_fn}.tsv', 'w') as tsv_file:
        tsv_file.write(descr_stats_df.to_csv(sep='\t', na_rep='n/a'))
    descr_stats_df.to_pickle(f'{descr_stats_fn}.pkl')
    # TODO: bw-events not robust!
    for block_, block_df in descr_stats_bw.items():
        block_df.to_pickle(f'{descr_stats_fn}_run-{block_:02d}.pkl')
        with open(f'{descr_stats_fn}_run-{block_:02d}.tsv', 'w') as f:
            f.write(block_df.to_csv(sep='\t', na_rep='n/a'))
if edited_grp_lvl_stats:
    with open(f'{grp_lvl_stats_fn}.tsv', 'w') as tsv_file:
        tsv_file.write(grp_lvl_stats_df.to_csv(sep='\t', na_rep='n/a'))
    grp_lvl_stats_df.to_pickle(f'{grp_lvl_stats_fn}.pkl')
if edited_t_wise_stats:
    with open(f'{t_wise_stats_fn}.tsv', 'w') as tsv_file:
        tsv_file.write(t_wise_stats_df.to_csv(sep='\t', na_rep='n/a'))
    t_wise_stats_df.to_pickle(f'{t_wise_stats_fn}.pkl')
    # TODO: bw-events not robust!
    for block_, block_df in t_wise_bw.items():
        block_df.to_pickle(f'{t_wise_stats_fn}_run-{block_:02d}.pkl')
        with open(f'{t_wise_stats_fn}_run-{block_:02d}.tsv', 'w') as f:
            f.write(block_df.to_csv(sep='\t', na_rep='n/a'))
if edited_r_wise_stats:
    with open(f'{r_wise_stats_fn}.tsv', 'w') as tsv_file:
        tsv_file.write(r_wise_stats_df.to_csv(sep='\t', na_rep='n/a'))
    r_wise_stats_df.to_pickle(f'{r_wise_stats_fn}.pkl')

# ---------------------------------------------------------------------------------------------
# ------Create subject level figures-----------------------
# ---------------------------------------------------------------------------------------------
# Extract descr stats of subjects only
descr_stats_all_subs_df = descr_stats_df[descr_stats_df['sub_id'].isin(sub_id_list)]

# Create general figure components:
subject_label = list(descr_stats_all_subs_df.sub_id.unique())

# figure initialization
fig_1_fn = os.path.join(out_fig_dir, 'sub_level_stats.pdf')
plt, greens, blues, yellows = get_fig_template(plt)
frm = mticker.ScalarFormatter(useMathText=True)
fig = plt.figure(figsize=(15, 13))
gs = gridspec.GridSpec(6, 2)
ax = {}

# Continuous color space
cm = plt.get_cmap('Blues')  # color map of interest
cm_subsection = np.linspace(.2, .8, 6)  # color map sampling indices
colors = [cm(x) for x in cm_subsection]  # colors of interest list

# # Plot subject level treasures discovery
# unveils = descr_stats_all_subs_df['n_tr'].values
# ax[0] = plt.subplot(gs[0, :])
# ax[0].bar(subject_label, unveils)
# # ax['tr_disc'].set_title('Treasure discovery / solved tasks', fontsize=14)
# # ax['tr_disc'].set_xlabel('Participants', fontsize=12)
# ax[0].grid(True, axis='y', linewidth=.5, color=[.9, .9, .9])
# ax[0].set_ylabel('Number of treasures', fontsize=12)
# ax[0].set_ylim(0, 20)
# ax[0].set_yticks([5, 10, 15, 20])

# # Plot subject level drill-versus-step (stacked bar plot)
# drills = descr_stats_all_subs_df['n_drills'].values
# steps = descr_stats_all_subs_df['n_steps'].values
# ax[1] = plt.subplot(gs[1, :])
# ax[1].bar(subject_label, drills, label='drill')
# # ax['drill_vs_step'].set_title('Action choices', fontsize=14)
# ax[1].bar(subject_label, steps, bottom=drills, label='steps')
# # ax['drill_vs_step'].set_xlabel('Participants', fontsize=12)
# ax[1].grid(True, axis='y', linewidth=.5, color=[.9, .9, .9])
# ax[1].set_ylabel('Number of actions', fontsize=12)
# ax[1].set_ylim(0, 450)
# ax[1].set_yticks([50, 150, 250, 350, 450])
# ax[1].legend()

# # Plot subject level %drills
# drills = descr_stats_all_subs_df['p_drills'].values
# ax[2] = plt.subplot(gs[2, :])
# ax[2].bar(subject_label, drills, label='drill')
# # ax['drill_vs_step'].set_title('Action choices', fontsize=14)
# # ax['drill_vs_step'].set_xlabel('Participants', fontsize=12)
# ax[2].grid(True, axis='y', linewidth=.5, color=[.9, .9, .9])
# ax[2].set_ylabel('\% Informative actions', fontsize=12)
# ax[2].set_ylim(0, 1)
# ax[2].set_yticks([.25, .5, .75, 1])

# Plot subject level percentage of action in which drilling led to unveiling ("Drilling success")
# unveils = descr_stats_all_subs_df['p_unv_if_drill'].replace(np.nan, 0).values
# ax[3] = plt.subplot(gs[3, :])
# ax[3].bar(subject_label, unveils)
# # ax['drill_success'].set_title('% Successful drills')
# # ax['drill_success'].set_xlabel('Participants', fontsize=12)
# ax[3].grid(True, axis='y', linewidth=.5, color=[.9, .9, .9])
# ax[3].set_ylabel('\% Successful drills', fontsize=12)
# ax[3].set_ylim([0, 1])
# ax[3].set_yticks([.25, .5, .75, 1])

# Plot subject level percentage of treasures discovered on visible hides vs. other
# were_hides = descr_stats_all_subs_df['p_visible_hide_giv_1.0_tr_disc']
# ax[4] = plt.subplot(gs[4, :])
#
# ax[4].bar(subject_label, were_hides)
# ax[4].set_ylim([0, 1])
# ax[4].set_ylabel('P(visible hide|treasure discovered)')
# # ax['found_on_hide'].set_title('% treasures found on visible hiding spots')
# ax[4].set_ylim([0, 1])
# ax[4].set_yticks([.25, .5, .75, 1])
# ax[4].set_ylabel('\% Found on visible hides', fontsize=12)
# ax[4].grid(True, axis='y', linewidth=.5, color=[.9, .9, .9])
# ax[4].set_xlabel('Participants', fontsize=12)

# Plot treasure discovery as a function of number of unveiled hiding spots
# ax[5] = plt.subplot(gs[5, 0])
# x = []
# y = []
#
# column_names_p_giv_hides = [col for col in descr_stats_all_subs_df.columns if 'p_tr_giv_hides_' in col]
#
# for column in column_names_p_giv_hides:
#     for sub in descr_stats_all_subs_df.index:
#         x.append(column[column.find('hides_') + 6:])
#         y.append(descr_stats_all_subs_df.loc[sub, column])
#
# ax[5].scatter(x, y)
# ax[5].set_xlabel('Number of visible hiding spots')
# ax[5].set_ylabel('P(tr disc giv N hides)')

# n_hides_categories = [col for col in summary_stats_all_subs.columns if 'p_tr_giv_hides_' in col]
# p_tr_giv_hides_df = summary_stats_all_subs[n_hides_categories]
# labels_n_hides = []
# for number_hides in n_hides_categories:
#     labels_n_hides.append(number_hides[-1])
#
# for index, sub_id in enumerate(subject_label):
#     ax_p_tr_giv_n_hides[index].set_ylim([0, 0.5])
#     p_tr_giv_n_hides = p_tr_giv_hides_df.loc[f'{sub_id}'].values
#     ax_p_tr_giv_n_hides[index].bar(labels_n_hides, p_tr_giv_n_hides)
#     ax_p_tr_giv_n_hides[index].set_ylabel(f'{sub_id}')

# Add letter to sub_id-figures
for key, value in ax.items():
    value.text(-0.06, 1.1, string.ascii_uppercase[key],
               transform=value.transAxes,
               size=20, weight='bold')

# Print subject level descriptive figure
fig.tight_layout()
fig.savefig(fig_1_fn, dpi=300, format='pdf')

# ---------------------------------------------------------------------------------------------
# ------Create group level figures-------------
# ---------------------------------------------------------------------------------------------
if dataset == 'sim':
    subject_label = list(events_all_subs_df.sub_id.unique())

    descr_stats_all_agents_df = descr_stats_df[descr_stats_df['sub_id'].isin(subject_label)]

    # figure initialization
    fig_3_fn = os.path.join(out_fig_dir, 'descr_stats_agents.pdf')
    plt, blue, red, yellows = get_fig_template(plt)
    frm = mticker.ScalarFormatter(useMathText=True)
    fig = plt.figure(figsize=(15, 13))
    gs = gridspec.GridSpec(6, 2)
    ax = {}

    # Continuous color space
    cm = plt.get_cmap('Blues')  # color map of interest
    cm_subsection = np.linspace(.2, .8, 6)  # color map sampling indices
    colors = [cm(x) for x in cm_subsection]  # colors of interest list

    # # Plot group mean treasures discovery
    # x = np.arange(len(subject_label))
    # y = grp_lvl_stats_df.n_tr_mean.values
    # yerr = grp_lvl_stats_df.n_tr_std.values
    # ax[0] = plt.subplot(gs[0, 0])
    # ax[0].errorbar(x, y, yerr,
    #                marker='o', ls='', capsize=2, lw=1, ms=4)
    # ax[0].set_xlabel('Group')
    # ax[0].set_xticks(x)
    # ax[0].set_xticklabels(subject_label)
    # ax[0].set_ylabel('Mean number of treasure', fontsize=12)
    # ax[0].set_ylim(0, 20)
    # ax[0].set_yticks([5, 10, 15, 20])
    # ax[0].grid(True, axis='y', linewidth=.5, color=[.9, .9, .9])

    # # Plot subject level drill-versus-step (stacked bar plot)
    # drills = descr_stats_all_agents_df['n_drills'].values
    # steps = descr_stats_all_agents_df['n_steps'].values
    # ax[1] = plt.subplot(gs[1, :])
    # ax[1].bar(subject_label, drills, label='drill')
    # # ax['drill_vs_step'].set_title('Action choices', fontsize=14)
    # ax[1].bar(subject_label, steps, bottom=drills, label='steps')
    # # ax['drill_vs_step'].set_xlabel('Participants', fontsize=12)
    # ax[1].grid(True, axis='y', linewidth=.5, color=[.9, .9, .9])
    # ax[1].set_ylabel('Number of actions', fontsize=12)
    # ax[1].set_ylim(0, 450)
    # ax[1].set_yticks([50, 150, 250, 350, 450])
    # ax[1].legend()

    # # Plot subject level %drills
    # drills = descr_stats_all_agents_df['p_drills'].values
    # ax[2] = plt.subplot(gs[2, :])
    # ax[2].bar(subject_label, drills, label='drill')
    # # ax['drill_vs_step'].set_title('Action choices', fontsize=14)
    # # ax['drill_vs_step'].set_xlabel('Participants', fontsize=12)
    # ax[2].grid(True, axis='y', linewidth=.5, color=[.9, .9, .9])
    # ax[2].set_ylabel('\% Informative actions', fontsize=12)
    # ax[2].set_ylim(0, 1)
    # ax[2].set_yticks([.25, .5, .75, 1])

    # # Plot subject level percentage of action in which drilling led to unveiling ("Drilling success")
    # unveils = descr_stats_all_agents_df['p_unv_if_drill'].replace(np.nan, 0).values
    # ax[3] = plt.subplot(gs[3, :])
    # ax[3].bar(subject_label, unveils)
    # # ax['drill_success'].set_title('% Successful drills')
    # # ax['drill_success'].set_xlabel('Participants', fontsize=12)
    # ax[3].grid(True, axis='y', linewidth=.5, color=[.9, .9, .9])
    # ax[3].set_ylabel('\% Successful drills', fontsize=12)
    # ax[3].set_ylim([0, 1])
    # ax[3].set_yticks([.25, .5, .75, 1])

    # # Plot subject level percentage of treasures discovered on visible hides vs. other
    # were_hides = descr_stats_all_agents_df['p_visible_hide_giv_1.0_tr_disc']
    # ax[4] = plt.subplot(gs[4, :])
    #
    # ax[4].bar(subject_label, were_hides)
    # ax[4].set_ylim([0, 1])
    # ax[4].set_ylabel('P(visible hide|treasure discovered)')
    # # ax['found_on_hide'].set_title('% treasures found on visible hiding spots')
    # ax[4].set_ylim([0, 1])
    # ax[4].set_yticks([.25, .5, .75, 1])
    # ax[4].set_ylabel('\% Found on visible hides', fontsize=12)
    # ax[4].grid(True, axis='y', linewidth=.5, color=[.9, .9, .9])
    # ax[4].set_xlabel('Participants', fontsize=12)

    # Plot treasure discovery as a function of number of unveiled hiding spots
    ax[5] = plt.subplot(gs[5, 0])
    x = []
    y = []

    column_names_p_giv_hides = [col for col in descr_stats_all_agents_df.columns if 'p_tr_giv_hides_' in col]

    for column in column_names_p_giv_hides:
        for sub in descr_stats_all_agents_df.index:
            x.append(column[column.find('hides_') + 6:])
            y.append(descr_stats_all_agents_df.loc[sub, column])

    ax[5].scatter(x, y)
    ax[5].set_xlabel('Number of visible hiding spots')
    ax[5].set_ylabel('P(tr disc giv N hides)')

    # Add letter to sub_id-figures
    for key, value in ax.items():
        value.text(-0.06, 1.1, string.ascii_uppercase[key],
                   transform=value.transAxes,
                   size=20, weight='bold')

    # Print subject level descriptive figure
    fig.tight_layout()
    fig.savefig(fig_3_fn, dpi=300, format='pdf')


# figure initialization
fig_2_fn = os.path.join(out_fig_dir, 'group_descr_stats_beh.pdf')
plt, blue, red, yellow = get_fig_template(plt)
frm = mticker.ScalarFormatter(useMathText=True)
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(4, 2)
ax = {}

# Plot trial-by-trial choice rate
trial_label = t_wise_stats_df.index
steps = t_wise_stats_df['n_steps'].values
drills = t_wise_stats_df['n_drills'].values
ax[0] = plt.subplot(gs[0, :])
ax[0].bar(trial_label, drills, label='drill')
# ax['drill_vs_step'].set_title('Action choices', fontsize=14)
ax[0].bar(trial_label, steps, bottom=drills, label='steps')
# ax['drill_vs_step'].set_xlabel('Participants', fontsize=12)
ax[0].grid(True, axis='y', linewidth=.5, color=[.9, .9, .9])
ax[0].set_ylabel('Number of actions', fontsize=12)
# ax[0].set_ylim(0, 30)
# ax[0].set_yticks([50, 150, 250, 350, 450])
# ax[0].legend()

# Plot trial-by-trial choice rate
trial_label = t_wise_stats_df.index
steps = t_wise_stats_df['p_steps'].values
drills = t_wise_stats_df['p_drills'].values
ax[1] = plt.subplot(gs[1, :])
ax[1].bar(trial_label, drills, label='drill')
# ax['drill_vs_step'].set_title('Action choices', fontsize=14)
ax[1].bar(trial_label, steps, bottom=drills, label='steps')
ax[1].set_xlabel('Trial', fontsize=12)
ax[1].grid(True, axis='y', linewidth=.5, color=[.9, .9, .9])
ax[1].set_ylabel('\% Action choice', fontsize=12)
ax[1].set_ylim(0, 1)
ax[1].set_yticks([0, 0.25, 0.5, 0.75])
ax[1].legend()

# Plot number of drills over rounds
x = [int(col[col.find('n_drills_r') + 10:]) for col in descr_stats_all_subs_df.columns if 'n_drills_r' in col]
y = r_wise_stats_df.drills_mean.values
yerr = r_wise_stats_df.drills_sdt.values
ax[2] = plt.subplot(gs[2, 0])
ax[2].errorbar(x, y, yerr,
               marker='o', ls='', capsize=2, lw=1, ms=4)
# ax[2].set_ylim([0, max(y)+5])
ax[2].set_xlabel('Round')
ax[2].set_ylabel('Mean Number of drills')
ax[2].grid(True, linewidth=.5, color=[.9, .9, .9])

# Plot group percent of drills over rounds
roundwise_drills_col_names = [col for col in descr_stats_all_subs_df.columns if 'p_drills_r' in col]
roundnumbers = [int(col[col.find('p_drills_r') + 10:]) for col in descr_stats_all_subs_df.columns if
                'p_drills_r' in col]
group_drills_p_df = descr_stats_all_subs_df[roundwise_drills_col_names]
group_drills_p_array = group_drills_p_df.to_numpy().reshape(len(roundnumbers))
ax[3] = plt.subplot(gs[3, 0])
ax[3].bar(roundnumbers, group_drills_p_array)
ax[3].set_xlabel('Round')
ax[3].set_ylabel('Group \% of drills')
ax[3].set_ylim(0, 1)
ax[3].grid(True, linewidth=.5, color=[.9, .9, .9])

# Add letter to sub_id-figures
for key, value in ax.items():
    value.text(-0.06, 1.1, string.ascii_uppercase[key],
               transform=value.transAxes,
               size=20, weight='bold')

# Print group level descriptive figure
fig.tight_layout()
fig.savefig(fig_2_fn, dpi=300, format='pdf')

# scatter number of drills over rounds
x = []
y = []
# for sub in descr_stats_all_subs_df.index:
#     roundwise_drill_counts_column_names = [col for col in descr_stats_all_subs_df.columns if 'n_drills_r' in col]
#     roundnumbers = [int(col[-1]) for col in descr_stats_all_subs_df.columns if 'n_drills_r' in col]
#     x.extend(roundnumbers)
#     y.extend(descr_stats_all_subs_df.loc[sub, roundwise_drill_counts_column_names])
# fig_drill_r_scatter_fn = os.path.join(out_fig_dir, 'drills_roundwise.png')
# fig_drill_r_scatter, ax_drill_r_scatter = plt.subplots()
# ax_drill_r_scatter.scatter(x, y)
# ax_drill_r_scatter.set_ylim([0, max(y) + 5])
# ax_drill_r_scatter.set_xlabel('round')
# ax_drill_r_scatter.set_ylabel('number of drills')
# ax_drill_r_scatter.set_title('number of drills in each round')
# fig_drill_r_scatter.savefig(fig_drill_r_scatter_fn)
