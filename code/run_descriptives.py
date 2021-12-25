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
#exp_label = 'main_sim_100'
dim = 5
n_block_this_label = {'main': 3, 'main_sim_100': 100}
n_blocks = n_block_this_label[exp_label]

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
part_fn = os.path.join(input_data_dir, 'participants.tsv')
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

events_all_subs_bw = {(run + 1): pd.DataFrame() for run in range(n_blocks)}
descr_stats_all_subs_bw = {(run + 1): pd.DataFrame() for run in range(n_blocks)}
grp_lvl_stats_bw = {(run + 1): pd.DataFrame() for run in range(n_blocks)}
trialwise_bw = {(run + 1): pd.DataFrame() for run in range(n_blocks)}

# Initialize or load dataframe for events_all_subs
if os.path.exists(f'{events_all_subs_fn}.pkl'):
    # Load events all subs
    events_all_subs_df = pd.read_pickle(f'{events_all_subs_fn}.pkl')
    # Load events block (run) wise all subs
    for block_, block_df in events_all_subs_bw.items():
        events_all_subs_bw[block_] = pd.read_pickle(os.path.join(out_proc_data_dir, f'sub-all_task-th_run-{block_:02d}_beh.pkl'))
else:
    events_all_subs_df = pd.DataFrame()
    edited_events_all_subs = True

if os.path.exists(f'{descr_stats_fn}.pkl'):
    descr_stats_all_subs_df = pd.read_pickle(f'{descr_stats_fn}.pkl')
    for block_, block_df in descr_stats_all_subs_bw.items():
        descr_stats_all_subs_bw[block_] = pd.read_pickle(f'{descr_stats_fn}_run-{block_:02d}.pkl')
else:
    descr_stats_all_subs_df = pd.DataFrame()  # (one row per subject)
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

# Create file list for all subjects
ev_file_list = glob.glob(input_data_dir + '/*/*/sub-*_task-th_beh.tsv')
ev_file_list.sort()

# Initialize id list
sub_id_list = []
# ------Loop through subjects to create events and descr stats----------------
for index, events_fn in enumerate(ev_file_list):

    # Get subject ID
    sub_id = events_fn[(events_fn.find('beh/sub-') + 8):events_fn.find('_task-th')]
    sub_id_list.append(sub_id)

    # -------Process data------------------
    # Check if subject in descr_stats_all_subs_df
    if events_all_subs_df.empty or (sub_id not in events_all_subs_df.sub_id.values) or \
            descr_stats_all_subs_df.empty or (sub_id not in descr_stats_all_subs_df.sub_id.values):

        # Check if processed data existent
        proc_data_fn = os.path.join(out_proc_data_dir, f'sub-{sub_id}_task-th_run-all_beh')
        if os.path.exists(f'{proc_data_fn}.pkl'):
            print(f'unpacking sub-{sub_id} proc_events.pkl')
            events_this_sub_df = pd.read_pickle(f'{proc_data_fn}.pkl')

        # Check separately, if processed data blockwise existent
        events_bw_run1_this_sub_fn = f'{out_proc_data_dir}/sub-{sub_id}_task-th_run-01_beh.pkl'
        if os.path.exists(events_bw_run1_this_sub_fn):
            events_block_this_sub = {}
            # Unpack each run's (block's) event datafram and write to dict
            for run_number in range(n_blocks):
                block_ = run_number + 1
                events_block_this_sub[block_] = pd.read_pickle(f'{out_proc_data_dir}/sub-{sub_id}_task-th_run-{block_:02d}_beh.pkl')

        else:
            print(f'processing events data for sub-{sub_id}')
            # Get and prepare data from this subject's events and append to events_df of all subs
            data_this_sub = Data(dataset, sub_id, events_fn, dim)  # Instantiate data class object
            data_this_sub.prep_data()  # --> Adds object attribute: data.event_data
            events_this_sub_df = data_this_sub.events_df
            events_block_this_sub = data_this_sub.events_block

            # ------Save processed data--------
            with open(f'{proc_data_fn}.tsv', 'w') as tsv_file:
                tsv_file.write(events_this_sub_df.to_csv(sep='\t', na_rep=np.NaN, index=False))
            events_this_sub_df.to_pickle(f'{proc_data_fn}.pkl')

        # Add this subs events to events_all_subs_df
        events_all_subs_df = events_all_subs_df.append(events_this_sub_df, ignore_index=True)
        for run_number in range(n_blocks):
            block_ = run_number + 1
            events_all_subs_bw[block_] = events_all_subs_bw[block_].append(events_block_this_sub[block_], ignore_index=True)

        # TODO: why is descr_stats_all_subs_bw sometimes empty??
        # Perform descriptive statistics for this subject and append to descr stats df of all subs
        descr_stats_this_sub = DescrStats(events_this_sub_df, dataset, subject=sub_id)
        descr_stats_this_sub_df = descr_stats_this_sub.perform_descr_stats()  # (One row for this subject)
        descr_stats_all_subs_df = descr_stats_all_subs_df.append(descr_stats_this_sub_df, ignore_index=True)

        for block_, block_df in events_block_this_sub.items():
            descr_stats_this_block = DescrStats(block_df, dataset, subject=sub_id)
            descr_stats_this_block_df = descr_stats_this_block.perform_descr_stats()
            descr_stats_all_subs_bw[block_] = descr_stats_all_subs_bw[block_].append(descr_stats_this_block_df, ignore_index=True)

    else:
        print(f'Skipping processing for sub-{sub_id}, already in descr_stats_all_subs_df')

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
    descr_stats_whole_sample = DescrStats(events_all_subs_df, dataset, subject='whole_sample',
                                          part_fn=part_fn)

    if dataset == 'sim':
        descr_stats_whole_sample.agent = np.nan

    descr_stats_whole_sample_df = descr_stats_whole_sample.perform_descr_stats()
    grp_lvl_stats_df = grp_lvl_stats_df.append(descr_stats_whole_sample_df, ignore_index=True)
    if dataset == 'exp':
        for block_, block_df in events_all_subs_bw.items():
            descr_stats_whole_sample_bw = DescrStats(block_df, dataset, subject=f'whole_sample_b')
            grp_lvl_stats_bw[block_] = descr_stats_whole_sample_bw.perform_descr_stats()
            grp_stats_object = GroupStats(block_df, dataset, descr_stats_all_subs_bw[block_])
            grp_descr_stats_df = grp_stats_object.perform_group_descr_stats(group_by='block_type')
            grp_lvl_stats_bw[block_] = grp_lvl_stats_bw[block_].append(grp_descr_stats_df)

    edited_grp_lvl_stats = True

grp_stats = GroupStats(events_all_subs_df, dataset, descr_stats_all_subs_df)

if dataset == 'sim':
    agent_group_descr_stats_incomplete = False
    for agent in list(events_all_subs_df['agent'].unique()):
        if agent not in grp_lvl_stats_df.sub_id.values:
            agent_group_descr_stats_incomplete = True
            break
    if agent_group_descr_stats_incomplete:
        grp_lvl_stats_df = grp_lvl_stats_df.append(grp_stats.perform_group_descr_stats(group_by='agent'))

        for block_, block_df in events_all_subs_bw.items():
            descr_stats_whole_sample_bw = DescrStats(block_df, dataset, subject=f'whole_sample_b')
            grp_lvl_stats_bw[block_] = descr_stats_whole_sample_bw.perform_descr_stats()
            grp_stats_object = GroupStats(block_df, dataset, descr_stats_all_subs_bw[block_])
            grp_descr_stats_df = grp_stats_object.perform_group_descr_stats(group_by='agent')
            grp_lvl_stats_bw[block_] = grp_lvl_stats_bw[block_].append(grp_descr_stats_df)

        edited_grp_lvl_stats = True

if edited_t_wise_stats:  # Change check for 'output_incomplete' to output-specific check
    print('Computing trialwise stats')
    t_wise_stats_df = grp_stats.eval_t_wise_stats(groupby='trial_cont_overallb')

# TODO: not yet robust!
#if dataset == 'exp':
t_wise_bw = {}
for block_, block_df in events_all_subs_bw.items():
    grp_stats = GroupStats(block_df, dataset, descr_stats_all_subs_bw[block_])
    t_wise_bw[block_] = grp_stats.eval_t_wise_stats(groupby='trial_cont')


if dataset == 'sim':
    print('Computing trialwise stats for each agent ')
    agent_list = list(events_all_subs_df.agent.unique())
    t_wise_bw_agent = {}
    t_wise_stats_all_blocks = {}

    for agent in agent_list:
        # Initialize dictionary for blockwise trialswise counts
        t_wise_bw_agent[agent] = {}
        t_wise_stats_all_blocks[agent] = {}

        # ------Compute trialwise stats overall blocks--------
        # Create object to group events_df all subs by agent
        agent_group_by_object = events_all_subs_df.groupby('agent')

        # Iterate through agents
        for agent, agent_df in agent_group_by_object:
            grp_stats_thisagent_all_blocks = GroupStats(agent_df, dataset, descr_stats_all_subs_df)
            t_wise_stats_all_blocks[agent] = grp_stats_thisagent_all_blocks.eval_t_wise_stats(groupby='trial_cont_overallb')

    # Save data  # TODO: move downwards, once working
    for agent, t_wise_thisagent_df in t_wise_stats_all_blocks.items():
            t_wise_thisagent_df.to_pickle(f'{t_wise_stats_fn}_agent-{agent}.pkl')
            with open(f'{t_wise_stats_fn}_agent-{agent}.tsv', 'w') as f:
                f.write(t_wise_thisagent_df.to_csv(sep='\t', na_rep='n/a'))

    # ------Compute trialwise stats for each block--------
    for block_, block_df in events_all_subs_bw.items():

        # Create object to group this blocks events_df by agent
        agentgroupby_object = block_df.groupby('agent')

        # Iterate through agents
        for agent, agent_block_df in agentgroupby_object:
            grp_stats_thisagent = GroupStats(agent_block_df, dataset, descr_stats_all_subs_bw[block_])
            t_wise_bw_agent[agent][block_] = grp_stats_thisagent.eval_t_wise_stats(groupby='trial_cont')

    # Save data  # TODO: move downwards, once working
    for agent, t_wise_bw_thisagent in t_wise_bw_agent.items():
        for block_, block_df in t_wise_bw_thisagent.items():
            block_df.to_pickle(f'{t_wise_stats_fn}_agent-{agent}_run-{block_:02d}.pkl')
            with open(f'{t_wise_stats_fn}_agent-{agent}_run-{block_:02d}.tsv', 'w') as f:
                f.write(block_df.to_csv(sep='\t', na_rep='n/a'))

if edited_r_wise_stats:
    print('Computing roundwise stats')
    descr_stats_all_subs_df = descr_stats_all_subs_df[descr_stats_all_subs_df['sub_id'].isin(sub_id_list)]
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
    for block_, block_df in events_all_subs_bw.items():
        block_df.to_pickle(f'{out_proc_data_dir}/sub-all_task-th_run-{block_:02d}_beh.pkl')
        with open(f'{out_proc_data_dir}/sub-all_task-th_run-{block_:02d}_beh.tsv', 'w') as f:
            f.write(block_df.to_csv(sep='\t', na_rep='n/a'))
if edited_descr_stats:
    with open(f'{descr_stats_fn}.tsv', 'w') as tsv_file:
        tsv_file.write(descr_stats_all_subs_df.to_csv(sep='\t', na_rep='n/a'))
    descr_stats_all_subs_df.to_pickle(f'{descr_stats_fn}.pkl')
    # TODO: bw-events not robust!
    for block_, block_df in descr_stats_all_subs_bw.items():
        block_df.to_pickle(f'{descr_stats_fn}_run-{block_:02d}.pkl')
        with open(f'{descr_stats_fn}_run-{block_:02d}.tsv', 'w') as f:
            f.write(block_df.to_csv(sep='\t', na_rep='n/a'))
if edited_grp_lvl_stats:
    with open(f'{grp_lvl_stats_fn}.tsv', 'w') as tsv_file:
        tsv_file.write(grp_lvl_stats_df.to_csv(sep='\t', na_rep='n/a'))
    grp_lvl_stats_df.to_pickle(f'{grp_lvl_stats_fn}.pkl')
    for block_, block_df in grp_lvl_stats_bw.items():
        block_df.to_pickle(f'{grp_lvl_stats_fn}_run-{block_:02d}.pkl')
        with open(f'{grp_lvl_stats_fn}_run-{block_:02d}.tsv', 'w') as f:
            f.write(block_df.to_csv(sep='\t', na_rep='n/a'))

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
