"""This script runs and plots a set of descriptive statistics on the 
treasure hunt task 

File creation
    tr_found.png  : Number of treasures found for each subject
    todo: update png that are created
"""

import os
import glob
import numpy as np
import pandas as pd
from utilities.config import DirectoryManager
from utilities.data_class import Data
from utilities.data_analyses import DescrStats
from utilities.data_analyses import GroupStats


def main():

    # Specify directories and create if not existent
    dir_mgr = DirectoryManager()
    dir_mgr.define_raw_beh_data_out_path(data_type=EXP_OR_SIM,
                                         out_dir_label=EXP_LABEL,
                                         make_dir=False)
    dir_mgr.define_processed_data_path(data_type=EXP_OR_SIM,
                                       dir_label=EXP_LABEL,
                                       make_dir=True)
    dir_mgr.define_descr_stats_path(data_type=EXP_OR_SIM,
                                    dir_label=EXP_LABEL,
                                    make_dir=True)
    dir_mgr.define_stats_filenames()

    # Initialize or Load dataframes for event and descr stats
    edited_processed_events_all_subs = False
    edited_descr_stats = False
    edited_grp_lvl_stats = False
    edited_t_wise_stats = False
    edited_r_wise_stats = False

    events_all_subs_bw = {(run + 1): pd.DataFrame() for run in range(N_BLOCKS)}
    subj_level_stats_all_subs_bw = {
        (run + 1): pd.DataFrame() for run in range(N_BLOCKS)}
    grp_lvl_stats_bw = {(run + 1): pd.DataFrame() for run in range(N_BLOCKS)}

    # ---------------------------------------------------------------------
    # Initialize or load dataframe for events_all_subs for processed data
    if os.path.exists(f'{dir_mgr.paths.events_all_subs_fn}.pkl'):

        # Load events all subs
        events_all_subs_df = pd.read_pickle(
            f'{dir_mgr.paths.events_all_subs_fn}.pkl')
        # # Load events block (run) wise all subs
        # for block_, block_df in events_all_subs_bw.items():
        #     events_all_subs_bw[block_] = pd.read_pickle(os.path.join(
        #         dir_mgr.paths.this_analyses_proc_data_path, f'sub-all_task-th_run-{block_:02d}_beh.pkl'))
    else:
        events_all_subs_df = pd.DataFrame()
        edited_processed_events_all_subs = True

    # ---------------------------------------------------------------------
    # Initialize or load dataframe for subject level descr stats all subs
    if os.path.exists(f'{dir_mgr.paths.subj_lvl_descr_stats_fn}.pkl'):
        descr_stats_all_subs_df = pd.read_pickle(f'{dir_mgr.paths.subj_lvl_descr_stats_fn}.pkl')
        # for block_, block_df in subj_level_stats_all_subs_bw.items():
        #     subj_level_stats_all_subs_bw[block_] = pd.read_pickle(
        #         f'{dir_mgr.paths.subj_lvl_descr_stats_fn}_run-{block_:02d}.pkl')
    else:
        descr_stats_all_subs_df = pd.DataFrame()  # (one row per subject)
        edited_descr_stats = True

    # ---------------------------------------------------------------------
    # Initialize or load dataframe for group level descr stats all subs
    if os.path.exists(f'{dir_mgr.paths.grp_lvl_descr_stats_fn}.pkl'):
        grp_lvl_stats_df = pd.read_pickle(f'{dir_mgr.paths.grp_lvl_descr_stats_fn}.pkl')
    else:
        grp_lvl_stats_df = pd.DataFrame()
        edited_grp_lvl_stats = True

    # ---------------------------------------------------------------------
    # Initialize or load dataframe for trialwise stats all subs
    if os.path.exists(f'{dir_mgr.paths.t_wise_stats_fn}.pkl'):
        t_wise_stats_df = pd.read_pickle(f'{dir_mgr.paths.t_wise_stats_fn}.pkl')
    else:
        t_wise_stats_df = pd.DataFrame()
        edited_t_wise_stats = True

    # ---------------------------------------------------------------------
    # Initialize or load dataframe for roundwise stats all subs
    if os.path.exists(f'{dir_mgr.paths.r_wise_stats_fn}.pkl'):
        r_wise_stats_df = pd.read_pickle(f'{dir_mgr.paths.r_wise_stats_fn}.pkl')
    else:
        r_wise_stats_df = pd.DataFrame()
        edited_r_wise_stats = True

    # Create file list for all subjects
    ev_file_list = glob.glob(
        dir_mgr.paths.this_analyses_raw_data_path
        + '/*/*/sub-*_task-th_beh.tsv')
    ev_file_list.sort()

    # Initialize id list
    sub_id_list = []

    # ------Loop through subjects to create events and descr stats-------------

    for events_fn in ev_file_list:

        # Get subject ID
        sub_id = events_fn[(events_fn.find('beh/sub-') + 8):events_fn.find(
            '_task-th')]
        sub_id_list.append(sub_id)

        # -------Process data------------------
        # Check if subject in descr_stats_all_subs_df
        if events_all_subs_df.empty or (
                sub_id not in events_all_subs_df.sub_id.values) or \
                descr_stats_all_subs_df.empty or \
                (sub_id not in descr_stats_all_subs_df.sub_id.values):

            # Check if processed data existent, and load if existent
            proc_data_fn = os.path.join(
                dir_mgr.paths.this_analyses_proc_data_path, f'sub-{sub_id}_task-th_run-all_beh')
            if os.path.exists(f'{proc_data_fn}.pkl'):
                print(f'unpacking sub-{sub_id} proc_events.pkl')
                events_this_sub_df = pd.read_pickle(f'{proc_data_fn}.pkl')

            # # Check separately, if processed data block wise existent, and load
            # events_bw_run1_this_sub_fn = \
            #     f'{dir_mgr.paths.this_analyses_proc_data_path}/sub-{sub_id}_task-th_run-01_beh.pkl'
            # if os.path.exists(events_bw_run1_this_sub_fn):
            #     events_block_this_sub = {}
            #     # Unpack each run's (block's) event dataframe and write to dict
            #     for run_number in range(N_BLOCKS):
            #         block_ = run_number + 1
            #         events_block_this_sub[block_] = pd.read_pickle(
            #             f'{dir_mgr.paths.this_analyses_proc_data_dir}/'
            #             f'sub-{sub_id}_task-th_run-{block_:02d}_beh.pkl')

            else:
                print(f'processing events data for sub-{sub_id}')
                # Get and prepare data from this subject's events and
                # append to events_df of all subs
                data_this_sub = Data(  # Instantiate data class object
                    EXP_OR_SIM, sub_id, events_fn, DIM)
                data_this_sub.prep_data()
                # --> Adds object attribute: data.event_data
                events_this_sub_df = data_this_sub.events_df
                events_block_this_sub = data_this_sub.events_block

                # ------Save processed data--------
                with open(f'{proc_data_fn}.tsv', 'w') as tsv_file:
                    tsv_file.write(
                        events_this_sub_df.to_csv(
                            sep='\t', na_rep=np.NaN, index=False))
                events_this_sub_df.to_pickle(f'{proc_data_fn}.pkl')

            # Add this subs events to events_all_subs_df
            events_all_subs_df = pd.concat([events_all_subs_df,
                                            events_this_sub_df],
                                            ignore_index=True)
            # for run_number in range(N_BLOCKS):
            #     block_ = run_number + 1
            #     events_all_subs_bw[block_] = pd.concat(
            #         [events_all_subs_bw[block_],events_block_this_sub[block_]],
            #           ignore_index=True)

            # TODO: why is descr_stats_all_subs_bw sometimes empty??
            # Perform descriptive statistics for this subject and
            # append to descr stats df of all subs
            descr_stats_this_sub = DescrStats(
                events_this_sub_df, EXP_OR_SIM, subject=sub_id)
        
            # (One row for this subject)
            descr_stats_this_sub_df = descr_stats_this_sub.perform_descr_stats()

            descr_stats_all_subs_df = pd.concat([descr_stats_all_subs_df,
                                                 descr_stats_this_sub_df],
                                                ignore_index=True)

            # for block_, block_df in events_block_this_sub.items():
            #     descr_stats_this_block = DescrStats(
            #         block_df, EXP_OR_SIM, subject=sub_id)
            #     descr_stats_this_block_df = \
            #         descr_stats_this_block.perform_descr_stats()
            #     subj_level_stats_all_subs_bw[block_] = pd.concat([
            #         subj_level_stats_all_subs_bw[block_],
            #         descr_stats_this_block_df], ignore_index=True)
        else:
            print(
                f'Skipping processing for sub-{sub_id}, '
                f'already in descr_stats_all_subs_df')

    # ------Evaluate group-level stats-------------------------
    # if grp_lvl_stats_df.empty or (
    #         'whole_sample' not in grp_lvl_stats_df.sub_id.values):
    #     print("Computing 'whole sample' descr stats")
    #     descr_stats_whole_sample = DescrStats(
    #         events_all_subs_df, EXP_OR_SIM, subject='whole_sample',
    #         part_fn=dir_mgr.paths.part_fn)

    #     if EXP_OR_SIM == 'sim':
    #         descr_stats_whole_sample.agent = np.nan

    #     descr_stats_whole_sample_df = \
    #         descr_stats_whole_sample.perform_descr_stats()
    #     grp_lvl_stats_df = pd.concat([grp_lvl_stats_df,
    #                                 descr_stats_whole_sample_df],
    #                                 ignore_index=True)
    #     if EXP_OR_SIM == 'exp':
    #         grp_lvl_stats_whole_sample_object = GroupStats(
    #             events_all_subs_df, EXP_OR_SIM, descr_stats_all_subs_df)
    #         grp_lvl_stats_whole_sample_df = \
    #             grp_lvl_stats_whole_sample_object.perform_group_descr_stats(
    #                 group_by='block_type')
    #         grp_lvl_stats_df = pd.concat([grp_lvl_stats_df,
    #                                     grp_lvl_stats_whole_sample_df],
    #                                     ignore_index=True)
    #         for block_, block_df in events_all_subs_bw.items():
    #             descr_stats_whole_sample_bw = DescrStats(
    #                 block_df, EXP_OR_SIM, subject='whole_sample_b')
    #             grp_lvl_stats_bw[
    #                 block_] = descr_stats_whole_sample_bw.perform_descr_stats()
    #             grp_stats_object = GroupStats(
    #                 block_df, EXP_OR_SIM, subj_level_stats_all_subs_bw[block_])
    #             grp_descr_stats_df = grp_stats_object.perform_group_descr_stats(
    #                 group_by='block_type')
    #             grp_lvl_stats_bw[block_] = pd.concat([grp_lvl_stats_bw[block_],
    #                                                 grp_descr_stats_df],
    #                                                 ignore_index=True)
    #             grp_lvl_stats_bw[block_] = pd.concat([grp_lvl_stats_bw,
    #                                                 grp_descr_stats_df],
    #                                                 ignore_index=True)

    #     edited_grp_lvl_stats = True

    # grp_stats = GroupStats(events_all_subs_df, EXP_OR_SIM, descr_stats_all_subs_df)

    # if EXP_OR_SIM == 'sim':
    #     agent_group_descr_stats_incomplete = False
    #     for agent in list(events_all_subs_df['agent'].unique()):
    #         if agent not in grp_lvl_stats_df.sub_id.values:
    #             agent_group_descr_stats_incomplete = True
    #             break
    #     if agent_group_descr_stats_incomplete:
    #         grp_lvl_stats_df = pd.concat([grp_lvl_stats_df,
    #                                     grp_stats.perform_group_descr_stats(group_by='agent')],
    #                                     ignore_index=True)

    #         # HIER WIRDS INTERESSANT
    #         for block_, block_df in events_all_subs_bw.items():

    #             # whole sample descripte stats
    #             descr_stats_whole_sample_bw = DescrStats(
    #                 block_df, EXP_OR_SIM, subject='whole_sample_b')
    #             grp_lvl_stats_bw[
    #                 block_] = descr_stats_whole_sample_bw.perform_descr_stats()
        
    #             # descriptive stats group by agent
    #             grp_stats_object = GroupStats(
    #                 block_df, EXP_OR_SIM, subj_level_stats_all_subs_bw[block_])
    #             grp_descr_stats_df = grp_stats_object.perform_group_descr_stats(
    #                 group_by='agent')

    #             grp_lvl_stats_bw[block_] = pd.concat([grp_lvl_stats_bw[block_],
    #                                                 grp_descr_stats_df],
    #                                                 ignore_index=True)

    #         edited_grp_lvl_stats = True

    # # Change check for 'output_incomplete' to output-specific check
    # if edited_t_wise_stats:
    #     print('Computing trialwise stats')
    #     t_wise_stats_df = grp_stats.eval_t_wise_stats(
    #         groupby='trial_cont_overallb')

    # # TODO: not yet robust!
    # # if dataset == 'exp':
    # t_wise_bw = {}
    # for block_, block_df in events_all_subs_bw.items():
    #     grp_stats = GroupStats(block_df, EXP_OR_SIM, subj_level_stats_all_subs_bw[block_])
    #     t_wise_bw[block_] = grp_stats.eval_t_wise_stats(groupby='trial_cont')


    # if EXP_OR_SIM == 'sim':
    #     print('Computing trialwise stats for each agent ')
    #     agent_list = list(events_all_subs_df.agent.unique())
    #     t_wise_bw_agent = {}
    #     t_wise_stats_all_blocks = {}

    #     for agent in agent_list:
    #         # Initialize dictionary for blockwise trialwise counts
    #         t_wise_bw_agent[agent] = {}
    #         t_wise_stats_all_blocks[agent] = {}

    #         # ------Compute trialwise stats overall blocks--------
    #         # Create object to group events_df all subs by agent
    #         agent_group_by_object = events_all_subs_df.groupby('agent')

    #         # Iterate through agents
    #         for agent_, agent_df in agent_group_by_object:
    #             grp_stats_thisagent_all_blocks = GroupStats(
    #                 agent_df, EXP_OR_SIM, descr_stats_all_subs_df)
    #             t_wise_stats_all_blocks[agent_] = \
    #                 grp_stats_thisagent_all_blocks.eval_t_wise_stats(
    #                     groupby='trial_cont_overallb')

        # # Save data  # TODO: move downwards, once working
        # for agent, t_wise_thisagent_df in t_wise_stats_all_blocks.items():
        #     t_wise_thisagent_df.to_pickle(
        #         f'{dir_mgr.paths.t_wise_stats_fn}_agent-{agent}.pkl')
        #     with open(f'{dir_mgr.paths.t_wise_stats_fn}_agent-{agent}.tsv', 'w') as f:
        #         f.write(t_wise_thisagent_df.to_csv(sep='\t', na_rep='n/a'))

        # # ------Compute trialwise stats for each block--------
        # for block_, block_df in events_all_subs_bw.items():

        #     # Create object to group this blocks events_df by agent
        #     agentgroupby_object = block_df.groupby('agent')

        #     # Iterate through agents
        #     for agent, agent_block_df in agentgroupby_object:
        #         grp_stats_thisagent = GroupStats(
        #             agent_block_df, EXP_OR_SIM, subj_level_stats_all_subs_bw[block_])
        #         t_wise_bw_agent[agent][
        #             block_] = grp_stats_thisagent.eval_t_wise_stats(
        #             groupby='trial_cont')

        # # Save data  # TODO: move downwards, once working
        # for agent, t_wise_bw_thisagent in t_wise_bw_agent.items():
        #     for block_, block_df in t_wise_bw_thisagent.items():
        #         block_df.to_pickle(
        #             f'{dir_mgr.paths.t_wise_stats_fn}_agent-{agent}_run-{block_:02d}.pkl')
        #         with open(
        #                 f'{dir_mgr.paths.t_wise_stats_fn}_agent-{agent}_'
        #                 f'run-{block_:02d}.tsv', 'w') as f:
        #             f.write(block_df.to_csv(sep='\t', na_rep='n/a'))

    # if edited_r_wise_stats:
    #     print('Computing roundwise stats')
    #     descr_stats_all_subs_df = descr_stats_all_subs_df[descr_stats_all_subs_df[
    #         'sub_id'].isin(sub_id_list)]
    #     grp_stats.descr_df = descr_stats_all_subs_df
    #     r_wise_stats_df = grp_stats.eval_r_wise_stats()

    # ---------------------------------------------------------------------------------------------
    # --------------Save files-------------------------
    # ---------------------------------------------------------------------------------------------
    if edited_processed_events_all_subs:
        with open(f'{dir_mgr.paths.events_all_subs_fn}.tsv', 'w',
                  encoding="utf8") as tsv_file:
            tsv_file.write(events_all_subs_df.to_csv(sep='\t', na_rep='n/a'))
        events_all_subs_df.to_pickle(f'{dir_mgr.paths.events_all_subs_fn}.pkl')
        # TODO: bw-events not robust!
        # for block_, block_df in events_all_subs_bw.items():
        #     block_df.to_pickle(
        #         f'{dir_mgr.paths.this_analyses_proc_data_path}/sub-all_task-th_run-{block_:02d}_beh.pkl')
        #     with open(
        #             f'{dir_mgr.paths.this_analyses_proc_data_path}/'
        #             f'sub-all_task-th_run-{block_:02d}_beh.tsv', 'w',
        #             encoding="utf8") as f:
        #         f.write(block_df.to_csv(sep='\t', na_rep='n/a'))

    if edited_descr_stats:
        with open(f'{dir_mgr.paths.subj_lvl_descr_stats_fn}.tsv', 'w',
                  encoding="utf8") as tsv_file:
            tsv_file.write(descr_stats_all_subs_df.to_csv(sep='\t', na_rep='n/a'))
        descr_stats_all_subs_df.to_pickle(
            f'{dir_mgr.paths.subj_lvl_descr_stats_fn}.pkl')

        # TODO: bw-events not robust!
        # for block_, block_df in subj_level_stats_all_subs_bw.items():
        #     block_df.to_pickle(f'{dir_mgr.paths.subj_lvl_descr_stats_fn}_run-{block_:02d}.pkl')
        #     with open(f'{dir_mgr.paths.subj_lvl_descr_stats_fn}_run-{block_:02d}.tsv', 'w') as f:
        #         f.write(block_df.to_csv(sep='\t', na_rep='n/a'))

    # if edited_grp_lvl_stats:
    #     with open(f'{dir_mgr.paths.grp_lvl_descr_stats_fn}.tsv', 'w') as tsv_file:
    #         tsv_file.write(grp_lvl_stats_df.to_csv(sep='\t', na_rep='n/a'))
    #     grp_lvl_stats_df.to_pickle(f'{dir_mgr.paths.grp_lvl_descr_stats_fn}.pkl')
    #     for block_, block_df in grp_lvl_stats_bw.items():
    #         block_df.to_pickle(f'{dir_mgr.paths.grp_lvl_descr_stats_fn}_run-{block_:02d}.pkl')
    #         with open(f'{dir_mgr.paths.grp_lvl_descr_stats_fn}_run-{block_:02d}.tsv', 'w') as f:
    #             f.write(block_df.to_csv(sep='\t', na_rep='n/a'))

    # if edited_t_wise_stats:
    #     with open(f'{dir_mgr.paths.t_wise_stats_fn}.tsv', 'w') as tsv_file:
    #         tsv_file.write(t_wise_stats_df.to_csv(sep='\t', na_rep='n/a'))
    #     t_wise_stats_df.to_pickle(f'{dir_mgr.paths.t_wise_stats_fn}.pkl')
    #     # TODO: bw-events not robust!
    #     for block_, block_df in t_wise_bw.items():
    #         block_df.to_pickle(f'{dir_mgr.paths.t_wise_stats_fn}_run-{block_:02d}.pkl')
    #         with open(f'{dir_mgr.paths.t_wise_stats_fn}_run-{block_:02d}.tsv', 'w') as f:
    #             f.write(block_df.to_csv(sep='\t', na_rep='n/a'))

    # if edited_r_wise_stats:
    #     with open(f'{dir_mgr.paths.r_wise_stats_fn}.tsv', 'w') as tsv_file:
    #         tsv_file.write(r_wise_stats_df.to_csv(sep='\t', na_rep='n/a'))
    #     r_wise_stats_df.to_pickle(f'{dir_mgr.paths.r_wise_stats_fn}.pkl')


if __name__ == "__main__":
    EXP_LABEL = "exp_msc_50parts"  # 'exp_msc' or 'sim_100_msc' or 'test'""
    EXP_OR_SIM = "sim"  # str(input("Enter dataset ('exp' or 'sim): "))
    DIM = 5
    N_BLOCKS = 3

    main()
