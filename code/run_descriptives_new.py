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


def main():

    # Specify directories and create if not existent
    dir_mgr = DirectoryManager()
    dir_mgr.define_raw_beh_data_out_path(data_type=EXP_OR_SIM,
                                         exp_label=EXP_LABEL,
                                         make_dir=False)
    dir_mgr.define_processed_data_path(data_type=EXP_OR_SIM,
                                       exp_label=EXP_LABEL,
                                       make_dir=True)
    dir_mgr.define_descr_stats_path(data_type=EXP_OR_SIM,
                                    exp_label=EXP_LABEL,
                                    make_dir=True)
    dir_mgr.define_stats_filenames()

    # Initialize or Load dataframes for event and descr stats
    # edited_processed_events_all_subs = False
    edited_descr_stats = False

    # ---------------------------------------------------------------------
    # Initialize or load dataframe for events_all_subs for processed data
    if os.path.exists(f'{dir_mgr.paths.events_all_subs_fn}.pkl'):

        # Load events all subs
        events_all_subs_df = pd.read_pickle(
            f'{dir_mgr.paths.events_all_subs_fn}.pkl')

    else:
        events_all_subs_df = pd.DataFrame()
        # edited_processed_events_all_subs = True

    # ---------------------------------------------------------------------
    # Initialize or load dataframe for subject level descr stats all subs
    if os.path.exists(f'{dir_mgr.paths.subj_lvl_descr_stats_fn}.pkl'):
        descr_stats_all_subs_df = pd.read_pickle(
            f'{dir_mgr.paths.subj_lvl_descr_stats_fn}.pkl')
        # for block_, block_df in subj_level_stats_all_subs_bw.items():
        #     subj_level_stats_all_subs_bw[block_] = pd.read_pickle(
        #         f'{dir_mgr.paths.subj_lvl_descr_stats_fn}_run-{block_:02d}.pkl')
    else:
        descr_stats_all_subs_df = pd.DataFrame()  # (one row per subject)
        edited_descr_stats = True

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
                dir_mgr.paths.this_analyses_proc_data_path,
                f'sub-{sub_id}_task-th_run-all_beh')
            if os.path.exists(f'{proc_data_fn}.pkl'):
                print(f'unpacking sub-{sub_id} proc_events.pkl')
                events_this_sub_df = pd.read_pickle(f'{proc_data_fn}.pkl')

            else:
                print(f'processing events data for sub-{sub_id}')
                # Get and prepare data from this subject's events and
                # append to events_df of all subs
                data_this_sub = Data(  # Instantiate data class object
                    EXP_OR_SIM, sub_id, events_fn, DIM)
                data_this_sub.prep_data()
                # --> Adds object attribute: data.event_data
                events_this_sub_df = data_this_sub.events_df

                # ------Save processed data--------
                with open(f'{proc_data_fn}.tsv', 'w',
                          encoding="utf-8") as tsv_file:
                    tsv_file.write(
                        events_this_sub_df.to_csv(
                            sep='\t', na_rep=np.NaN, index=False))
                events_this_sub_df.to_pickle(f'{proc_data_fn}.pkl')

            # Add this subs events to events_all_subs_df
            events_all_subs_df = pd.concat([events_all_subs_df,
                                            events_this_sub_df],
                                           ignore_index=True)

            descr_stats_this_sub = DescrStats(
                events_this_sub_df, EXP_OR_SIM, subject=sub_id)

            # (One row for this subject)
            descr_stats_this_sub_df = descr_stats_this_sub.perform_descr_stats(
            )

            descr_stats_all_subs_df = pd.concat([descr_stats_all_subs_df,
                                                 descr_stats_this_sub_df],
                                                ignore_index=True)
            edited_descr_stats = True

        else:
            print(
                f'Skipping processing for sub-{sub_id}, '
                f'already in descr_stats_all_subs_df')

    if edited_descr_stats:
        with open(f'{dir_mgr.paths.subj_lvl_descr_stats_fn}.tsv', 'w',
                  encoding="utf8") as tsv_file:
            tsv_file.write(
                descr_stats_all_subs_df.to_csv(sep='\t', na_rep='n/a'))
        descr_stats_all_subs_df.to_pickle(
            f'{dir_mgr.paths.subj_lvl_descr_stats_fn}.pkl')


if __name__ == "__main__":
    EXP_LABEL = "test_ahmm_11_20"  # 'exp_msc' or 'sim_100_msc' or 'test'""
    EXP_OR_SIM = "sim"  # str(input("Enter dataset ('exp' or 'sim): "))
    DIM = 2
    N_BLOCKS = 1

    main()
