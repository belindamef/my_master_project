import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import json
import os
import glob
from utilities.data_class import Data
from utilities.data_analyses import DescrStats
from utilities.abm_structure import AbmStructure

"""
This script plots a set of descriptive statistics on a treasure hunt task

Input


File creation
    tr_found.png  : Number of treasures found for each sub_id
    todo: update png that are created

"""
# Specify dataset
input_dataset = 'rem'

# Specify directories
working_dir = os.getcwd()  # working dir
project_dir = os.sep.join(working_dir.split(os.sep)[:4])  # Should be Users/<{$USER}>/<{$PROJECTFOLDER}>
data_dir = os.path.join(project_dir, 'data')  # data directory
sim_data_dir = os.path.join(data_dir, 'simulations')  # directory for generated data
input_data_dir = os.path.join(data_dir, input_dataset)  # input data directory
results_dir = os.path.join(project_dir, 'results')  # results directory
figures_dir = os.path.join(results_dir, 'figures')  # figure directory
output_dir = os.path.join(figures_dir, f'{input_dataset}')
if not os.path.exists(output_dir):  # Create if non existent
    os.makedirs(output_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# Create file list for subjects
event_file_list = glob.glob(input_data_dir + '/**/*events.tsv')
event_file_list.sort()

# Initialize dataframe dictionary and stats dataframe
stats_summary = pd.DataFrame(columns=['interrupted', 'duration',
                                      'n_blocks', 'n_rounds', 'n_trials',
                                      'drills', 'steps', 'n_tr'])

# ------Initialize figures-------------------------------------------------------
# Total number of treasure all participants
treasures_all_subs_fn = os.path.join(output_dir, f'treasures_all-subs.png')
treasures_all_subs, ax_1 = plt.subplots()  #todo: undone

# Plot numbers treasures found
no_tr_found, axs_tr = plt.subplots(1, len(event_file_list), sharey=True)
no_tr_found.subplots_adjust(hspace=0.5)
no_tr_found.suptitle('Number of treasures found')
tr_found_fn = os.path.join(output_dir, f'tr_found.png')

# Plot action distribution
a_dist, axs_act = plt.subplots(len(event_file_list), 1, sharex=True)
a_dist.suptitle('Action distribution over rounds')
a_dist_fn = os.path.join(output_dir, f'action_dis.png')

# Percentage tr disc no unv. hiding spots
tr_disc_dist, axs_tr_disc = plt.subplots(1, 4, sharey=True)
tr_disc_dist.suptitle('Treasure discovery over hiding spot unv')
tr_disc_dist_fn = os.path.join(output_dir, f'tr_disc_dist.png')

# treasures per round
tr_per_round, axs_per_round = plt.subplots(len(event_file_list), 1, sharex=True)
tr_per_round.suptitle('Treasures per round')
tr_per_round_fn = os.path.join(output_dir, f'tr_per_round')

# ------Loop through subjects----------------
for index, this_file in enumerate(event_file_list):

    # Get subject ID
    this_sub = this_file[(this_file.find('events') - 4):this_file.find('events') - 1]

    # Print progress
    print(f'Progress: processing sub_id-{this_sub}')

    # Get and prepare data
    data = Data(this_file)  # Instantiate data class object with this files data set
    data.prep_data()

    # Run descriptive tics
    stats = DescrStats(data.df)
    stats.count_all()
    stats_summary.loc[this_sub] = stats.stats.loc[0]  # Append this subjects stats stats

    stats.compute_blockwise_metrics()

    # Plot plots
    # ----------------------------------------------

    # Total number of treasures

    # Number of treasures per block
    values = stats.tr_per_block_df.values
    names = list(stats.tr_per_block_df.index.values)
    axs_tr[index].scatter(names, values)
    axs_tr[index].set_xlabel(f'{this_sub}')

    # Action distribution
    stats.df.groupby(['block', 'round', 'action']).size().unstack().plot(ax=axs_act[index], kind='bar', stacked=True, legend=False)
    axs_act[index].set_ylabel(f'{this_sub}')

    # Treasure discovery as a function of number of unveiled spots
    if this_sub.startswith('dr'):
        # Compute relative frequ treasure discovery dependent on no of unveiled hides
        number_treasures = stats.tr_hides_df.values
        number_hides_unv = list(stats.tr_hides_df.index.values)
        #axs_tr_disc[index].scatter(number_treasures, number_hides_unv)
        #axs_tr_disc[index].set_xlabel(n_nodes'{this_sub_ID}')

    # Treasure found in which round
    number_treasures_per_round = stats.tr_per_round_df.values
    t_round = list(stats.tr_per_round_df.index.values)
    axs_per_round[index].scatter(number_treasures, number_hides_unv)
    axs_per_round[index].set_ylabel(f'{this_sub}')

# Save figures
no_tr_found.savefig(tr_found_fn)

handles, labels = axs_act[-1].get_legend_handles_labels()
a_dist.legend(handles, labels, loc='upper right')
a_dist.savefig(a_dist_fn)

tr_disc_dist.savefig(tr_disc_dist_fn)

tr_per_round.savefig(tr_per_round_fn)

# Save stats stats tsv file
summary_stats_fn = os.path.join(results_dir, f'summary_stats_{input_dataset}.tsv')
with open(summary_stats_fn, 'w') as tsv_file:
    tsv_file.write(stats_summary.to_csv(sep='\t', na_rep='n/a'))

# --------------------------------------------------------------
# archive
# --------------------------------------------------------------
# Plot figures
fig = plt.figure()  # Create empty figure with no axes
fig, ax = plt.subplots()  # a figure with a single Axes
fig, axs_act = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes
