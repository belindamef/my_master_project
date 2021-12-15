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
sim_data_dir = os.path.join(data_dir, 'rawdata', 'sim', f'{exp_label}')
exp_proc_data_dir = os.path.join(data_dir, 'processed_data', 'exp', f'{exp_label}')
sim_proc_data_dir = os.path.join(data_dir, 'processed_data', 'sim', f'{exp_label}')
sim_100_proc_data_dir = os.path.join(data_dir, 'processed_data', 'sim', f'{exp_label}_sim_100')

ev_exp_fn = os.path.join(exp_proc_data_dir, 'sub-all_task-th_run-all_beh')
ev_sim_fn = os.path.join(sim_proc_data_dir, 'sub-all_task-th_run-all_beh')
ev_exp_run_fn = os.path.join(exp_proc_data_dir, 'sub-all_task-th_run-')
ev_sim_run_fn = os.path.join(sim_proc_data_dir, 'sub-all_task-th_run-')

ds_exp_fn = os.path.join(descr_stats_dir, 'exp', f'{exp_label}', 'descr_stats')
ds_sim_fn = os.path.join(descr_stats_dir, 'sim', f'{exp_label}', 'descr_stats')
ds_sim_100_fn = os.path.join(descr_stats_dir, 'sim', f'{exp_label}_sim_100', 'descr_stats')
grp_stats_exp_fn = os.path.join(descr_stats_dir, 'exp', f'{exp_label}', 'grp_lvl_stats')
grp_stats_sim_fn = os.path.join(descr_stats_dir, 'sim', f'{exp_label}', 'grp_lvl_stats')
grp_stats_sim_100_fn = os.path.join(descr_stats_dir, 'sim', f'{exp_label}_sim_100', 'grp_lvl_stats')
tw_exp_fn = os.path.join(descr_stats_dir, 'exp', f'{exp_label}', 't_wise_stats')
tw_sim_fn = os.path.join(descr_stats_dir, 'sim', f'{exp_label}', 't_wise_stats')
tw_sim_100_fn = os.path.join(descr_stats_dir, 'sim', f'{exp_label}_sim_100', 't_wise_stats')

# ----Load data---------------------------------------------
# Load overall data
exp_ev_all_subs_df = pd.read_pickle(f'{ev_exp_fn}.pkl')
sim_ev_all_subs_df = pd.read_pickle(f'{ev_sim_fn}.pkl')
descr_stats_exp_df = pd.read_pickle(f'{ds_exp_fn}.pkl')
descr_stats_sim_df = pd.read_pickle(f'{ds_sim_fn}.pkl')
exp_grp_stats_df = pd.read_pickle(f'{grp_stats_exp_fn}.pkl')
sim_grp_stats_df = pd.read_pickle(f'{grp_stats_sim_fn}.pkl')

# Load blockwise data
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

# -----Load experimental data
events_all_subs_bw_exp = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
descr_stats_all_subs_bw_exp = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
grp_lvl_stats_bw_exp = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
trialwise_bw_exp = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
for block_ in range(3):
    block_ += 1
    events_all_subs_bw_exp[block_] = pd.read_pickle(
        os.path.join(exp_proc_data_dir, f'sub-all_task-th_run-{block_:02d}_beh.pkl'))
    descr_stats_all_subs_bw_exp[block_] = pd.read_pickle(f'{ds_exp_fn}_run-{block_:02d}.pkl')
    grp_lvl_stats_bw_exp_both_rows = pd.read_pickle(f'{grp_stats_exp_fn}_run-{block_:02d}.pkl')
    grp_lvl_stats_bw_exp[block_] = grp_lvl_stats_bw_exp_both_rows[grp_lvl_stats_bw_exp_both_rows['sub_id'].isin(['experiment'])]
    trialwise_bw_exp[block_] = pd.read_pickle(f'{tw_exp_fn}_run-{this_block:02d}.pkl')

# -----Load simulation data
events_all_subs_bw_sim = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
descr_stats_all_subs_bw_sim = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
grp_lvl_stats_bw_sim = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
trialwise_bw_sim = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
grp_lvl_stats_bw_sim_A = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
grp_lvl_stats_bw_sim_C = {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
for block_ in range(3):
    block_ += 1
    events_all_subs_bw_sim[block_] = pd.read_pickle(
        os.path.join(sim_proc_data_dir, f'sub-all_task-th_run-{block_:02d}_beh.pkl'))
    descr_stats_all_subs_bw_sim[block_] = pd.read_pickle(f'{ds_sim_fn}_run-{block_:02d}.pkl')
    grp_lvl_stats_bw_sim[block_] = pd.read_pickle(f'{grp_stats_sim_fn}_run-{block_:02d}.pkl')
    grp_lvl_stats_bw_sim_A[block_] = grp_lvl_stats_bw_sim[block_][
        grp_lvl_stats_bw_sim[block_]['sub_id'].isin(['Agent A1', 'Agent A2', 'Agent A3'])]
    grp_lvl_stats_bw_sim_C[block_] = grp_lvl_stats_bw_sim[block_][
        grp_lvl_stats_bw_sim[block_]['sub_id'].isin(['Agent C1', 'Agent C2', 'Agent C3'])]
    trialwise_bw_sim[block_] = pd.read_pickle(f'{tw_sim_fn}_run-{this_block:02d}.pkl')

# -----Load simulation 100 data---------------
events_all_subs_bw_sim_100 = {(run + 1): pd.DataFrame() for run in range(100)}
descr_stats_all_subs_bw_sim_100 = {(run + 1): pd.DataFrame() for run in range(100)}
grp_lvl_stats_bw_sim_100 = {(run + 1): pd.DataFrame() for run in range(100)}
trialwise_bw_sim_100 = {(run + 1): pd.DataFrame() for run in range(100)}
grp_lvl_stats_bw_sim_100_A = {(run + 1): pd.DataFrame() for run in range(100)}
grp_lvl_stats_bw_sim_100_C = {(run + 1): pd.DataFrame() for run in range(100)}
for block_ in range(100):
    block_ += 1
    events_all_subs_bw_sim_100[block_] = pd.read_pickle(
        os.path.join(sim_100_proc_data_dir, f'sub-all_task-th_run-{block_:02d}_beh.pkl'))
    descr_stats_all_subs_bw_sim_100[block_] = pd.read_pickle(f'{ds_sim_100_fn}_run-{block_:02d}.pkl')
    grp_lvl_stats_bw_sim_100[block_] = pd.read_pickle(f'{grp_stats_sim_100_fn}_run-{block_:02d}.pkl')
    grp_lvl_stats_bw_sim_100_A[block_] = grp_lvl_stats_bw_sim_100[block_][
        grp_lvl_stats_bw_sim_100[block_]['sub_id'].isin(['Agent A1', 'Agent A2', 'Agent A3'])]
    grp_lvl_stats_bw_sim_100_C[block_] = grp_lvl_stats_bw_sim_100[block_][
        grp_lvl_stats_bw_sim_100[block_]['sub_id'].isin(['Agent C1', 'Agent C2', 'Agent C3'])]
    trialwise_bw_sim_100[block_] = pd.read_pickle(f'{tw_sim_fn}_run-{this_block:02d}.pkl')
grp_lvl_stats_sim_100 = pd.read_pickle(f'{grp_stats_sim_100_fn}.pkl')
grp_lvl_stats_sim_100_A = grp_lvl_stats_sim_100[grp_lvl_stats_sim_100['sub_id'].isin(['Agent A1', 'Agent A2', 'Agent A3'])]
grp_lvl_stats_sim_100_C = grp_lvl_stats_sim_100[grp_lvl_stats_sim_100['sub_id'].isin(['Agent C1', 'Agent C2', 'Agent C3'])]
# trialwise stats each agent over all blocks
tw_sim_100_aw = {}
for agent in ['A1', 'A2', 'A3']:
    tw_sim_100_aw[agent] = pd.read_pickle(f'{tw_sim_100_fn}_agent-Agent {agent}.pkl')

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
# Extract task configuration-specific model components
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
gs = gridspec.GridSpec(4, 10)
bar_width = 0.6
half_bar_width = bar_width / 3

# ------Plot subject level treasures discovery--------------------------------
for block in range(n_blocks):
    ax[block] = plt.subplot(gs[block, 0:2])
    this_ax = ax[block]
    block += 1
    very_plotter.plot_bar(ax=this_ax, x=0,  height=grp_lvl_stats_bw_exp[block]['mean_tr_over_subs'],
                          yerr=grp_lvl_stats_bw_exp[block]['std_tr_over_subs'],
                          colors=col_exp[0])
    print(f"parti mean n_tr: {np.mean(ds_exp_run[block]['n_tr'])}")
    print(f"part std n_tr: {np.std(ds_exp_run[block]['n_tr'])}")
    very_plotter.plot_bar(ax=this_ax, x=[1, 1.5, 2], height=grp_lvl_stats_bw_sim_A[block]['mean_tr_over_subs'].values,
                          colors=col_A, bar_width=half_bar_width)
    very_plotter.plot_bar(ax=this_ax, x=[2.5, 3, 3.5], height=grp_lvl_stats_bw_sim_C[block]['mean_tr_over_subs'].values,
                          yerr=grp_lvl_stats_bw_sim_C[block]['std_tr_over_subs'], colors=col_C, bar_width=half_bar_width,
                          errorbar_size=3)

    very_plotter.plot_bar_scatter(this_ax, ds_exp_run[block]['n_tr'], color=col_exp[1], bar_width=bar_width)

    very_plotter.config_axes(this_ax,
                             y_label="Number of Treasures", y_lim=[0, n_tr_b],
                             xticks=[0, 1, 1.5, 2, 2.5, 3, 3.5],
                             xticklabels=['Participants', 'A1', 'A2', 'A3', 'C1', 'C2', 'C3'],
                             yticks=np.linspace(0, n_tr_b, 6),
                             ytickslabels=np.around(np.linspace(0, n_tr_b, 6), 2))

very_plotter.config_axes(ax[0], title="Task performance")

# Add letter
very_plotter.add_letters({0: ax[0]})

# ------Plot subject level %drills--------------------------------------------
for block in range(n_blocks):
    ax[block] = plt.subplot(gs[block, 2:4])
    this_ax = ax[block]
    block += 1
    very_plotter.plot_bar(ax=this_ax, x=0,  height=grp_lvl_stats_bw_exp[block]['mean_drills'],
                          colors=col_exp[0])
                           #yerr = grp_lvl_stats_bw_exp[block]['std_drills'])
    very_plotter.plot_bar(ax=this_ax, x=[1, 1.5, 2], height=grp_lvl_stats_bw_sim_A[block]['mean_drills'].values,
                          colors=col_A, bar_width=half_bar_width)
    very_plotter.plot_bar(ax=this_ax, x=[2.5, 3, 3.5], height=grp_lvl_stats_bw_sim_C[block]['mean_drills'].values,
                         colors=col_C, bar_width=half_bar_width)
                         #yerr=grp_lvl_stats_bw_sim_C[block]['std_drills'].values, errorbar_size=3)

    very_plotter.plot_bar_scatter(this_ax, descr_stats_all_subs_bw_exp[block]['mean_drills'], color=col_exp[1],
                                  bar_width=bar_width)

    very_plotter.config_axes(this_ax,
                             y_label="\% Informative actions", y_lim=[0, 1],
                             xticks=[0, 1, 1.5, 2, 2.5, 3, 3.5],
                             xticklabels=['Participants', 'A1', 'A2', 'A3', 'C1', 'C2', 'C3'],
                             yticks=np.linspace(0, 1.0, 6),
                             ytickslabels=np.around(np.linspace(0, 1.0, 6), 2))

very_plotter.config_axes(ax[0], title="Action choice rates")

# Add letter
very_plotter.add_letters({1: ax[0]})

# ------Plot trialwise action choices--------------------------------------------
s = 14
for block in range(n_blocks):
    block += 1
    ax[block] = plt.subplot(gs[block - 1, 4:10])
    this_ax = ax[block]
    x = tw_exp_run[block].trial.values
    y = tw_exp_run[block].mean_drill.values
    this_ax.scatter(x, y, alpha=0.2, s=s, color=col_exp[1], clip_on=False,
                    label="Participant's \n group average") #facecolors='none', edgecolors=col_exp[1])

    # for i, agent in enumerate(['A1', 'A2', 'A3']):
    #     action_types = list(np.nan_to_num(ev_sim_run[block][ev_sim_run[block]['sub_id'] == agent].action_type.values))
    #     y_drills = [(1.2 + i * 0.2) if action == 'drill' else np.nan for action in action_types]
    #     y_steps = [(1.1 + i * 0.2) if action == 'step' else np.nan for action in action_types]
    #     #this_ax.scatter(x, y_steps, marker="v", s=s, edgecolors=col_A[i + 1], label=f'{agent} step', facecolors='black') #, facecolors='none', edgecolors=col_A[i])
    #     #this_ax.scatter(x, y_drills, marker="v", s=s, color=col_A[i + 1], label=f'{agent} drill')
    #     this_ax.scatter(x, y_steps, marker="o", s=s, edgecolors=col_A[i], label=f'{agent} step', facecolors='none') #, facecolors='none', edgecolors=col_A[i])
    #     this_ax.scatter(x, y_drills, marker="o", s=s, color=col_A[i], label=f'{agent} drill')
    # for i, agent in enumerate(['C1', 'C2', 'C3']):
    #     #yellows_ = [col_C[0], col_C[2]]
    #     yellows_ = col_C
    #     action_types = list(np.nan_to_num(ev_sim_run[block][ev_sim_run[block]['sub_id'] == agent].action_type.values))
    #     y_drills = [(1.8 + i * 0.2) if action == 'drill' else np.nan for action in action_types]
    #     y_steps = [(1.7 + i * 0.2) if action == 'step' else np.nan for action in action_types]
    #     #this_ax.scatter(x, y_steps, marker="o", s=s, edgecolors=yellows_[i], label=f'{agent} step', facecolors='black')
    #     #this_ax.scatter(x, y_drills, marker="o", s=s, facecolors='none', edgecolors=yellows_[i], label=f'{agent} drill')
    #     this_ax.scatter(x, y_steps, marker="v", s=s, edgecolors=yellows_[i], label=f'{agent} step', facecolors='none')
    #     this_ax.scatter(x, y_drills, marker="v", s=s, color=yellows_[i], label=f'{agent} drill')

    # ------Plot round-wise  averages----------------------------
    # Experimental data
    tw_exp_run_grpby_round = tw_exp_run[block].groupby('round')
    x = [((round_ * 12) - 5.5) for round_, tw_exp_run_thisround in tw_exp_run_grpby_round]
    y = [np.mean(tw_exp_run_thisround['mean_drill']) for round_, tw_exp_run_thisround in tw_exp_run_grpby_round]
    e = [np.std(tw_exp_run_thisround['mean_drill']) for round_, tw_exp_run_thisround in tw_exp_run_grpby_round]
    this_ax.errorbar(x, y, alpha=0.7, markersize=4, yerr=e, color=col_exp[1], fmt='o', linestyle='-', linewidth=0.8,
                     label="Participant's \n group average over round", clip_on=False)

    # Specify agent colors
    agent_colors = col_A + col_C
    i = 0
    ev_sim_run_grpby_agent = ev_sim_run[block].groupby('sub_id')
    for agent, agent_df in ev_sim_run_grpby_agent:
        if 'C' in agent:
            continue
        agent_df_grpbyround = agent_df.groupby('round')
        vlines = [(round_ * 12 - 11) for round_, ev_this_agent_thisround in agent_df_grpbyround]
        x = [((round_ * 12) - 5.5) for round_, ev_this_agent_thisround in agent_df_grpbyround]
        y = [np.mean(ev_this_agent_thisround['action_type_num']) for round_, ev_this_agent_thisround in agent_df_grpbyround]
        e = [np.std(ev_this_agent_thisround['action_type_num']) for round_, ev_this_agent_thisround in agent_df_grpbyround]
        this_ax.errorbar(x, y, alpha=0.7, markersize=4, color=agent_colors[i], fmt='o', linestyle='-',
                         linewidth=0.8, clip_on=False, label=f"{agent}'s average over round")
        this_ax.vlines(vlines, colors=[.9, .9, .9], linewidth=.4, ymin=0, ymax=1)
        i += 1

    vlines.append(120)
    very_plotter.config_axes(this_ax,
                             x_lim=[0, 150], x_label='Trial', xticks=vlines,
                             xticklabels=np.around((np.linspace(1, 120, 11))).astype(int),
                             y_label="\% Informative actions", y_lim=[0, 1],
                             yticks=np.linspace(0, 1.0, 6),
                             ytickslabels=np.around(np.linspace(0, 1.0, 6), 2))

ax[1].legend(loc='center right')

very_plotter.config_axes(ax[1], title="Trial- and roundwise action choice rates")
# Add letter
very_plotter.add_letters({2: ax[1]})

# ------Plot simulation 100 results--------------------------------
# Treasure discovery
ax[4] = plt.subplot(gs[3, 0:2])
this_ax = ax[4]
block += 1
very_plotter.plot_bar(ax=this_ax, x=[1, 1.5, 2], height=grp_lvl_stats_sim_100_A['mean_tr_over_b'].values,
                      yerr=grp_lvl_stats_sim_100_A['std_tr_over_b'], errorbar_size=3,
                      colors=col_A, bar_width=half_bar_width)
very_plotter.plot_bar(ax=this_ax, x=[2.5, 3, 3.5], height=grp_lvl_stats_sim_100_C['mean_tr_over_b'].values,
                      yerr=grp_lvl_stats_sim_100_C['std_tr_over_b'], colors=col_C, bar_width=half_bar_width,
                      errorbar_size=3)

very_plotter.config_axes(this_ax,
                         title="Task performance",
                         y_label="Average Number of Treasures", y_lim=[0, n_tr_b],
                         xticks=[1, 1.5, 2, 2.5, 3, 3.5],
                         xticklabels=['A1', 'A2', 'A3', 'C1', 'C2', 'C3'],
                         yticks=np.linspace(0, n_tr_b, 6),
                         ytickslabels=np.around(np.linspace(0, n_tr_b, 6), 2))

# Add letter
very_plotter.add_letters({3: ax[4]})

# Overall Action choice rates
ax[5] = plt.subplot(gs[3, 2:4])
this_ax = ax[5]
very_plotter.plot_bar(ax=this_ax, x=[1, 1.5, 2], height=grp_lvl_stats_sim_100_A['mean_drills'].values,
                      colors=col_A, bar_width=half_bar_width)
                      #yerr=grp_lvl_stats_bw_sim_100_A['std_drills'], errorbar_size=3)
very_plotter.plot_bar(ax=this_ax, x=[2.5, 3, 3.5], height=grp_lvl_stats_sim_100_C['mean_drills'].values,
                     colors=col_C, bar_width=half_bar_width)
                     #yerr=grp_lvl_stats_bw_sim_C[block]['std_drills'].values, errorbar_size=3)

very_plotter.config_axes(this_ax,
                         title="Action choice rates",
                         y_label="\% Informative actions", y_lim=[0, 1],
                         xticks=[1, 1.5, 2, 2.5, 3, 3.5],
                         xticklabels=['A1', 'A2', 'A3', 'C1', 'C2', 'C3'],
                         yticks=np.linspace(0, 1.0, 6),
                         ytickslabels=np.around(np.linspace(0, 1.0, 6), 2))

# Roundwise action choices
s = 14
ax[6] = plt.subplot(gs[3, 4:10])
this_ax = ax[6]
x = tw_exp_run[1].trial.values

# ------Plot round-wise  averages----------------------------
# Specify agent colors
agent_colors = col_A + col_C
i = 0
for agent, agent_tw_df in tw_sim_100_aw.items():

    agent_df_grpbyround = agent_tw_df.groupby('round')
    vlines = [(round_ * 12 - 11) for round_, tw_this_agent_thisround in agent_df_grpbyround]
    x = [((round_ * 12) - 5.5) for round_, tw_this_agent_thisround in agent_df_grpbyround]
    y = [np.nanmean(ev_this_agent_thisround['p_drills']) for round_, ev_this_agent_thisround in agent_df_grpbyround]
    e = [np.nanstd(ev_this_agent_thisround['p_drills']) for round_, ev_this_agent_thisround in agent_df_grpbyround]
    this_ax.errorbar(x, y, alpha=0.7, markersize=4, color=agent_colors[i], fmt='o', linestyle='-',
                     linewidth=0.8, clip_on=False, label=f"{agent}'s average over round", yerr=e)
    this_ax.vlines(vlines, colors=[.9, .9, .9], linewidth=.4, ymin=0, ymax=1)
    i += 1

vlines.append(120)
very_plotter.config_axes(this_ax,
                         title="Roundwise action choice rates",
                         x_lim=[0, 150], x_label='Trial', xticks=vlines,
                         xticklabels=np.around((np.linspace(1, 120, 11))).astype(int),
                         y_label="\% Informative actions", y_lim=[0, 1],
                         yticks=np.linspace(0, 1.0, 6),
                         ytickslabels=np.around(np.linspace(0, 1.0, 6), 2))

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
