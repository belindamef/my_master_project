import numpy as np
import os
import pandas as pd
from PIL import Image
from psychopy import monitors, visual
from utilities.abm_structure import AbmStructure
from utilities.create_stimuli import StimulusCreation

# Specify dataset, 'sim' or 'beh' and run (e.g. 'pretest_4')
dataset = 'sim'
run = 'sim_25_12_6_neworder'
dim = 5

# Specify directories and create if not existent
working_dir = os.getcwd()
project_dir = os.sep.join(working_dir.split(os.sep)[:4])  # Should be Users/<{$USER}>/<{$PROJECTFOLDER}>
data_dir = os.path.join(project_dir, '02-data')
results_dir = os.path.join(project_dir, '03-results')
figures_dir = os.path.join(project_dir, '04-figures')
input_data_dir = os.path.join(data_dir, f'{dataset}', f'{run}')
out_proc_data_dir = os.path.join(results_dir, f'{dataset}', f'{run}', 'processed_data')
out_descr_stats_dir = os.path.join(results_dir, f'{dataset}', f'{run}')
out_fig_dir = os.path.join(figures_dir, f'{dataset}', f'{run}')
if not os.path.exists(out_proc_data_dir):
    os.makedirs(out_proc_data_dir)
if not os.path.exists(out_descr_stats_dir):
    os.makedirs(out_descr_stats_dir)
if not os.path.exists(out_fig_dir):
    os.makedirs(out_fig_dir)

# Specify experimental parameter
blocks = 1  # No. of task blocks (each block has different tr location, but same hiding spots
rounds = 10  # No. of hunting rounds per task block
dim = 5  # dimension: No. of rows and columns of gridworld
trials = 12
n_hides = 6  # No. of  hiding spots in gridworld
n_nodes = dim ** 2  # No. of fields in the gridworld

# Initialize components stimulus creation
gridsize = 12  # Size of gridworld in cm
cube_size = gridsize / dim  # size of one cube in the gridworld

# Open event file
sub_id = 'A602'
proc_data_fn = os.path.join(out_proc_data_dir, f'events_sub-{sub_id}_proc')
events_this_sub_df = pd.read_pickle(f'{proc_data_fn}.pkl')
# Create subject-specific figure output-directory if not existent
out_fig_sub_dir = os.path.join(out_fig_dir, f'sub-{sub_id}')
if not os.path.exists(out_fig_sub_dir):  # Create if non existent
    os.makedirs(out_fig_sub_dir)

# Moves Visualization
# ----------------------------------------------------------------------------
s1_t = np.array(events_this_sub_df['s1_pos'].tolist())
s7_c = np.array(events_this_sub_df['s3_tr_loc']).tolist()

# monitor setting
my_mac = monitors.Monitor(name='my_mac')
my_mac.setSizePix((1280, 800))
my_mac.setWidth(28.7)
my_mac.setDistance(33.872)
my_mac.saveMon()

# screen setup
win = visual.Window(color=[-1, -1, -1],
                    monitor="my_mac", units='cm')


# Create dictionary for stimulus creation
task_params = {
    "my_mac": my_mac,
    "win": win,
    "dim": dim,
    "gridsize": gridsize,
    "n_hides": n_hides,
    "trials": trials,
    "rounds": rounds,
    "blocks": blocks,
    "s1_pos": s1_t,
    "s7_tr_loc": s7_c
}

# Create gridworld background
block_filenames = []  # Initialize image name list
for block in range(blocks):
    #stim_init.block = block  # Embed block count
    hround_filenames = []  # Initialize image name list
    for hround in range(rounds):
        #stim_init.hround = hround  # Embed hunting round count
        stimuli = StimulusCreation(task_params)  # Create gridworld stimulus
        #grid_init.hround = hround  # Embed hunting round count
        #grid_init.block = block  # Embed block count
        stimuli.create_stimuli()
        stimuli.grid.draw()
        stimuli.create_pos_stim_for_fig(block=block, hround=hround)
        stimuli.startcube.draw()
        stimuli.starttext.draw()
        stimuli.endcube.draw()
        stimuli.endtext.draw()
        stimuli.treasure.draw()
        win.flip()
        win.getMovieFrame()
        win.saveMovieFrames(f"{out_fig_sub_dir}/block-{block}_hround-{hround}.png")
        hround_filenames.append(f"{out_fig_sub_dir}/block-{block}_hround-{hround}.png")

    # Concatenate multiple round png-files to one figure
    # ------------------------------------------------------------------------
    hround_figures = [Image.open(x) for x in hround_filenames]
    c_widths, c_heights = zip(*(i.size for i in hround_figures))
    # Create new figure (one block with all rounds) with one gridworld for each round
    total_width_c = sum(c_widths)
    max_height_c = max(c_heights)
    block_figure = Image.new('RGB', (total_width_c, max_height_c))
    x_offset = 0
    for hround in hround_figures:
        block_figure.paste(hround, (x_offset, 0))
        x_offset += hround.size[0]
    block_figure.save(f"{out_fig_sub_dir}/block-{block}.png")
    block_filenames.append(f"{out_fig_sub_dir}/block-{block}.png")

# Concatenate multiple block png-files to one run-figure
block_figures = [Image.open(x) for x in block_filenames]
b_widths, b_heights = zip(*(i.size for i in block_figures))
# Create new figure with all blocks stapled
total_width_b = max(b_widths)
total_height_b = sum(b_heights)
run_figure = Image.new('RGB', (total_width_b, total_height_b))
y_offset = 0
for block_figure in block_figures:
    run_figure.paste(block_figure, (0, y_offset))
    y_offset += block_figure.size[1]
run_figure.save(f"{out_fig_sub_dir}/all_blocks.png")

# # Create new figure with one gridworld for each this_trial
# total_width = int(sum(widths) / 3)
# max_height = max(heights) * 3
# new_fig = Image.new('RGB', (total_width, max_height))

# y_offset = 0
# for row in range(3):
#     x_offset = 0
#     for fig in hround_figures[(0 + row*6):(6 + row*6)]:
#         new_fig.paste(fig, (x_offset, y_offset))
#         x_offset += fig.size[0]
#     y_offset += fig.size[1]

# new_fig.save(n_nodes"{out_fig_sub_dir}/all_trials.png")


win.close()  # Close the window
# core.quit()  # Close psychopy
