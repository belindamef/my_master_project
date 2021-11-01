"""
This script implements the treasure hunt task.

file creations
--------------
bids-compatible tabular file with behavioral data:
    /<raw_beh_data_dir>/<exp_label>/sub-<ID>/beh/sub-<ID>_task-th_beh.tsv

bids-compatible json file with meta data:
    /<raw_beh_data_dir>/<exp_label>/participants.json
    /<raw_beh_data_dir>/<exp_label>/task-th.json

files with additional data:
    /<raw_beh_data_dir>/<exp_label>_ext/sub-<ID>/all_data.pkl
    /<raw_beh_data_dir>/<exp_label>_ext/sub-<ID>/practice_events.tsv

Author: Belinda Fleischmann
"""

import numpy as np
import copy as cp
from psychopy import monitors, visual, core, event
import os.path
import pandas as pd
import json

from utilities.create_task_config import TaskConfigurator
from utilities.create_stimuli import StimulusCreation
from utilities.rowcol_to_xy import rowcol_to_xy
from utilities.node_to_rowcol import node_to_rowcol

np.set_printoptions(linewidth=500)

# -----------------------------------------------------------------------------
# -----Define paths and experimental parameters and presentation setup---------
# -----------------------------------------------------------------------------

# Specify work, stimuli, and data directories
working_dir = os.getcwd()
project_dir = os.sep.join(working_dir.split(os.sep)[:4])  # Should be Users/$USER/my_master_project
stimuli_dir = os.path.join(working_dir, 'stimuli')
raw_exp_data_dir = os.path.join(project_dir, 'data', 'rawdata', 'exp')  # raw experimental data output directory

if not os.path.exists(raw_exp_data_dir):  # create if non existent
    os.makedirs(raw_exp_data_dir)

# Specify experimental parameter
exp_blocks = 3  # No. of task blocks (each block has different tr location, but same hiding spots
exp_rounds = 10  # No. of hunting rounds per task block
exp_trials = 12
dim = 5  # dimension: No. of rows and columns of stimuli.grid
n_nodes = dim ** 2
n_hides = 6  # No. of  hiding spots in stimuli.grid
A = np.array([0, -dim, 1, dim, -1])  # set of actions

# Specify number of practice blocks rounds and trials
pract_blocks = 1
pract_rounds = 4
pract_trials = 12

# Specify experimental settings
unv_hide_with_tr = False  # If True, hiding spots will be unveiled upon treasure discovery
task_configuration_type = 'create_n_save'  # will load config files, if experiment name existent
# task_configuration_type = 'load'
# task_configuration_type = 'sample'

# Monitor setting
my_mac = monitors.Monitor(name='my_mac')
my_mac.setSizePix((1280, 800))
my_mac.setWidth(28.7)
my_mac.setDistance(33.872)
my_mac.saveMon()

# Get experiment name and create main data directory
exp_label = input("What kind of test is this? \n"
                  "(Entered name will be used as directory name): ")
exp_data_dir = os.path.join(raw_exp_data_dir, f'{exp_label}')  # main data directory
exp_data_ext_dir = os.path.join(raw_exp_data_dir, f'{exp_label}_ext')  # extended data
if not os.path.exists(exp_data_dir):  # create if non existent
    os.makedirs(exp_data_dir)
    print(f'Creating new data folder {exp_data_dir}')
else:
    print(f'Outputs will be saved to {exp_data_dir}')
if not os.path.exists(exp_data_ext_dir):  # create if non existent
    os.makedirs(exp_data_ext_dir)

# Store experiment data meta json files if not existent
events_json_fn = os.path.join(exp_data_dir, 'task-th_beh.json')

if not os.path.exists(events_json_fn):
    events_json_dic = {
        "ons": {
            "LongName": "onset",
            "Description": "event onset time with respect to the beginning of the first trial",
            "Units": "seconds"
        },
        "dur": {
            "LongName": "duration",
            "Description": "duration of the event, calculated by subtracting onset time "
                           "from the participants reaction time",
            "Units": "seconds"
        },
        "block_type": {
            "LongName": "block type",
            "Description": "the type of block the events belong to. it can either be a 'practice' "
                           f"block or a experimental block."
        },
        "block": {
            "LongName": "block number",
            "Description": f"the number of the block the event belongs to (values: 1-{exp_blocks}) "
        },
        "round": {
            "LongName": "round number",
            "Description": f"the number of the round the event belongs to (values: 1-{exp_rounds}); "
                           "events with the same block and round number have the same"
                           "treasure location"
        },
        "trial": {
            "LongName": "trial number",
            "Description": f"the number of the trial the event belongs to (values: 1-{exp_trials}); "
                           "for each trial there is one move / decision / action"
        },
        "s1_pos": {
            "LongName": "current position (state s1)",
            "Description": "value encoding the trial-specific current position in node-notation, "
                           "which corresponds to the directly observable state s1; "
                           f"values: 0-{dim ** 2 - 1}, enumerating the nodes in the {dim}x{dim}-dimensional grid world"
                           f"from left to right and top to bottom, e,g, 0 corresponds to the top left node "
                           f"{dim - 1} corresponds to the top right node "
                           f"and {dim * dim - 1} corresponds to the right bottom node"
        },
        "s2_node_color": {
            "LongName": f"node background colors (state s2)",
            "Description": f"({dim * dim - 1} x 1)-dimensional vector denoting each node's current background color.",
            "Levels:": [
                "0: black",
                "1: grey",
                "2: blue"
            ]
        },
        "s3_tr_loc": {
            "LongName": "treasure location (state s3)",
            "Description": "value encoding the round-specific location of the treasure in node-notation, "
                           f"values: 0-{dim ** 2 - 1} (for detailed description, see 's1_pos')",
        },
        "s4_hide_node": {
            "LongName": "hiding spot nodes (state s4)",
            "Description": f"({dim * dim - 1} x 1)-dimensional vector denoting for each node whether it "
                           f"is a hiding spot or not",
            "Levels:": [
                "0: not a hiding spot",
                "1: hiding spot"
            ]
        },
        "o": {
            "LonName": "observation",
            "Description": "observation the participant makes on the current position at the beginning"
                           "of the trial.",
            "Levels:": [
                "0: black background color",
                "1: grey background color",
                "2: blue background color",
                "3: treasure discovery"
            ]
        },
        "a_s1": {
            "LongName": "state dependent action set",
            "Description": "the set of actions available to the participant dependent on their"
                           f"current position. The state dependent action set corresponds to or "
                           f" is a subset of (-{dim}, 1, {dim}, -1), encoding drilling, upward, "
                           f"right, downward or left movements, while limiting possible action "
                           f"choices to those that do not cross the grid world borders in each "
                           f"direction. For instance, if the current position s1 = 0, "
                           f"a_s1 = (0, 1, {dim}) "
        },
        "action": {
            "LongName": "action",
            "Description": "value encoding participants action in current trial",
            "Levels": [
                " 0: drilling",
                "-6: take a step to the left",
                "-1: take a step upwards",
                "+1: take a step downwards",
                "+6: take a step to the right",
                "999: escape, i.e. experiment interrupted in trial the event belongs to"
            ]
        },
        "tr_disc": {
            "LongName": "treasure discovery",
            "Description": "value encoding whether or not the treasure was discovered in the current trial.",
            "Levels:": [
                "0: treasure was not found in the current trial",
                "1: treasure was found in the current trial"
            ]
        },
        "drill_finding": {
            "LongName": "drill finding",
            "Description": "value encoding whether or not a hiding spot was unveiled in the current trial, "
                           "if the participant chose the action to drill",
            "Levels": [
                "0: hiding spot was unveiled",
                "1: no hiding spot was unveiled"
            ]
        },
        "tr_found_on_blue": {
            "LongName": "treasure found on blue node",
            "Description": "value encoding whether the treasure was discovered on a blue node",
            "Levels": [
                "0: found on a black or grey node",
                "1: found on a blue node"
            ]
        },
        "n_black": {
            "LongName": "number of black nodes",
            "Description": f"value encoding the number of nodes that have a black background color"
                           f"values: 0-{(dim ** 2 - 1) - n_hides}",
        },
        "n_grey": {
            "LongName": "number of grey nodes",
            "Description": f"value encoding the number of nodes that have a grey background color"
                           f"values: 0-{(dim ** 2 - 1) - n_hides}",
        },
        "n_blue": {
            "LongName": "number of blue nodes",
            "Description": f"value encoding the number of nodes that have a blue background color"
                           f"values: 0-{n_hides}",
        },
        "hiding_spots": {
            "Description": f"({n_hides} x 1)-dimensional vector denoting the locations of the hiding spots"
                           f"for each entry, values range from values: 0-{dim ** 2 - 1} "
                           f"(for detailed description, see 's1_pos')"
        }
    }

    with open(events_json_fn, 'w') as json_f:
        json.dump(events_json_dic, json_f, indent=4)

participants_fn = os.path.join(exp_data_dir, 'participants')
if not os.path.exists(participants_fn):
    participants_dic = {
        "age": {
            "Description": "age of the participant",
            "Units": "years"
        },
        "sex": {
            "Description": "sex of the participant as reported by the participant",
            "Levels": {
                "male": "male",
                "female": "female",
                "other": "other"
            }
        },
        "handedness": {
            "Description": "handedness of the participant as reported by the participant",
            "Levels": {
                "left": "left",
                "right": "right",
                "ambidextrous": "ambidextrous"
            }
        },
    }
    with open(f'{participants_fn}.json', 'w') as json_file:
        json.dump(participants_dic, json_file, indent=4)

# Get subject ID and create subject output data directory
while True:
    try:
        sub_ID = str(input("Enter subject ID: "))
        if len(sub_ID) != 2:
            raise ValueError
        # Create subject folder (if not existing) and output filenames
        sub_dir = os.path.join(exp_data_dir, f'sub-{sub_ID}')
        sub_beh_dir = os.path.join(sub_dir, 'beh')
        sub_ext_dir = os.path.join(exp_data_ext_dir, f'sub-{sub_ID}')
        if not os.path.exists(sub_beh_dir):
            os.makedirs(sub_beh_dir)
        if not os.path.exists(sub_ext_dir):
            os.makedirs(sub_ext_dir)

        # Create file name stem; later add .tsv, .pkl etc
        fn_stem = f'{sub_beh_dir}/sub-{sub_ID}'
        for file in os.listdir(sub_dir):
            if file.startswith(f'sub-{sub_ID}'):
                raise Exception
        break
    except ValueError:
        print("Invalid entry. ID must have 2 digits.")
    except Exception:
        print(f'Existing data for subject {sub_ID}. \n'
              f'Please make sure you entered the correct ID')

# Get subject information
while True:
    try:
        age = int(input("Enter age: "))
        if not isinstance(age, int):
            raise ValueError
        if len(str(age)) > 2:
            raise ValueError
        break
    except ValueError:
        print("Invalid entry. Age must be integer and two digits")
part_info_dic = {'m': 'male', 'f': 'female', 'o': 'other',
                 'l': 'left', 'r': 'right', 'a': 'ambidextrous'}
while True:
    try:
        sex = part_info_dic[input("Enter sex (male, female, other): ")]
        if sex not in ['male', 'female', 'other']:
            raise ValueError
        break
    except ValueError:
        print("Invalid entry. Must be male, female or other")
while True:
    try:
        handedness = part_info_dic[input("Enter handedness: ")]
        if handedness not in ['left', 'right', 'ambidextrous']:
            raise ValueError
        break
    except ValueError:
        print("Invalid entry. must be left, right or a")
part_df = pd.DataFrame(index=range(1))
part_df['participant_id'] = f'sub-{sub_ID}'
part_df['age'] = age
part_df['sex'] = sex
part_df['handedness'] = handedness

# Add info to participant info
if os.path.exists(f'{participants_fn}.tsv'):
    with open(f'{participants_fn}.tsv', 'a') as f:
        print(f'sub-{sub_ID}', file=f, end='\t')
        print(age, file=f, end='\t')
        print(sex, file=f, end='\t')
        print(handedness, file=f)

else:
    with open(f'{participants_fn}.tsv', 'w') as tsv_file:
        tsv_file.write(part_df.to_csv(sep='\t', na_rep=np.NaN, index=False))

# Get user input whether or not to show instructions
while True:
    try:
        show_instr_details = input("Show instructions (y/n)?: ")
        if show_instr_details not in ["y", "n"]:
            raise ValueError
        break
    except ValueError:
        print("Invalid value")

# Get user input whether or not to run exercise trials
while True:
    try:
        run_practice_blocks = input("Start practice trials (y/n)?: ")
        if run_practice_blocks not in ["y", "n"]:
            raise ValueError
        break
    except ValueError:
        print("Invalid value")
# Include or exclude practice blocks according to user input
if run_practice_blocks == "y":
    blocks = pract_blocks + exp_blocks  # Initialize number of total blocks
    run_practice = True
else:
    blocks = cp.deepcopy(exp_blocks)
    run_practice = False

# Get screen setup and define psychopy window
while True:
    try:
        scr_pres = input("Select presentation screen: Full screen (f) / window (w): ")
        if scr_pres not in ["f", "w"]:
            raise ValueError
        break
    except ValueError:
        print("Invalid response.")
if scr_pres == "f":
    win = visual.Window(color=[-1, -1, -1], monitor="my_mac",
                        units='cm', fullscr=True)
else:
    win = visual.Window(size=[1280, 800], pos=[0, 0],
                        monitor="my_mac", units="cm")

# ------Create or load task configuration-------------------------------------
config_files_dir = os.path.join(working_dir, 'task_config',
                                f'b-{exp_blocks}_r-{exp_rounds}_'
                                f't-{exp_trials}', f'{exp_label}')
if os.path.exists(config_files_dir):
    print(f'loading task configuration for {exp_data_dir}')

# Create task config (object will load config if existing for task_params and sim_name)
task_configurator = TaskConfigurator(task_config_dir=config_files_dir,
                                     n_blocks=exp_blocks, n_rounds=exp_rounds,
                                     dim=dim, n_hides=n_hides)
task_configs = task_configurator.return_task_configuration()

config_pr_file_dir = os.path.join(working_dir, 'task_config',
                                  f'b-{pract_blocks}_r-{pract_rounds}_t-{pract_trials}',
                                  f'{exp_label}')

pr_task_configurator = TaskConfigurator(task_config_dir=config_pr_file_dir,
                                        n_blocks=exp_blocks, n_rounds=exp_rounds,
                                        dim=dim, n_hides=n_hides)
pr_task_configs = pr_task_configurator.return_task_configuration()

# ------Initialize global task variables--------------------------------------
s1_node = np.nan  # Current position
s1_rowcol = np.nan
s2_node_color = np.full(n_nodes, 0)  # Node colors (black=0, grey=1, blue=2)
s3_tr_loc = np.nan  # Treasure location in current round
s4_hide_node = np.full(n_nodes, 0)  # Hiding spot status (0=non_hide, 1=hide)
action = np.nan
drill_finding = np.nan
tr_found = np.nan  # Treasure discovery in current trial (0=no tr, 1=tr found)
tr_found_on_hide = np.nan  # Node status of location where treasure was found
n_black = 25  # number of black nodes in current trial
n_grey = 0  # number of grey nodes in current trial
n_blue = 0  # number of blue nodes in current trial
score = 0  # Score count (number of treasures)
this_resp = None  # Initialize this_resp variable for quit ("escape") option
key_list = ["left", "right",  # possible key responses during task routine
            "up", "down",
            "space", "escape"]
rawdata = pd.DataFrame()  # Dataframe for data recording

# ------Initialize for components stimulus creation---------------------------
gridsize = 12  # Size of gridworld in cm
cube_size = gridsize / dim  # size of one cube in the gridworld in cm
n_nodes = dim ** 2  # No. of fields in the gridworld

# Create dictionary for stimulus creation
stim_params = {
    "my_mac": my_mac,
    "win": win,
    "dim": dim,
    "gridsize": gridsize,
    "n_hides": n_hides,
    "trials": exp_trials,
    "rounds": exp_rounds,
    "blocks": exp_blocks,
}

# Create stimuli
stimuli = StimulusCreation(stim_params)
stimuli.create_stimuli()


# -----------------------------------------------------------------------------
# -----Define methods for task presentation------------------------------------
# -----------------------------------------------------------------------------


def prompt_welcome():
    """Prompt welcome text at beginning of task"""
    stimuli.instr_center.text = 'Welcome to the treasure hunt game!'
    stimuli.instr_center.draw()
    win.flip()
    core.wait(1.0)
    stimuli.instr_low.text = "Press 'return' to enter the game"
    stimuli.instr_center.draw()
    stimuli.instr_low.draw()
    win.flip()
    event.waitKeys()


def show_instructions():
    """Show interactive task instructions"""
    # Create hiding spot objects and stimuli
    possible_hides = np.array([22, 6, 4, 15, 24, 19, 13, 12, 11, 18, 17, 14, 5])
    # hides_demo = possible_hides[0:n_hides]  # Extract first n_hides numbers for hiding spots for demonstration
    hides_demo = np.array([24, 2, 23, 5, 6, 21])
    s4_hide_node_demo = np.full(n_nodes, 0)
    for node_ in hides_demo:
        s4_hide_node_demo[node_] = 1
    non_hides_demo = np.array([2, 9])
    non_hides_xy_demo = np.full((2, 2), np.nan)
    for index, hide in enumerate(non_hides_demo):
        non_hides_xy_demo[index] = rowcol_to_xy(node_to_rowcol(hide, dim), dim, gridsize)
    stimuli.create_hides_stims(s4_hide_node_demo)
    hide_stims_demo = stimuli.hides

    # Specify a starting position for demonstration
    pos_demo = np.array([11])
    pos_rowcol_demo = node_to_rowcol(pos_demo, dim)
    pos_xy_demo = rowcol_to_xy(pos_rowcol_demo, dim, gridsize)

    # ------Start routine "instructions"--------------------
    # Explain basic goal and show setting
    stimuli.instr_top.text = f'Your task is to find the treasure in this grid world.'
    stimuli.instr_top.draw()
    stimuli.instr_low.text = "Press 'return' to continue"
    stimuli.instr_low.autoDraw = True
    stimuli.grid.draw()
    stimuli.treasure.size = cube_size * 1.5
    stimuli.treasure.draw()
    win.flip()
    event.waitKeys()
    moves = cp.deepcopy(exp_trials)  # Initialize move count
    stimuli.move_count.text = f"Moves left: {moves} / {exp_trials}"
    stimuli.round_count.text = f"Round: 1 / {exp_rounds}"
    stimuli.instr_center.text = f'One game has {exp_rounds} rounds. ' \
                                f'In every round, your goal is to find the treasure within {exp_trials} moves.'
    stimuli.instr_center.draw()
    win.flip()
    event.waitKeys()
    stimuli.instr_top.text = f'The grid world will be displayed in the center, ' \
                             f'and your remaining moves, current round and score ' \
                             f'on the left.'
    stimuli.instr_top.draw()
    stimuli.grid.draw()
    stimuli.move_count.draw()
    stimuli.round_count.draw()
    stimuli.score_count.draw()
    stimuli.score_tr.draw()
    win.flip()
    event.waitKeys()

    # Show possible moving directions
    stimuli.instr_top.text = f'At the beginning of each round you will start from a new position. ' \
                             f'From there you can move to neighbouring fields.'
    stimuli.grid.draw()
    stimuli.instr_top.draw()
    stimuli.cube.pos = pos_xy_demo
    stimuli.cube.draw()
    #stimuli.starttext.pos = pos_xy_demo
    #stimuli.starttext.draw()
    stimuli.current_pos.pos = pos_xy_demo
    stimuli.current_pos.draw()
    stimuli.arrow_right.pos = rowcol_to_xy(node_to_rowcol((pos_demo + 1), dim), dim, gridsize)
    stimuli.arrow_left.pos = rowcol_to_xy(node_to_rowcol((pos_demo - 1), dim), dim, gridsize)
    stimuli.arrow_up.pos = rowcol_to_xy(node_to_rowcol((pos_demo - 5), dim), dim, gridsize)
    stimuli.arrow_down.pos = rowcol_to_xy(node_to_rowcol((pos_demo + 5), dim), dim, gridsize)
    stimuli.arrow_right.draw()
    stimuli.arrow_left.draw()
    stimuli.arrow_up.draw()
    stimuli.arrow_down.draw()
    win.flip()
    event.waitKeys()

    # Explain how to find a treasure and show example
    explanation_text = ["Once you step on the field where the treasure lies, it will be revealed.",
                        "Let's say for example the treasure was hidden on the field "
                        "above your current position."]
    for page in explanation_text:
        stimuli.instr_center.text = page
        stimuli.instr_center.draw()
        win.flip()
        event.waitKeys()
    stimuli.cube.pos = pos_xy_demo
    stimuli.current_pos.pos = pos_xy_demo
    stimuli.cube.draw()
    stimuli.current_pos.draw()
    stimuli.grid.draw()
    win.flip()
    event.waitKeys()
    stimuli.instr_top.text = f"If you now take a step upwards... \n" \
                             f"(press up-arrow key)"
    stimuli.instr_low.autoDraw = False
    stimuli.instr_top.autoDraw = True
    stimuli.grid.draw()
    stimuli.cube.draw()
    stimuli.current_pos.draw()
    event.waitKeys(keyList='up')
    win.flip()
    event.waitKeys(keyList='up')
    pos_demo -= 5  # Present movement to new field
    pos_xy_demo = rowcol_to_xy(node_to_rowcol(pos_demo, dim), dim, gridsize)
    stimuli.cube.pos = pos_xy_demo
    stimuli.cube.draw()
    stimuli.current_pos.pos = pos_xy_demo
    stimuli.current_pos.draw()
    stimuli.grid.draw()
    event.waitKeys(keyList='up')
    win.flip()
    event.waitKeys(keyList='up')
    core.wait(0.5)
    stimuli.instr_top.text = "...the treasure will be revealed."
    stimuli.instr_top.draw()
    pos_demo += 5  # Reset position
    stimuli.treasure.pos = pos_xy_demo
    stimuli.treasure.size = cube_size
    stimuli.cube.draw()
    stimuli.treasure.draw()
    stimuli.instr_low.autoDraw = True
    stimuli.grid.draw()
    win.flip()
    event.waitKeys(keyList='up')

    # Explain the purpose hiding spots
    pos_xy_demo = rowcol_to_xy(node_to_rowcol(pos_demo, dim), dim, gridsize)
    stimuli.cube.pos = pos_xy_demo
    stimuli.instr_center.text = f'But not all fields are potential treasure locations.'
    stimuli.instr_center.draw()
    stimuli.instr_top.autoDraw = False
    win.flip()
    event.waitKeys(keyList='up')
    stimuli.grid.draw()
    stimuli.instr_top.text = f'Only {n_hides} of the {n_nodes} fields are hiding spots. \n'
    stimuli.instr_top.autoDraw = True
    event.waitKeys()
    win.flip()
    event.waitKeys()
    event.waitKeys(keyList='up')

    # Show hiding spot examples
    pages = [f"Only {n_hides} of the {n_nodes} fields are hiding spots. \n",
             "The hiding spots you see here are examples. Each game has new hiding spots. ",
             f'Hiding spots remain the same throughout all 10 rounds of one game.',
             "At the beginning of each round the treasure "
             "will be hidden in one of these hiding spots.",
             "It could for example be here",
             "or here",  # TODO: This is starting position! Change
             "or here",
             "or here",
             "or here",
             "or here",
             "or here",
             "or here",
             "or here"]
    for index, example in enumerate(pages):
        #stimuli.instr_top.text = example
        for node_ in range(n_nodes):
            hide_stims_demo[node_].draw()
        if index >= 2:  # Only draw treasure starting at third flip
            stimuli.treasure.pos = hide_stims_demo[hides_demo[index - 2]].pos
            stimuli.treasure.draw()
        stimuli.grid.draw()
        win.flip()
        event.waitKeys()
        event.waitKeys(keyList='up')

    # Show not hiding spot examples
    not_hides_examples = ["but note here",
                          "or here",
                          "It will also never be at your starting position.",
                          "It will also never be at your starting position."]
    stimuli.treasure.size = 0.7 * cube_size  # Make treasure image smaller
    stimuli.treasure.opacity = 0.6  # Make treasure image transparent
    for index, example in enumerate(not_hides_examples):
        stimuli.instr_top.text = example
        stimuli.grid.draw()
        for node_ in range(n_nodes):
            hide_stims_demo[node_].draw()
        if index <= 1:  # Place treasure on non-hide for first two pages
            stimuli.treasure.pos = non_hides_xy_demo[index]
            stimuli.pos_cross.pos = non_hides_xy_demo[index]
            stimuli.treasure.draw()
            stimuli.pos_cross.draw()
        elif index == len(not_hides_examples) - 2:
            # Place START on a hiding spot for forelast page
            stimuli.starttext.pos = hide_stims_demo[hides_demo[1]].pos
            stimuli.starttext.draw()
        else:
            # Place treasure on START for last page
            stimuli.treasure.pos = hide_stims_demo[hides_demo[1]].pos
            stimuli.pos_cross.pos = hide_stims_demo[hides_demo[1]].pos
            stimuli.starttext.pos = hide_stims_demo[hides_demo[1]].pos
            stimuli.starttext.draw()
            stimuli.treasure.draw()
            stimuli.pos_cross.draw()

        win.flip()
        event.waitKeys(keyList='up')
        event.waitKeys()
        event.waitKeys(keyList='up')
    stimuli.treasure.opacity = 1.0  # Reset treasure opacity
    stimuli.treasure.size = cube_size  # Reset treasure size
    stimuli.instr_top.autoDraw = False
    for node_ in range(n_nodes):
        hide_stims_demo[node_].autoDraw = False

    # Explain how to unveil hiding spots
    one_option_version = "To find out where the hiding spots are located, " \
                         "you can use the unveiling option. "
    two_options_version = "There are two ways to find out where the hiding spots are located. " \
                          "One is by finding treasures. The same hiding spot could " \
                          "again harbour a treasure in a following round, " \
                          "as indicated by a change of the field background color." \
                          " Another one is to use the unveil-hiding-spot option"
    # Select explanation that fits the setting whether or not a hide should be unveiled upon treasure discovery
    if unv_hide_with_tr:
        n_options = two_options_version
    elif not unv_hide_with_tr:
        n_options = one_option_version
    explanation_text = ["Consequently, going on hiding spots will increase "
                        "you chances to find the treasure",
                        "But at the beginning of each game, the locations of hiding spots "
                        "are invisible.",
                        "This is how you see the grid world at the start.",
                        n_options,
                        "To unveil hiding spots, you can trade one move for a 'drill'.",
                        "Let's assume for example that your current location was "
                        "a hiding spot, which you don't know yet.\n",
                        "Let's see what happens when you choose to drill "
                        ]
    for index, page in enumerate(explanation_text):
        if index == 2:
            stimuli.instr_top.text = page
            stimuli.instr_top.draw()
            stimuli.grid.draw()
        else:
            stimuli.instr_center.text = page
            stimuli.instr_center.draw()
        event.waitKeys(keyList='up')
        win.flip()
        event.waitKeys()
        event.waitKeys(keyList='up')
    stimuli.instr_low.autoDraw = False

    # Demonstrate "drilling" action
    stimuli.instr_top.text = f"So this is where you stand. \n" \
                             f"Press 'space' to drill."
    stimuli.instr_top.draw()
    stimuli.cube.pos = hide_stims_demo[hides_demo[3]].pos
    stimuli.cube.draw()
    stimuli.move_count.autoDraw = True
    stimuli.round_count.autoDraw = True
    stimuli.score_count.autoDraw = True
    stimuli.score_tr.autoDraw = True
    stimuli.grid.draw()
    event.waitKeys(keyList='up')
    win.flip()
    event.waitKeys(keyList='up')
    event.waitKeys()
    moves -= 1  # Update move count
    stimuli.move_count.text = f"Moves left: {moves} / {exp_trials}"
    for x in range(60):  # Present drilling stimulus
        stimuli.drill.pos = hide_stims_demo[hides_demo[3]].pos
        stimuli.drill.setSize(stimuli.drill.ori % 300 / 300)
        stimuli.drill.setOri(5, '-')
        stimuli.drill.draw()
        stimuli.grid.draw()
        win.flip()
    stimuli.instr_top.text = f"Hiding Spot detected!"
    stimuli.instr_top.draw()
    hide_stims_demo[hides_demo[3]].draw()
    stimuli.score_tr.draw()
    stimuli.grid.draw()
    event.waitKeys(keyList='up')
    win.flip()
    event.waitKeys(keyList='up')
    core.wait(2.0)

    # Continue to explain the purpose of hiding spots
    stimuli.instr_top.text = f"The hiding spots will be unveiled and stay visible " \
                             f"for the rest of the game."
    stimuli.instr_top.draw()
    hide_stims_demo[hides_demo[3]].draw()
    stimuli.instr_low.autoDraw = True
    stimuli.grid.draw()
    win.flip()
    event.waitKeys()
    stimuli.move_count.autoDraw = False
    stimuli.round_count.autoDraw = False
    stimuli.score_count.autoDraw = False
    stimuli.score_tr.autoDraw = False
    stimuli.instr_center.text = f"Since the treasure will always be hidden at one of these " \
                                f"hiding spot, finding them will increase your chances to " \
                                f"find the treasure. \n "
    stimuli.instr_center.autoDraw = True
    win.flip()
    # # Show addition information about hide unveiling upton treasure discovery if option is selected
    # if not unv_hide_with_tr:
    #     event.waitKeys()
    #     stimuli.instr_center.text = f"You might have noticed that finding a treasure also unveils " \
    #                                 f"a hiding spot. \n" \
    #                                 f"However, those hides that are unveiled with the treasure " \
    #                                 f"will not remain visible, so you would have to remember them"
    #     win.flip()
    #     event.waitKeys()

    # Show stats of the rules
    rules_summary = [f"To summarize, "
                     f"You will play the treasure hunt game {exp_blocks} times and "
                     f"each game has {exp_rounds} rounds.",
                     f"In each round, your goal is to "
                     f"find the treasure within {exp_trials} moves.",
                     "The treasure will be hidden on one of the hiding spots, "
                     "which remain the same throughout all rounds of one game.",
                     "The treasure can be hidden at the same spot more than once, "
                     "but never at your starting position of the round.",
                     "For each move you can choose between two different actions: \n\n"
                     "Walk on a neighbouring field \n or \n drill at your current location",
                     "Either action will cost you one move."]
    # f"In the end, your score will be summed up over all games."]
    for page in rules_summary:
        stimuli.instr_center.text = page
        win.flip()
        event.waitKeys()

    # ------Ending Routine "Instructions"------
    stimuli.instr_center.text = f"Before we start the real game, you will play a practice game.\n\n" \
                                f"Please ask the instructor in case anything is unclear. \n\n" \
                                f"Or press 'return' when you're ready to start the exercise."
    stimuli.instr_low.autoDraw = False
    win.flip()
    event.waitKeys()
    stimuli.instr_center.autoDraw = False
    stimuli.ready.draw()  # Prompt "Ready"
    win.flip()
    core.wait(2.0)


def return_s4_values():
    """Sample hiding spot locations"""
    global s4_hide_node
    global hides_loc
    # hides_loc = np.random.choice(n_nodes, n_hides, replace=False)
    s4_hide_node = np.full(n_nodes, 0)
    for node_ in hides_loc:
        s4_hide_node[node_] = 1
    return s4_hide_node


def sample_start_n_tr(hides_loc_):
    """Sample starting position and treasure location"""
    start = np.random.choice(n_nodes, 1)  # node-notation

    # Sample treasure location from hiding spots until it's not starting position
    tr = cp.deepcopy(start)
    while tr == start:
        tr = np.random.choice(hides_loc_, 1)  # node-notation

    return start, tr


def identify_a_s1():
    """Identify and record state-dependent valid actions """
    global a_s1
    a_s1 = cp.deepcopy(A)
    for a in np.nditer(A):
        # Remove forbidden steps (walk outside border)
        if ((s1_node + a) < 0) or (s1_node + a) >= n_nodes or \
                ((s1_node % dim == 0) and a == -1) or \
                (((s1_node + 1) % dim == 0) and a == 1):
            a_s1 = a_s1[a_s1 != a]


def get_response():
    global ons
    global dur
    global continue_routine
    global this_resp

    while continue_routine:
        # Present decision prompt
        stimuli.instr_top.text = "Take a step or drill!"
        stimuli.grid.draw()
        stimuli.cube.draw()
        stimuli.instr_top.draw()
        ons = globalClock.getTime()  # Get and record onset time
        win.flip()

        # Get keyboard response and reaction time
        this_key = event.waitKeys(keyList=key_list, timeStamped=globalClock)
        this_resp = this_key[0][0]  # Get keyboard response
        dur = this_key[0][1] - ons  # Evaluate and record duration

        # If sub_id makes invalid action (i.e. crossing stimuli.grid boarder)
        if ((this_resp == 'left' and s1_rowcol[0, 1] == 0)
                or (this_resp == 'right' and s1_rowcol[0, 1] == dim - 1)
                or (this_resp == 'up' and s1_rowcol[0, 0] == 0)
                or (this_resp == 'down' and s1_rowcol[0, 0] == dim - 1)):
            stimuli.instr_top.text = "You cannot cross the boarder"
            stimuli.instr_top.draw()
            stimuli.grid.draw()
            stimuli.cube.draw()
            win.flip()
            core.wait(1.0)

            # If participants gives valid response, end routine after current trial
        else:
            continue_routine = False

        # Clear all events (they clog the buffer)
        event.clearEvents()


def translate_resp():
    """Evaluate subjects response and return action value"""
    global action
    global this_resp
    key_to_action = {
        "left": -1,
        "right": 1,
        "up": - dim,
        "down": + dim,
        "space": 0,
        "escape": 999
    }
    action = key_to_action[this_resp]


def perform_state_transition():
    """Perform state transitions"""
    global s1_node
    global action
    global s1_rowcol
    global s2_node_color
    global s4_hide_node
    global drill_finding

    s1_node += action  # Update position (s1);
    s1_rowcol = node_to_rowcol(s1_node, dim)

    # After informative actions:
    if action == 0:
        # Change node colors (trasnsition s_2):
        if s4_hide_node[int(s1_node)] == 0:  # If s1 is not a hiding spot
            if s2_node_color[int(s1_node)] == 0:  # If node is (was) black
                drill_finding = 0
            else:
                drill_finding = 3  # Drill finding = 2, fi drilled on unveiled spot (i.e. not black)
            s2_node_color[int(s1_node)] = 1  # Change color to grey (not a hiding spot)
        elif s4_hide_node[int(s1_node)] == 1:  # Elif s1 is a hiding spot
            if s2_node_color[int(s1_node)] == 0:  # If node is (was) black
                drill_finding = 1
            else:
                drill_finding = 3  # Drill finding = 3, if drilled on unveiled spot (i.e. not black)
            s2_node_color[int(s1_node)] = 2  # Change color to blue (hiding spot)


def eval_whether_treasure():
    global s1_node
    global s2_node_color
    global s3_tr_loc
    global tr_found
    global tr_found_on_hide

    if s1_node == s3_tr_loc:
        tr_found = 1

        # Evaluate whether found on hide
        if s2_node_color[int(s1_node)] == 2:
            tr_found_on_hide = 1
        elif s2_node_color[int(s1_node)] == 0:
            tr_found_on_hide = 0

    else:
        tr_found = 0
        tr_found = 1  # TODO: remove again (only for screenshot!)


def present_move():
    """Present move to new position, after step action"""
    global stimuli
    # Present "move to new position (s1_(t+1))"
    stimuli.grid.draw()
    stimuli.cube.pos = rowcol_to_xy(s1_rowcol, dim, gridsize)
    stimuli.cube.draw()
    stimuli.instr_top.draw()
    win.flip()
    event.waitKeys(keyList='up')


def present_drilling():
    """Present animated drill stimulus"""
    global stimuli

    # Present "drilling"
    for x in range(60):
        stimuli.drill.pos = rowcol_to_xy(s1_rowcol, dim, gridsize)
        stimuli.drill.setSize(stimuli.drill.ori % 300 / 300)
        stimuli.drill.setOri(5, '-')
        stimuli.drill.draw()
        stimuli.grid.draw()
        stimuli.instr_top.text = "Take a step or drill!"
        stimuli.instr_top.draw()
        win.flip()


def eval_action():
    global action
    global n_black
    global n_grey
    global n_blue
    global s2_node_color
    global score
    global score_b
    global hides_stims
    global tr_found
    translate_resp()
    perform_state_transition()

    # If participant decides to take a step
    # -----------------------------------------------------
    if action != 0:
        # Evaluate whether new position is treasure location
        eval_whether_treasure()
        present_move()

        # If new pos. is treasure loc., reveal treasure,
        if tr_found == 1:
            # Update Score count
            score += 1  # Update total score count
            score_b += 1  # Update blockwise score count
            stimuli.score_tr.autoDraw = False
            stimuli.create_score_tr_stim(score=score)
            stimuli.score_tr = stimuli.score_tr
            # Turn off autoDraws
            stimuli.move_count.autoDraw = False
            stimuli.round_count.autoDraw = False
            stimuli.score_count.autoDraw = False
            stimuli.score_tr.autoDraw = False
            # Present treasure and consequent hiding spot unveiling
            stimuli.instr_top.text = "=) Treasure found!"
            stimuli.instr_top.draw()
            stimuli.grid.draw()
            # hides[str(s1_node)].draw()
            stimuli.treasure.pos = rowcol_to_xy(s1_rowcol, dim, gridsize)
            stimuli.treasure.draw()
            win.flip()
            core.wait(3.0)
            for node_ in range(n_nodes):
                hides_stims[int(node_)].autoDraw = False

        # If new pos. is not treasure location, prompt info text
        elif tr_found == 0:
            stimuli.instr_top.text = "No treasure"
            stimuli.instr_top.draw()
            stimuli.grid.draw()
            stimuli.cube.draw()
            win.flip()
            event.waitKeys(keyList='up')
            core.wait(0.5)

    # If participant decides to drill (a == 0)
    # -----------------------------------------------------
    else:
        # Update treasure discovery count
        tr_found = 0

        n_black = np.count_nonzero(s2_node_color == 0)
        n_grey = np.count_nonzero(s2_node_color == 1)
        n_blue = np.count_nonzero(s2_node_color == 2)

        present_drilling()

        # If new position (s1_(t+1)) is a hiding spot
        if drill_finding == 1:
            # Present "hiding spot revealing"
            stimuli.instr_top.text = "Hiding Spot detected!"
            stimuli.instr_top.draw()
            stimuli.grid.draw()
            hides_stims[int(s1_node)].autoDraw = True
            win.flip()
            core.wait(0.5)
        # If new position (s1_(t+1)) is not a hiding spot
        elif drill_finding == 0:
            # Present "non-hiding spot revealing"
            stimuli.instr_top.text = "not a Hiding Spot"
            hides_stims[int(s1_node)].autoDraw = True
            stimuli.instr_top.draw()
            stimuli.grid.draw()
            win.flip()
            core.wait(0.5)


# -----------------------------------------------------------------------------
# -----Start task presentation-------------------------------------------------
# -----------------------------------------------------------------------------
prompt_welcome()
if show_instr_details == 'y':
    show_instructions()

globalClock = core.Clock()  # Instantiate clock object

# ------Start Routine "task"--------------------------------------------------
for this_block in range(blocks):

    # Re-initialize global variables
    s1_node = np.nan
    s1_rowcol = np.nan
    s2_node_color = np.full(n_nodes, 0)
    s3_tr_loc = np.nan
    s4_hide_node = np.full(n_nodes, 0)
    action = np.nan
    drill_finding = np.nan
    tr_found = np.nan
    tr_found_on_hide = np.nan
    n_black = 25
    n_grey = 0
    n_blue = 0

    # Check for quit; break block
    if this_resp == 'escape':
        break

    # ------Prepare Routine "practice" (if selected)------------------------------
    if run_practice:
        # Specify round and trial numbers for first two (practice) blocks
        if this_block < pract_blocks:
            block_type = "practice"
            n_rounds = cp.deepcopy(pract_rounds)
            trials = cp.deepcopy(pract_trials)
            if this_block == 0:
                # Inform participant about first practice block
                stimuli.instr_center.text = f"The practice game is shorter than the real game " \
                                            f"and the score does not count.\n\n " \
                                            f"Use this game to practice " \
                                            f"how to move around and drill. \n\n" \
                                            f"Press 'return' to start."
                stimuli.instr_center.draw()
                win.flip()
                event.waitKeys()
            elif 0 < this_block < pract_blocks:
                # Inform participant about new practice block
                stimuli.instr_center.text = f"You finished one exercise game and will now start " \
                                            f"a new one. \n\n" \
                                            f"Remember that the hiding spots will change in the new game.\n\n" \
                                            f"Press 'return' to start the next game."
                stimuli.instr_center.draw()
                win.flip()
                event.waitKeys()

        # ------Ending Routine "practice "--------------------------------------------
        elif this_block == pract_blocks:  # If this block is first real repetition of Routine "block"
            run_practice = False  # End practice repetitions of Routine "block"
            score = 0  # (Re-)Initialize score count
            stimuli.create_score_tr_stim(score=0)  # Reset score count
            stimuli.score_tr = stimuli.score_tr
            # Inform participant about end of practice games
            stimuli.instr_center.text = f"Good Job! You finished the exercise and " \
                                        f"can now start the real game. \n\n" \
                                        f"Please ask the instructor in case you have any questions.\n" \
                                        f"If everything is clear, press 'return' to enter the actual game."
            stimuli.instr_center.draw()
            win.flip()
            event.waitKeys()
            stimuli.instr_center.text = "Ready? \n\n Press key --> Start game"
            stimuli.instr_center.draw()
            win.flip()
            event.waitKeys()

    elif this_block == 0:
        stimuli.instr_center.text = "Ready? \n\n Press key --> Start game"
        stimuli.instr_center.draw()
        win.flip()
        event.waitKeys()

    # ------Fetch parameters for experimental block-------------------------------
    if not run_practice:
        block_type = "experiment"
        this_resp = None  # Initialize this_resp variable for quit ("escape") option
        n_rounds = cp.deepcopy(exp_rounds)
        trials = cp.deepcopy(exp_trials)

    # ------Prepare Routine "block"-----------------------------------------------
    score_b = 0  # blockwise score count
    rawdata_b = pd.DataFrame()  # dataframe for data of all rounds in current block

    # Sample and record hiding spots
    if run_practice:
        hides_loc = pr_task_configs['hides_loc'][this_block]

    else:
        check = this_block - (blocks - exp_blocks)
        hides_loc = task_configs['hides_loc'][this_block - (blocks - exp_blocks)]

    s4_hide_node = return_s4_values()
    hides_loc_t = hides_loc  # Record hiding spots
    hides_rowcol = node_to_rowcol(hides_loc, dim)  # Translate to row-columns notation
    s2_node_color = np.full(n_nodes, 0)

    # Create hiding spot stimuli
    stimuli.create_hides_stims(s4_hide_node)
    hides_stims = stimuli.hides

    # ------Start Routine "block"-------------------------------------------------
    for this_round in range(n_rounds):
        # Check for quit; break hunt round
        if this_resp == 'escape':
            break

        # ------Prepare Routine "round"-----------------------------------------------
        # Initialize arrays and objects for trial-wise data recording
        # Behavioral data
        ons_t = np.full(trials + 1, np.nan)  # Event onset
        dur_t = np.full(trials + 1, np.nan)  # Event duration

        # Task states
        s1_t = np.full(trials + 1, np.nan)  # Position in current trial
        s2_node_color_t = np.full((trials + 1, n_nodes), np.nan)

        # Variables for computations
        tr_disc_t = np.full(trials + 1, np.nan)  # Treasure discovery in current trial (after action)
        n_black_t = np.full(trials + 1, np.nan)  # number of black nodes
        n_grey_t = np.full(trials + 1, np.nan)  # number of grey nodes
        n_blue_t = np.full(trials + 1, np.nan)  # number of blue nodes
        drill_finding_t = np.full(trials + 1, np.nan)  # finding after drilling (0 = non-hide, 1 = hide)
        tr_found_on_hide_t = np.full(trials + 1, np.nan)  # Treasure found on unv hide or random

        # Observation, marginal beliefs, valence, decision, action
        o_t = np.full(trials + 1, np.nan)  # observation of treasure discovery
        a_s1_t = np.full(trials + 1, np.nan, dtype=object)  # state-dependent action set
        a_t = np.full(trials + 1, np.nan)  # Participant's action

        # ------Start Routine "round"-------------------------------------------------
        tr_found = 0  # Reset treasure discovery flag
        tr_found_on_hide = np.nan
        moves = cp.deepcopy(trials)  # Reset move count

        # Fetch starting position and tr location from task configuration
        if run_practice:
            start_node = pr_task_configs['s_1'][this_block, this_round]
            s3_tr_loc = pr_task_configs['s_3_tr_loc'][this_block, this_round]
        else:
            start_node = task_configs['s_1'][this_block - (blocks - exp_blocks), this_round]
            s3_tr_loc = task_configs['s_3_tr_loc'][this_block - (blocks - exp_blocks), this_round]

        # Initialize current position (s1) to equal start node
        s1_node = cp.deepcopy(start_node)

        # Prepare s1 stimulus
        s1_rowcol = node_to_rowcol(s1_node, dim)
        stimuli.cube.pos = rowcol_to_xy(s1_rowcol, dim, gridsize)

        # Record and print treasure location
        s3_tr_loc_t = np.full(trials + 1, s3_tr_loc)
        tr_rowcol = node_to_rowcol(s3_tr_loc, dim)  # Transform to row-col notation
        print(f"target_position (nodes / row-col): \n "
              f"{str(s3_tr_loc)}\n", tr_rowcol)

        for this_trial in range(trials):

            # ------Prepare Routine "trial"-----------------------------------------------
            # (Re-) initialize trial-wise dynamic variables
            ons = np.nan
            dur = np.nan
            drill_finding = np.nan
            a_s1 = np.nan
            action = np.nan

            # Update text for move and round counter stimuli
            stimuli.move_count.text = f"Moves left: {moves} / {trials}"
            stimuli.round_count.text = f"Round: {this_round + 1} / {n_rounds}"

            # Turn on autoDraws for move, round and counter stimuli
            stimuli.move_count.autoDraw = True
            stimuli.round_count.autoDraw = True
            stimuli.score_count.autoDraw = True
            stimuli.score_tr.autoDraw = True
            # Turn on autoDraws for unveiled hiding spot stimuli
            for node in range(n_nodes):
                if s2_node_color[node] in [1, 2]:
                    hides_stims[node].autoDraw = True
            this_resp = None  # Reset this response
            continue_routine = True

            # ------Start Routine "trial"-------------------------------------------------

            # ---------------------------------------------------------------------------
            # trial BEGINNING recordings
            # ---------------------------------------------------------------------------
            # values at beginning of this round, i.e. at end of last round (after a_{t-1})
            s1_t[this_trial] = s1_node
            s2_node_color_t[this_trial] = s2_node_color  # reflects state at end of last trial
            n_black_t[this_trial] = n_black
            n_grey_t[this_trial] = n_grey
            n_blue_t[this_trial] = n_blue
            o_t[this_trial] = cp.deepcopy(s2_node_color[int(s1_node)])
            # ---------------------------------------------------------------------------

            identify_a_s1()  # state s^1 dependent action set
            get_response()  # Get participant response
            if this_resp == 'escape':  # Check for quit
                break  # Break this round
            eval_action()  # evalute participant's action choixe

            # ------Ending Routine "trial"------
            # Update move count and move count stimulus
            moves -= 1
            stimuli.move_count.text = f"Moves left: {moves} / {trials}"

            # Check for quit
            if this_resp == 'escape':
                # turn off autoDraws and break trial loop
                stimuli.move_count.autoDraw = False
                stimuli.round_count.autoDraw = False
                stimuli.score_count.autoDraw = False
                stimuli.score_tr.autoDraw = False
                for hide_node in range(n_nodes):
                    hides_stims[f'[{hide_node}]'].autoDraw = False
                break  # break this trial

            # If move limit is reached, prompt "Move limit reached" info
            if moves == 0:
                # Turn off autoDraws
                stimuli.move_count.autoDraw = False
                stimuli.round_count.autoDraw = False
                stimuli.score_count.autoDraw = False
                stimuli.score_tr.autoDraw = False
                for node in range(n_nodes):
                    hides_stims[node].autoDraw = False
                # Create and draw new stimulus and flip window
                stimuli.instr_center.text = "Move limit reached."
                stimuli.instr_center.draw()
                win.flip()
                core.wait(2.0)

            # ---------------------------------------------------------------------------
            # trial END recordings
            # ---------------------------------------------------------------------------
            ons_t[this_trial] = ons  # Record onset time
            dur_t[this_trial] = dur  # Record duration time
            a_s1_t[this_trial] = a_s1
            a_t[this_trial] = action
            tr_disc_t[this_trial] = tr_found  # Record treasure discovery
            drill_finding_t[this_trial] = drill_finding
            tr_found_on_hide_t[this_trial] = tr_found_on_hide
            # ---------------------------------------------------------------------------

            # End round, if treasure discovered
            if tr_found == 1:
                # Record otherwise trial BEGINNING recordings
                s1_t[this_trial + 1] = s1_node
                s2_node_color_t[this_trial + 1] = s2_node_color
                n_black_t[this_trial + 1] = np.count_nonzero(s2_node_color == 0)
                n_grey_t[this_trial + 1] = np.count_nonzero(s2_node_color == 1)
                n_blue_t[this_trial + 1] = np.count_nonzero(s2_node_color == 2)
                break  # Break "trial" loop

            # completed all repeats of "trials"

        # ------Ending Routine "round"------------------------------------------------
        # Record data from all trials of this round round-specific dataframe
        rawdata_c = pd.DataFrame(index=range(0, trials + 1))
        rawdata_c['ons'] = ons_t
        rawdata_c['dur'] = dur_t
        rawdata_c['block_type'] = block_type
        rawdata_c['block'] = this_block + 1
        rawdata_c['round'] = this_round + 1
        rawdata_c['trial'] = range(1, trials + 2)
        rawdata_c['s1_pos'] = s1_t
        rawdata_c['s2_node_color'] = np.full(trials + 1, np.nan)
        rawdata_c['s2_node_color'] = rawdata_c['s2_node_color'].astype('object')
        for trial in range(trials + 1):
            rawdata_c.at[trial, 's2_node_color'] = s2_node_color_t[trial]
        rawdata_c['s3_tr_loc'] = s3_tr_loc_t
        rawdata_c['s4_hide_node'] = np.full(trials + 1, np.nan)
        rawdata_c['s4_hide_node'] = rawdata_c['s4_hide_node'].astype('object')
        for trial in range(trials + 1):
            rawdata_c.at[trial, 's4_hide_node'] = s4_hide_node
        rawdata_c['o'] = o_t
        rawdata_c['a_s1'] = np.full(trials + 1, np.nan)
        rawdata_c['a_s1'] = rawdata_c['a_s1'].astype('object')
        for t in range(trials + 1):
            rawdata_c.at[t, 'a_s1'] = a_s1_t[t]
        rawdata_c['action'] = a_t
        rawdata_c['tr_disc'] = tr_disc_t
        rawdata_c['drill_finding'] = drill_finding_t
        rawdata_c['tr_found_on_blue'] = tr_found_on_hide_t
        rawdata_c['n_black'] = n_black_t
        rawdata_c['n_grey'] = n_grey_t
        rawdata_c['n_blue'] = n_blue_t
        rawdata_c['hiding_spots'] = np.full(trials + 1, np.nan)
        rawdata_c['hiding_spots'] = rawdata_c['hiding_spots'].astype('object')
        for this_trial in range(trials + 1):
            rawdata_c.at[this_trial, 'hiding_spots'] = hides_loc_t

        # Append dataframe from 'this_round' to dataframe from 'this_block'
        rawdata_b = rawdata_b.append(rawdata_c, ignore_index=True)

        # Check quit
        if this_resp == 'escape':
            break  # Break this round

        # If 'this_round' is NOT forelast or last round, inform participant about new round and score
        if this_round < (n_rounds - 2):
            stimuli.instr_center.text = (f"You finished round {this_round + 1} of {n_rounds}. \n\n"
                                         f"There are {n_rounds - (this_round + 1)} rounds left in this game.\n\n"
                                         f"Press any key to continue.")
            stimuli.instr_center.draw()
            win.flip()
            event.waitKeys()

        # If 'this_round' is forelast round, inform participant about new round and score
        elif this_round == (n_rounds - 2):
            stimuli.instr_center.text = (f"You finished round {this_round + 1} of {n_rounds}. \n\n"
                                         f"There is {n_rounds - (this_round + 1)} round left in this game.\n\n"
                                         f"Press any key to continue.")
            stimuli.instr_center.draw()
            win.flip()
            event.waitKeys()

        # If 'this_round' is first or second round in first experimental block, repeat basic rules
        if run_practice or \
                (run_practice_blocks == 'y' and this_block > pract_blocks) or \
                (run_practice_blocks == 'n' and this_block > 1) or \
                this_round > 1:
            pass

        else:
            stimuli.instr_center.text = (f"Remember, that the hiding spots will remain the same over all rounds "
                                         f"but the treasure will be hidden at a new spot in every new round \n\n"
                                         f"Note that the treasure location is drawn at random and thus may be "
                                         f"located at the same spot more than once .\n\n"
                                         f"Press any key to continue.")
            stimuli.instr_center.draw()
            win.flip()
            event.waitKeys()

        # completed all repeats of "rounds"

    # ------Ending Routine "block"----------------------------------------------------
    # Append dataframe from 'this_block' to Dataframe for data over all blocks
    rawdata = rawdata.append(rawdata_b, ignore_index=True)

    # Check quit
    if this_resp == 'escape':
        break  # Break this block

    # If NOT last block, inform participant about start of new block (alias "game")
    if not run_practice:
        if this_block != blocks - 1:
            stimuli.instr_center.text = (f"You completed all rounds of this game. \n\n"
                                         f"You're total score: {score} treasure(s) \n\n"
                                         f"You will now start a new game with new hiding spots.\n\n"
                                         f"Press any key to start.")
            stimuli.instr_center.draw()
            win.flip()
            event.waitKeys()

        # If last block, inform participant about end of entire game
        else:
            stimuli.instr_center.text = (f"You finished the last round of the last game \n"
                                         f"This is the end of the experiment.\n\n"
                                         f"You're total score: {score} treasure(s)\n\n"

                                         f"Press 'return' ")

            stimuli.instr_center.draw()
            win.flip()
            event.waitKeys()
            stimuli.instr_top.text = "Thank you for participating!"
            stimuli.instr_low.text = f"Press any key to leave to game"
            stimuli.instr_top.draw()
            stimuli.instr_low.draw()
            stimuli.treasure.size = cube_size * 1.5
            stimuli.treasure.draw()
            win.flip()
            event.waitKeys()

# completed all repeats of "blocks"

# Clean up psychopy, close window
win.close()

# -----------------------------------------------------------------------------
# -----Save data---------------------------------------------------------------
# -----------------------------------------------------------------------------

# Write pkl file with complete data
rawdata.to_pickle(f"{sub_ext_dir}/all_data.pkl")

# Write complete raw data to event tabular file
with open(f"{sub_ext_dir}/allevents.tsv", 'w') as tsv_file:
    tsv_file.write(rawdata.to_csv(sep='\t', na_rep=np.NaN, index=False))

if rawdata.iloc[0]['block_type'] == 'practice':  # If practice trials exist
    index_exp_start = rawdata.where(rawdata['block_type'] == 'experiment').first_valid_index()
    n_prac_blocks = rawdata.iloc[rawdata.where(rawdata['block_type'] == 'practice').last_valid_index()]['block']

    experiment_data = rawdata.drop(rawdata.index[0:index_exp_start], axis=0)  # Create new df that drops practice blocks
    experiment_data.loc[:, 'block'] -= n_prac_blocks  # Reset block count to start with 1

    # Write practice data to extra file
    practice_data = rawdata.drop(rawdata.index[index_exp_start:-1], axis=0)  # Create new df that drops exp blocks
    with open(f'{sub_ext_dir}/practice_events.tsv', 'w') as tsv_file:
        tsv_file.write(practice_data.to_csv(sep='\t', na_rep=np.NaN, index=False))
else:
    experiment_data = cp.deepcopy(rawdata)

# Write bids compatible events.tsv to subject's beh/ directory
with open(f"{sub_beh_dir}/sub-{sub_ID}_task-th_beh.tsv", 'w') as tsv_file:
    tsv_file.write(experiment_data.to_csv(sep='\t', na_rep=np.NaN, index=False))

core.quit()

# End of script to run the treasure hunt task
