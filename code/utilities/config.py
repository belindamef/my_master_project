"""This module contains functions to implement basic configurations
that are shared across scripts."""

from dataclasses import dataclass
import os
import copy as cp
import numpy as np
import pandas as pd


@dataclass
class Paths:
    """A class to store directory and file paths as string values

    Attributes
    ----------
    project : str
        path of project root parent directory (path-to/treasure-hunt
    data : str
        path to data directory
    sim_data: str
        path to directory to store data generated in data simulations
    """

    project = os.sep.join(os.path.dirname(os.getcwd()).split(os.sep))
    code = os.path.join(project, "code")
    task_configs = os.path.join(code, "task_config")
    data = os.path.join(project, "data")
    sim_data = os.path.join(data, "rawdata", "sim")
    exp_data = os.path.join(data, "rawdata", "exp")


@dataclass
class TaskDesignParameters:
    """A Class to store experimental parameters

    Attributes
    ----------
    n_blocks : int
        number of blocks in one run;
    n_rounds : int
        number of rounds in one block
    n_trials : int
        number of trials in one run
    dim : int
        size (i.e. no. of rows and columns) in the 2-dimensional grid
    n_hides : int
        number of hiding spots in the grid world
    n_nodes : int
        number of fields in the grid world
    """
    n_blocks = 1
    n_rounds = 10
    n_trials = 12
    dim = 5
    n_hides = 6
    n_nodes = dim ** 2


class TaskConfigurator:
    """A Class to create or load task configurations given a set of
    experimental parameters or configuration label respectively.
    Sampled task configuration npy files are written to config directory.

    Parameters
    ----------
    path : obj # TODO macht das so Sinn?
        Object of class Paths
    task_design_params: obj
        Object of class TaskDesignParams

    Attributes
    ----------
    task_config : dict  # TODO
        dict of {str : array_like}
        s_1 : (n_blocks)x(n_rounds)-dimensional array with values for
        starting positions
        s_3: (n_blocks)x(n_rounds)-dimensional array with values for
        treasure locations
        hides_loc: (n_blocks)x(n_hides)-dimensional array with values
        for hiding spot locations
    """

    new_config_needed = False
    config_label = None
    # Initialize task states
    task_config = {}

    def __init__(self, path, task_design_params):
        self.paths = path
        self.task_design_params = task_design_params

    def get_user_input(self):
        """Get user input for simulation task configuration"""
        self.new_config_needed = bool(
            input("Create new task configuration? (yes/no->return)"))
        if self.new_config_needed:
            self.config_label = input(
                "Enter label for new task configuration: ")
            self.task_design_params.n_blocks = int(input(
                "Enter number of blocks: "))
        else:
            self.config_label = input("Enter label of existing task config ("
                                      "'exp_msc'/'sim_100_msc')")

    def add_config_paths(self):
        """Add path to this task configurations config files dir to path obj"""
        self.paths.this_config = os.path.join(
            self.paths.task_configs, self.config_label)

    def sample_hiding_spots(self):
        """Sample hiding spots from a discrete uniform distribution over
         all nodes (without replacement)"""
        hides_loc = np.full((self.task_design_params.n_blocks,
                             self.task_design_params.n_hides), np.nan)
        for block in range(self.task_design_params.n_blocks):
            hides_loc[block] = np.random.choice(
                self.task_design_params.n_nodes,
                self.task_design_params.n_hides,
                replace=False)
        self.task_config['hides_loc'] = hides_loc

    def sample_start_pos(self):
        """Sample the starting position from a discrete uniform distribution
        over all nodes"""
        s_1 = np.full((self.task_design_params.n_blocks,
                       self.task_design_params.n_rounds), np.nan)
        for block in range(self.task_design_params.n_blocks):
            for round_ in range(self.task_design_params.n_rounds):
                s_1[block, round_] = np.random.choice(
                    self.task_design_params.n_nodes, 1)
        self.task_config['s_1'] = s_1

    def sample_treasure_loc(self):
        """Sample the tr location from a discrete uniform distribution over all
        hiding spots"""
        s_3 = np.full((self.task_design_params.n_blocks,
                       self.task_design_params.n_rounds), np.nan)
        for block in range(self.task_design_params.n_blocks):
            for round_ in range(self.task_design_params.n_rounds):
                # Set treasure to equal start position
                s_3[block, round_] = cp.deepcopy(
                    self.task_config['s_1'][block, round_])
                # Sample tr location until it's not the starting position s_0
                while s_3[block, round_] == self.task_config['s_1'][block,
                                                                    round_]:
                    s_3[block, round_] = np.random.choice(
                        self.task_config['hides_loc'][block], 1)
        self.task_config['s_3'] = s_3

    def save_task_config(self):
        """Save newly sampled task states to task config directory"""
        if not os.path.exists(self.paths.this_config):
            os.makedirs(self.paths.this_config)
        else:
            print("A task configuration with this name already exists. \n"
                  "Program will be terminated.")
            exit()  # TODO: why sys.exit() better?

        for key, value in self.task_config.items():
            np.save(os.path.join(self.paths.this_config, f'{key}.npy'), value)

        config_df_fn = os.path.join(self.paths.this_config,
                                    'config_params.tsv')
        all_block_df = pd.DataFrame()
        for block_ in range(self.task_design_params.n_blocks):  # TODO:eleganter
            this_block_df = pd.DataFrame(
                index=range(0, self.task_design_params.n_rounds))
            this_block_df['block'] = block_ + 1
            this_block_df['round'] = range(1,
                                           self.task_design_params.n_rounds
                                           + 1)
            this_block_df['hides'] = np.full(
                self.task_design_params.n_rounds, np.nan)
            this_block_df['hides'] = this_block_df[
                'hides'].astype('object')
            for round_ in range(self.task_design_params.n_rounds):
                this_block_df.at[
                    round_, 'hides'] = self.task_config['hides_loc'][block_]
            this_block_df['s1'] = self.task_config['s_1'][block_]
            this_block_df['s3'] = self.task_config['s_3'][block_]

            all_block_df = pd.concat([all_block_df,
                                      this_block_df],
                                     ignore_index=True)
        with open(config_df_fn, 'w', encoding="utf-8") as tsv_file:
            tsv_file.write(all_block_df.to_csv(sep='\t', index=False))

    def sample_task_config(self):
        """Sample all task states s1, s3 and s4 for all trials/rounds
        and return dict with states"""
        self.sample_hiding_spots()
        self.sample_start_pos()
        self.sample_treasure_loc()
        self.save_task_config()

    def load_task_config(self):
        """Load existing task configuration files from task config directory"""
        for item in ['s_1', 's_3', 'hides']:
            self.task_config[item] = np.load(
                os.path.join(self.paths.this_config, f'{item}.npy'))

    def prepare_task_config(self):
        """Prepare task configuration according to user input"""
        self.get_user_input()
        self.add_config_paths()
        if self.new_config_needed:
            self.sample_task_config()
        else:
            self.load_task_config()

    def return_task_configuration(self):
        """Return task configuration"""
        return self.task_config
