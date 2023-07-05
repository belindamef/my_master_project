"""This module contains functions to implement basic configurations
that are shared across scripts."""

from dataclasses import dataclass
import os
import argparse
import copy as cp
import numpy as np
import pandas as pd


def get_user_yes_no(question):
    reply = input(question + " (Y/N): ").lower().strip()
    try:
        if reply[:1] == 'y':
            return True
        elif reply[:1] == 'n':
            return False
        else:
            print('Invalid answer. Please answer with (Y/N). ')
            return get_user_yes_no(question)
    except TypeError as error:
        print("Please enter valid inputs")
        print(error)
        return get_user_yes_no(question)


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
    # General directories
    utils = os.path.dirname(os.path.abspath(__file__))
    code = os.path.dirname(utils)
    project = os.path.dirname(code)
    task_configs = os.path.join(code, "task_config")  # all configurations
    data = os.path.join(project, "data")
    figures = os.path.join(project, "figures")
    sim_rawdata = os.path.join(data, "rawdata", "sim")
    exp_rawdata = os.path.join(data, "rawdata", "exp")
    results = os.path.join(project, "results")
    descr_stats = os.path.join(results, 'descr_stats')
    val_out = os.path.join(results, "validation")
    this_config: str = None  # the particular config used in this simulation

    # Raw behavioral data or estimation results directories
    this_exp_rawdata_dir: str = None
    this_sim_rawdata_dir: str = None
    this_val_out_dir: str = None

    # Subject-specific directories
    this_sub_dir: str = None
    this_sub_beh_out_filename: str = None

    # Processed data and descriptive stats directories
    this_analyses_raw_data_path: str = None
    this_analyses_proc_data_path: str = None
    this_analyses_descr_stats_path: str = None

    # Filenames
    part_fn: str = None
    events_all_subs_fn: str = None
    subj_lvl_descr_stats_fn: str = None
    grp_lvl_descr_stats_fn: str = None
    t_wise_stats_fn: str = None
    r_wise_stats_fn: str = None


class DirectoryManager:
    """Class of methods to create or check for directories"""

    paths = Paths()
    sub_id: str

    def define_raw_beh_data_out_path(self, data_type: str,
                                     out_dir_label: str = None,
                                     make_dir: bool = False,):
        """
        Create path variable for output directoy containing behavioral data

        Parameters
        ----------
        data_type: str
          "sim" or "exp"
        make_dir: bool
          if True, creates physical directory
          directory
        """

        if data_type == "sim":
            data_directory = self.paths.sim_rawdata
        else:
            data_directory = self.paths.exp_rawdata

        while not out_dir_label:
            out_dir_label = input(
                "Enter label for raw behav. data output directory: ")
            if os.path.exists(os.path.join(data_directory, out_dir_label)):
                print("A directory with this name already exists. "
                      "Please choose a different name.")
                out_dir_label = None

        raw_data_path = os.path.join(data_directory, out_dir_label)

        try:
            if make_dir:
                if not os.path.exists(raw_data_path):
                    os.makedirs(raw_data_path)
                else:
                    print("Output directory for raw data already exists. "
                          "Skipping makedirs. Raw data will be written to "
                          "existing directory.")
        except FileExistsError:
            print("Output directory for raw data already exists. "
                  "Skipping makedirs. Raw data will be written to "
                  "existing directory.")

        if data_type == "sim":
            self.paths.this_analyses_raw_data_path = raw_data_path
            self.paths.this_sim_rawdata_dir = raw_data_path
        else:
            self.paths.this_analyses_raw_data_path = raw_data_path

    def define_processed_data_path(self, data_type: str,
                                   dir_label: str,
                                   make_dir: bool = False):
        """Define path variable for directory containing processed behavioral
        data

        Parameters
        ----------
        data_type: str
          "sim" or "exp"
        make_dir: bool
          if True, creates physical directory
          directory"""
        self.paths.this_analyses_proc_data_path = os.path.join(
            self.paths.data, 'processed_data', f'{data_type}', f'{dir_label}')
        if make_dir:
            if not os.path.exists(self.paths.this_analyses_proc_data_path):
                os.makedirs(self.paths.this_analyses_proc_data_path)

    def define_descr_stats_path(self, data_type: str,
                                dir_label: str,
                                make_dir: bool = False):
        """Define path variable for directory containing descriptive stats

        Parameters
        ----------
        data_type: str
          "sim" or "exp"
        make_dir: bool
          if True, creates physical directory
          directory"""
        self.paths.this_analyses_descr_stats_path = os.path.join(
            self.paths.descr_stats, f'{data_type}', f'{dir_label}')
        if make_dir:
            if not os.path.exists(self.paths.this_analyses_descr_stats_path):
                os.makedirs(self.paths.this_analyses_descr_stats_path)

    def define_stats_filenames(self):
        self.paths.part_fn = os.path.join(
            self.paths.this_analyses_raw_data_path, "participants.tsv"
            )
        self.paths.events_all_subs_fn = os.path.join(
            self.paths.this_analyses_proc_data_path,
            "sub-all_task-th_run-all_beh"
        )
        self.paths.subj_lvl_descr_stats_fn = os.path.join(
            self.paths.this_analyses_descr_stats_path, "descr_stats"
        )
        self.paths.grp_lvl_descr_stats_fn = os.path.join(
            self.paths.this_analyses_descr_stats_path, "grp_lvl_stats"
        )
        self.paths.t_wise_stats_fn = os.path.join(
            self.paths.this_analyses_descr_stats_path, "t_wise_stats"
        )
        self.paths.r_wise_stats_fn = os.path.join(
            self.paths.this_analyses_descr_stats_path, "r_wise_stats"
        )

    def create_val_out_dir(self, out_dir_label=None, version=1):
        if not out_dir_label:
            while True:
                try:
                    dir_name = input(
                        "Enter label for validation output directory: ")
                    self.paths.this_val_out_dir = os.path.join(
                        self.paths.sim_rawdata, dir_name)
                    os.makedirs(self.paths.this_val_out_dir)
                    break
                except FileExistsError:
                    print('Validation output directory with this name already '
                          'exists.')
        else:
            self.paths.this_val_out_dir = os.path.join(
                self.paths.val_out, f"{out_dir_label}_{version}")
            if not os.path.exists(self.paths.this_val_out_dir):
                os.makedirs(self.paths.this_val_out_dir)

    def create_agent_sub_id(self, sim_params):
        """Create id for this subject. More than one subject id per agent
        possible if >1 repetition per agent

        Parameters
        ----------
        sim_obj: Simulator
        """
        self.sub_id = (
            f"{sim_params.current_agent_attributes.name}_" +
            f"rep-{sim_params.current_rep}_" +
            "tau-" + f"{sim_params.current_tau_gen * 1000}"[:4] + "_" +
            "lambda-" + f"{sim_params.current_lambda_gen * 1000}"[:4] + "_" +
            f"part-{sim_params.current_part}"
            ).replace(".", "")

        self.sub_id.replace(".", "")

    def define_and_make_sub_beh_out_dir(self):
        """Define paths to subject specific output directory and make
        directory if not existent"""
        self.paths.this_sub_dir = os.path.join(
            self.paths.this_sim_rawdata_dir, f"sub-{self.sub_id}", "beh")
        if not os.path.exists(self.paths.this_sub_dir):
            os.makedirs(self.paths.this_sub_dir)

    def define_beh_out_filename(self):
        self.paths.this_sub_beh_out_filename = os.path.join(
            self.paths.this_sub_dir,
            f"sub-{self.sub_id}_task-th_beh")

    def define_out_single_val_filename(self, rep, agent, tau, lambda_, part):
        if np.isnan(lambda_):
            filename = os.path.join(
                self.paths.this_val_out_dir,
                f"rep-{rep}_agent-{agent}_"
                f"tau-{int(round(tau * 1000, ndigits=4))}_"
                f"part-{part}")
        else:
            filename = os.path.join(
                self.paths.this_val_out_dir,
                f"rep-{rep}_agent-{agent}_"
                f"tau-{int(round(tau * 1000, ndigits=4))}_"
                f"lambda-{int(round(lambda_ * 1000, ndigits=4))}_"
                f"part-{part}")
        return filename

    def prepare_beh_output(self, sim_params):
        self.create_agent_sub_id(sim_params)
        self.define_and_make_sub_beh_out_dir()
        self.define_beh_out_filename()

    def save_data_to_tsv(self, data):
        """Safe dataframe to a .tsv file

        Parameters
        ----------
        data: pd.Dataframe
            dataframe containting simulated behavioral data
        """
        with open(f"{self.paths.this_sub_beh_out_filename}.tsv", "w",
                  encoding="utf8") as tsv_file:
            tsv_file.write(data.to_csv(sep="\t", na_rep=np.NaN, index=False))


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

    Attributes
    ----------
    params: obj
        Object of class TaskDesignParams
    states : dict  # TODO
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
    states = {}
    params = TaskDesignParameters()

    def __init__(self, path):
        """

        Parameters
        ----------
        path: Paths
        """
        self.paths = path

    def get_user_input(self):
        """Get user input for simulation task configuration
        """
        n_blocks = "as in loaded configuration"
        new_config_needed = get_user_yes_no("Create new task configuration?")
        if new_config_needed:
            while True:
                config_label = input("Enter label for new task "
                                     "configuration: ")
                if os.path.exists(os.path.join(
                        self.paths.task_configs, config_label)):
                    print("A task configuration with this name already exists."
                          "\nEnter another name. ")
                else:
                    break
            n_blocks = int(input("Enter number of blocks: "))
        else:
            while True:
                config_label = input("Enter label of existing task config ("
                                     "'exp_msc'/'sim_100_msc'): ")
                if not os.path.exists(os.path.join(
                        self.paths.task_configs, config_label)):
                    print(f"No configuration named '{config_label}' "
                          f"found.")
                else:
                    break
        return new_config_needed, config_label, n_blocks

    def add_config_paths(self, config_label):
        """Add path to this task configurations config files dir to path obj"""
        self.paths.this_config = os.path.join(
            self.paths.task_configs, config_label)

    def sample_hiding_spots(self):
        """Sample hiding spots from a discrete uniform distribution over
         all nodes (without replacement)"""
        hides_loc = np.empty((self.params.n_blocks,
                             self.params.n_hides), dtype=int)
        for block in range(self.params.n_blocks):
            hides_loc[block] = np.random.choice(
                self.params.n_nodes,
                self.params.n_hides,
                replace=False)
        self.states['hides'] = hides_loc

    def sample_start_pos(self):
        """Sample the starting position from a discrete uniform distribution
        over all nodes"""
        s_1 = np.full((self.params.n_blocks,
                       self.params.n_rounds), np.nan)
        for block in range(self.params.n_blocks):
            for round_ in range(self.params.n_rounds):
                s_1[block, round_] = np.random.choice(
                    self.params.n_nodes, 1)
        self.states['s_1'] = s_1

    def sample_treasure_loc(self):
        """Sample the tr location from a discrete uniform distribution over all
        hiding spots"""
        s_3 = np.full((self.params.n_blocks,
                       self.params.n_rounds), np.nan)
        for block in range(self.params.n_blocks):
            for round_ in range(self.params.n_rounds):
                # Set treasure to equal start position
                s_3[block, round_] = cp.deepcopy(
                    self.states['s_1'][block, round_])
                # Sample tr location until it's not the starting position s_0
                while s_3[block, round_] == self.states['s_1'][block, round_]:
                    s_3[block, round_] = np.random.choice(
                        self.states['hides'][block], 1)
        self.states['s_3'] = s_3

    def save_task_config(self):
        """Save newly sampled task states to task config directory"""
        os.makedirs(self.paths.this_config)
        for key, value in self.states.items():
            np.save(os.path.join(self.paths.this_config, f'{key}.npy'), value)

        config_df_fn = os.path.join(self.paths.this_config,
                                    'config_params.tsv')
        all_block_df = pd.DataFrame()
        for block_ in range(self.params.n_blocks):
            this_block_df = pd.DataFrame(
                index=range(0, self.params.n_rounds))
            this_block_df['block'] = block_ + 1
            this_block_df['round'] = range(1,
                                           self.params.n_rounds
                                           + 1)
            this_block_df['hides'] = np.full(
                self.params.n_rounds, np.nan)
            this_block_df['hides'] = this_block_df[
                'hides'].astype('object')
            for round_ in range(self.params.n_rounds):
                this_block_df.at[
                    round_, 'hides'] = self.states['hides'][block_]
            this_block_df['s1'] = self.states['s_1'][block_]
            this_block_df['s3'] = self.states['s_3'][block_]

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
            self.states[item] = np.load(
                os.path.join(self.paths.this_config, f'{item}.npy'))

    def get_config(self, config_label: str):
        """Create or load task configuration according to user input"""

        new_config_is_needed = False
        n_blocks = 3
        self.add_config_paths(config_label)
        if new_config_is_needed:
            self.params.n_blocks = n_blocks
            self.sample_task_config()
        else:
            self.load_task_config()
            self.params.n_blocks = list(
                self.states.values())[0].shape[0]

        return self


def get_arguments():
    """Get arguments from environment, if script is executed from command line
    or with a bash jobwrapper."""
    parser = argparse.ArgumentParser(description='Run model validation.')
    parser.add_argument('--parallel_computing', action="store_true")
    parser.add_argument('--repetition', type=int, nargs='+')
    parser.add_argument('--agent_model', type=str, nargs='+')
    parser.add_argument('--tau_value', type=float, nargs='+')
    parser.add_argument('--lambda_value', type=float, nargs='+')
    parser.add_argument('--participant', type=int, nargs='+')
    args = parser.parse_args()
    return args
