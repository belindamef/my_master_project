"""This module contains functions to implement basic configurations
that are shared across scripts."""

from dataclasses import dataclass
import os
import argparse
import copy as cp
import glob
import numpy as np
import pandas as pd


def get_user_yes_no(question: str) -> bool:
    """Function to get user input

    Args:
        question (str): Question to be printed to user

    Returns:
        bool: user answer
    """
    reply = input(question + " (Y/N): ").lower().strip()
    try:
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False
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
    val_results = os.path.join(results, "validation")
    model_fit_results = os.path.join(results, "model_fit")
    this_config: str = "not defined"  # particular config currently used

    # Raw behavioral data or estimation results directories
    this_exp_rawdata_dir: str = "not defined"
    this_sim_rawdata_dir: str = "not defined"
    this_val_results_dir: str = "not defined"
    this_model_fit_results_dir: str = "not defined"

    # Subject-specific directories and filenames
    this_sub_dir: str = "not defined"
    this_sub_beh_out_filename: str = "not defined"
    this_sub_val_result_fn: str = "not defined"
    this_sub_model_fit_results_fn: str = "not defined"

    # Processed data and descriptive stats directories
    this_analyses_raw_data_path: str = "not defined"
    this_analyses_proc_data_path: str = "not defined"
    this_analyses_descr_stats_path: str = "not defined"

    # Filenames
    part_fn: str = "not defined"
    events_all_subs_fn: str = "not defined"
    subj_lvl_descr_stats_fn: str = "not defined"
    grp_lvl_descr_stats_fn: str = "not defined"
    t_wise_stats_fn: str = "not defined"
    r_wise_stats_fn: str = "not defined"


class DirectoryManager:
    """Class of methods to create or check for directories"""

    paths = Paths()
    sub_id: str

    def define_raw_beh_data_out_path(self, data_type: str,
                                     out_dir_label: str = "not defined",
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

        while out_dir_label == "not defined":
            out_dir_label = input(
                "Enter label for raw behav. data output directory: ")
            if os.path.exists(os.path.join(data_directory, out_dir_label)):
                print("A directory with this name already exists. "
                      "Please choose a different name.")
                out_dir_label = "not defined"

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
        """Function to define filenames that store descriptive statistics
        """
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

    def define_val_results_path(self, dir_label: str = "not_given",
                                version="1", make_dir: bool = False):
        """Method to define the path variable for the directory containing
        validation results.

        Args:
            dir_label (str, optional): Name for this validation run.
                Defaults to "not_given".
            version (str, optional): Version for a given validation run.
                Two runs with the same label can be different versions.
                Defaults to "1".
            make_dir (bool, optional): If True, makes the directory defined
                in this method. Defaults to False.
        """
        if dir_label == "not_given":
            while True:
                try:
                    dir_name = input(
                        "Enter label for validation output directory: ")
                    self.paths.this_val_results_dir = os.path.join(
                        self.paths.sim_rawdata, dir_name)
                    os.makedirs(self.paths.this_val_results_dir)
                    break
                except FileExistsError:
                    print('Validation output directory with this name already '
                          'exists.')
        else:
            self.paths.this_val_results_dir = os.path.join(
                self.paths.val_results, f"{dir_label}_{version}")

        if make_dir:
            try:
                if not os.path.exists(self.paths.this_val_results_dir):
                    os.makedirs(self.paths.this_val_results_dir)
                else:
                    print("Output directory for validation results already "
                          "exists. Skipping makedirs. results will be written "
                          "to existing directory.")
            except FileExistsError:
                print("Output directory for validation results already exists."
                      " Skipping makedirs. results will be written to "
                      "existing directory.")

    def define_model_fitting_results_path(self, dir_label: str, version="main",
                                          make_dir: bool = False):
        """Define path variable for directory containing model fitting results

        Parameters
        ----------
        dir_label: str
          "sim" or "exp"
        version: int or str
            Version number, only needed during debugging
        make_dir: bool
          if True, creates physical directory
          directory"""
        self.paths.this_model_fit_results_dir = os.path.join(
            self.paths.model_fit_results, f"{dir_label}_{version}")

        if make_dir:
            try:
                if not os.path.exists(self.paths.this_model_fit_results_dir):
                    os.makedirs(self.paths.this_model_fit_results_dir)
                else:
                    print("Output directory for model fitting results already "
                          "exists. Skipping makedirs. results will be written "
                          "to existing directory.")
            except FileExistsError:
                print("Output directory for model fitting results already "
                      "exists. Skipping makedirs. results will be written to "
                      "existing directory.")

    def define_and_make_sub_beh_out_dir(self, sub_id: str):
        """Define paths to subject specific output directory and make
        directory if not existent"""

        self.paths.this_sub_dir = os.path.join(
            self.paths.this_sim_rawdata_dir, f"sub-{sub_id}", "beh")
        if not os.path.exists(self.paths.this_sub_dir):
            os.makedirs(self.paths.this_sub_dir)

    def define_beh_out_filename(self, sub_id):
        """Method to define the filename for this subjects behavioral data."""
        self.paths.this_sub_beh_out_filename = os.path.join(
            self.paths.this_sub_dir,
            f"sub-{sub_id}_task-th_beh")

    def define_val_results_filename(self, sub_id: str):
        """Method to define the filename for this subjects validation restuls.
        """
        self.paths.this_sub_val_result_fn = os.path.join(
            self.paths.this_val_results_dir,
            f"val_results_sub-{sub_id}")

    def define_model_fit_results_filename(self, sub_id: str):
        """Method to define the filename for the model estimation results for
        the data of subject <sub_id>

        Args:
            sub_id (str): Subject ID
        """
        self.paths.this_sub_model_fit_results_fn = os.path.join(
            self.paths.this_model_fit_results_dir,
            f"model_fit_results_sub-{sub_id}"
        )

    def define_sim_beh_output_paths(self, sub_id: str):
        """Method to define output paths and filename variables

        Args:
            sub_id (str): Subject ID
        """
        self.define_and_make_sub_beh_out_dir(sub_id)
        self.define_beh_out_filename(sub_id)

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


class DataLoader:
    """Class to load data or descriptive stats"""
    def __init__(self, paths: Paths, exp_label: str):
        self.paths = paths

        exp_proc_data_dir = os.path.join(
            paths.data, 'processed_data', 'exp', f'{exp_label}')
        sim_proc_data_dir = os.path.join(
            paths.data, 'processed_data', 'sim', f'sim_{exp_label}')

        self.events_exp_fn = os.path.join(exp_proc_data_dir,
                                          'sub-all_task-th_run-all_beh')
        self.ev_sim_run_fn = os.path.join(sim_proc_data_dir,
                                          'sub-all_task-th_run-')

        self.ds_exp_fn = os.path.join(paths.descr_stats, 'exp',
                                      f'{exp_label}', 'descr_stats')
        self.grp_stats_exp_fn = os.path.join(
            paths.descr_stats, 'exp', f'{exp_label}', 'grp_lvl_stats')
        self.grp_stats_sim_fn = os.path.join(
            paths.descr_stats, 'sim', f'{exp_label}', 'grp_lvl_stats')
        self.grp_stats_sim_100_fn = os.path.join(
            paths.descr_stats, 'sim', 'sim_100_msc', 'grp_lvl_stats')
        self.tw_exp_fn = os.path.join(
            paths.descr_stats, 'exp', f'{exp_label}', 't_wise_stats')
        self.tw_sim_100_fn = os.path.join(
            paths.descr_stats, 'sim', 'sim_100_msc', 't_wise_stats')

    def load_sim_subj_lvl_stats(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        subj_lvl_stats_df = pd.read_pickle(
            f"{self.paths.subj_lvl_descr_stats_fn}.pkl")
        return subj_lvl_stats_df

    def create_list_of_files_in_folder(self, folder_path: str) -> list:
        """_summary_

        Args:
            folder_path (str): _description_

        Returns:
            list: _description_
        """
        return glob.glob(os.path.join(folder_path, "*"))

    def load_data_in_one_folder(self, folder_path: str) -> pd.DataFrame:
        """_summary_

        Args:
            folder_path (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        file_list = self.create_list_of_files_in_folder(
            folder_path=folder_path)

        dataframe = pd.concat(
            (pd.read_csv(f, sep="\t") for f in file_list),
            ignore_index=True)
        return dataframe

    def load_exp_events(self) -> pd.DataFrame:
        """Method to load trialwise events

        Returns:
            pd.DataFrame: Dataframe containing events
        """
        return pd.read_pickle(f'{self.events_exp_fn}.pkl')

    def load_sim100_group_lvl_stats(self) -> pd.DataFrame:
        """Method to load stats

        Returns:
            pd.DataFrame: dataframe with group level stats
        """
        return pd.read_pickle(f'{self.grp_stats_sim_100_fn}.pkl')

    def load_sim100_trialwise_stats(self) -> dict:
        """Method to load trialwise descriptive statistics of the simulation
        with 100 different task configurations.

        Returns:
            dict: Dictiorany containting stats in dataframes
        """
        tw_sim_100_aw = {}  # trial wise stats each agent over all blocks
        for agent in ['A1', 'A2', 'A3']:
            tw_sim_100_aw[agent] = pd.read_pickle(
                f'{self.tw_sim_100_fn}_agent-Agent {agent}.pkl')
        return tw_sim_100_aw


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
    n_blocks: int = 1
    n_rounds: int = 10
    n_trials: int = 12
    dim: int = 5
    n_hides: int = 6
    n_nodes: int = dim ** 2


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
