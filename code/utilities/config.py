"""This module contains functions to implement basic configurations
that are shared across scripts."""

import os
import argparse
import copy as cp
import glob
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import pickle
import csv

from pygame import init

def humanreadable_time(time_in_seconds: float) -> str:
    """_summary_

    Args:
        time_in_seconds (float): _description_

    Returns:
        str: _description_
    """
    if time_in_seconds < 0.01:
        hr_time = "< 0.01 sec."
    # elif time_in_seconds < 1:
    #     hr_time = f"{round(time_in_seconds, 2)} sec."
    elif time_in_seconds < 60:
        hr_time = f"{round(time_in_seconds, 2)} sec."
    else:
        hr_time = (f"{int(round(time_in_seconds/60, 0))}:" +
                   f"{int(round(time_in_seconds%60, 0)):02d} min.")
    return hr_time


def get_user_yes_no(question: str) -> bool:
    """Function to get user input

    Args:
    ----
        question (str): Question to be printed to user

    Returns:
    -------
        bool: True, for yes and False for no from user response
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


def custom_sort_key(item):
    """Method to sort strings in customized order, i.e. letter "C" before "A".

    Args:
    -------
        item (TODO): TODO

    Returns:
    -------
        TODO: TODO
    """
    letter, number = item[0], item[1:]
    if letter == "C":
        return (0, number)  # Sort "C" items first, then by number
    elif letter == "A":
        return (2, number)  # Sort "A" items last, then by number
    else:
        return (1, item)  # For other letters, maintain original order


@dataclass
class Paths:
    """A data class to store directory and file paths as string values"""

    # General directories
    utils = os.path.dirname(os.path.abspath(__file__))
    code = os.path.dirname(utils)
    project = os.path.dirname(code)
    task_configs = os.path.join(code, "task_config")  # all configurations
    stoch_mats = os.path.join(code, "stoch_matrices")  # HMM stoch. matrices
    data = os.path.join(project, "data")
    figures = os.path.join(project, "figures")
    sim_rawdata = os.path.join(data, "rawdata", "sim")
    exp_rawdata = os.path.join(data, "rawdata", "exp")
    results = os.path.join(project, "results")
    descr_stats = os.path.join(results, 'descr_stats')
    model_recov_results = os.path.join(results, "model_recovery")
    model_est_results = os.path.join(results, "model_estimation")
    this_config: str = ""  # particular config currently used

    # Raw behavioral data or validation results directories
    this_exp_rawdata: str = ""
    this_sim_rawdata: str = ""
    this_model_recov_results: str = ""
    this_model_recov_sub_lvl_results: str = ""
    this_model_est_results: str = ""
    this_model_est_sub_lvl_results: str = ""

    # Subject-specific directories and filenames (fn)
    this_sub: str = ""
    this_sub_beh_out_filename: str = ""
    this_sub_model_recov_result_fn: str = ""
    this_sub_model_est_results_fn: str = ""

    # Processed data and descriptive stats directories
    this_analyses_raw_data_path: str = ""
    this_analyses_proc_data_path: str = ""
    this_analyses_descr_stats_path: str = ""

    # Filenames
    part_fn: str = ""
    events_all_subs_fn: str = ""
    subj_lvl_descr_stats_fn: str = ""
    grp_lvl_descr_stats_fn: str = ""
    t_wise_stats_fn: str = ""
    r_wise_stats_fn: str = ""
    grp_lvl_model_recovery_results_fn: str = ""
    grp_lvl_model_estimation_results_fn: str = ""


class DirectoryManager:
    """Class of methods to create path variables and manage directories."""

    paths = Paths()

    def define_raw_beh_data_out_path(self, data_type: str,
                                     exp_label: str,
                                     version: str = "",
                                     make_dir: bool = False,):
        """Method to create path variable for the directoy containing
        behavioral data

        Args:
        ------
            data_type (str): "sim" for simulated dataset or "exp" for
                experimental dataset
            exp_label (str): Name of experiment, i.e. task configuration
            version (str, optional): Version label. Defaults to "".
            make (bool, optional): If true, creates directory data folder.
                Defaults to False.
        """

        if data_type == "sim":
            dataectory = self.paths.sim_rawdata
        else:
            dataectory = self.paths.exp_rawdata

        while exp_label == "":
            exp_label = input(
                "Enter label for raw behav. data output directory: ")
            if os.path.exists(os.path.join(dataectory, exp_label)):
                print("A directory with this name already exists. "
                      "Please choose a different name.")
                exp_label = ""

        raw_data_path = os.path.join(dataectory, f"{exp_label}_{version}")

        try:
            if make_dir:
                if not os.path.exists(raw_data_path):
                    os.makedirs(raw_data_path)
                else:
                    print("Output directory for raw data already exists. "
                          "Skipping makedirs. Raw data will be written to "
                          f"{raw_data_path}. \n")
        except FileExistsError:
            print("Output directory for raw data already exists. "
                  "Skipping makedirs. Raw data will be written to "
                  f"{raw_data_path}. \n")

        if data_type == "sim":
            self.paths.this_analyses_raw_data_path = raw_data_path
            self.paths.this_sim_rawdata = raw_data_path
        else:
            self.paths.this_analyses_raw_data_path = raw_data_path

    def define_processed_data_path(self, data_type: str,
                                   exp_label: str,
                                   vers: str = "",
                                   make_dir: bool = False):
        """Define path variable for directory containing processed behavioral
        data

        Args:
        -----
            data_type (str): "sim" for simulated dataset or "exp" for
                experimental dataset
            exp_label (str): Name of experiment, i.e. task configuration
                version (str, optional): Version label. Defaults to "".
            vers (str): Version label. Defaults to "".
            make (bool, optional): If true, creates directory data folder.
                Defaults to False.. Defaults to False.
        """

        self.paths.this_analyses_proc_data_path = os.path.join(
            self.paths.data, 'processed_data', f'{data_type}',
            f'{exp_label}_{vers}')
        if make_dir:
            if not os.path.exists(self.paths.this_analyses_proc_data_path):
                os.makedirs(self.paths.this_analyses_proc_data_path)

    def define_descr_stats_path(self, data_type: str,
                                exp_label: str,
                                version: str = "",
                                make_dir: bool = False):
        """Method to define path variable for directory containing
        descriptive stats

        Args:
        -----
            data_type (str): "sim" for simulated dataset or "exp" for
                experimental dataset
            exp_label (str): Name of experiment, i.e. task configuration
                version (str, optional): Version label. Defaults to "".
            vers (str): Version label. Defaults to "".
            make (bool, optional): If true, creates directory data folder.
                Defaults to False.. Defaults to False.
        """
        self.paths.this_analyses_descr_stats_path = os.path.join(
            self.paths.descr_stats, f'{data_type}',
            f'{exp_label}_{version}')
        if make_dir:
            if not os.path.exists(self.paths.this_analyses_descr_stats_path):
                os.makedirs(self.paths.this_analyses_descr_stats_path)

    def define_stats_filenames(self):
        """Method to define filenames that store descriptive statistics
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

    def define_model_recov_results_path(self, exp_label: str = "",
                                        version="", make_dir: bool = False):
        """Method to define the path variable for the directory containing
        model recovery results.

        Args:
        -----
            exp_label (str): Name of experiment, i.e. task configuration
                version (str, optional): Version label. Defaults to "".
            vers (str): Version label. Defaults to "".
            make (bool, optional): If true, creates directory data folder.
                Defaults to False.. Defaults to False.
        """

        if not exp_label:
            while True:
                try:
                    dir_name = input(
                        "Enter label for model recovery output directory: ")
                    self.paths.this_model_recov_results = os.path.join(
                        self.paths.sim_rawdata, dir_name)
                    os.makedirs(self.paths.this_model_recov_results)
                    break
                except FileExistsError:
                    print('model recovery output directory with this name '
                          'already exists.')
        else:
            self.paths.this_model_recov_results = os.path.join(
                self.paths.model_recov_results, f"{exp_label}_{version}")

            self.paths.this_model_recov_sub_lvl_results = os.path.join(
                self.paths.this_model_recov_results, "sub_lvl")

        if make_dir:
            try:
                if not os.path.exists(self.paths.this_model_recov_results):
                    os.makedirs(self.paths.this_model_recov_results)
                else:
                    print("\n Output directory for this exp_label's validation"
                          " results already exists. Skipping makedirs. Results"
                          " will be written to "
                          f"{self.paths.this_model_recov_results}. \n")
                if not os.path.exists(
                        self.paths.this_model_recov_sub_lvl_results):
                    os.makedirs(
                        self.paths.this_model_recov_sub_lvl_results)
                else:
                    print("Output directory for subject level validation "
                          "results already exist. Skipping makedirs\n")
            except FileExistsError:
                print("Output directory for validation results already exists."
                      " Skipping makedirs.\n")

    def define_model_est_results_path(self, exp_label: str, version="main",
                                      make_dir: bool = False):
        """Method to define path variable for directory containing model
         estimation (i.e. model validation with experimental data) results

        Args:
        ------
            exp_label (str): Name of experiment, i.e. task configuration
                version (str, optional): Version label. Defaults to "".
            vers (str): Version label. Defaults to "".
            make (bool, optional): If true, creates directory data folder.
                Defaults to False.. Defaults to False."""
        self.paths.this_model_est_results = os.path.join(
            self.paths.model_est_results, f"{exp_label}_{version}")

        self.paths.this_model_est_sub_lvl_results = os.path.join(
            self.paths.this_model_est_results, "sub_lvl")

        if make_dir:
            try:
                if not os.path.exists(self.paths.this_model_est_results):
                    os.makedirs(self.paths.this_model_est_results)
                else:
                    print("Output directory for model estimation results "
                          "already exists. Skipping makedirs. Results will be "
                          f"written to {self.paths.this_model_est_results}.\n")

                if not os.path.exists(
                        self.paths.this_model_est_sub_lvl_results):
                    os.makedirs(
                        self.paths.this_model_est_sub_lvl_results)
                else:
                    print("Output directory for subject level model estimation"
                          " results already exists. Skipping makedirs.\n")

            except FileExistsError:
                print("Output directory for model estimation results already "
                      "exists. Skipping makedirs.\n")

    def define_and_make_sim_beh_out(self, sub_id: str):
        """Method to define paths to subject-specific output directory and make
        directory if not existent

        Args:
        -----
            sub_id (str): subject ID
        """
        self.paths.this_sub = os.path.join(
            self.paths.this_sim_rawdata, f"sub-{sub_id}", "beh")
        if not os.path.exists(self.paths.this_sub):
            os.makedirs(self.paths.this_sub)
        # TODO: generalize for experimental behavioral data

    def define_beh_out_fn(self, sub_id):
        """Method to define the filename for this subject's behavioral data.

        Args:
        -----
            sub_id (str): subject ID
        """
        self.paths.this_sub_beh_out_filename = os.path.join(
            self.paths.this_sub,
            f"sub-{sub_id}_task-th_beh")

    def define_sub_lvl_model_recov_results_fn(self, sub_id: str):
        """Method to define the filename for this subjects validation results.

        Args:
        -----
            sub_id (str): subject ID
        """
        self.paths.this_sub_model_recov_result_fn = os.path.join(
            self.paths.this_model_recov_sub_lvl_results,
            f"val_results_sub-{sub_id}")

    def define_grp_lvl_model_validation_results_fn_s(self):
        """Method to define the filename for group level model validation
        performance results, for both simulated and experimantal data."""

        self.paths.grp_lvl_model_recovery_results_fn = os.path.join(
            self.paths.this_model_recov_results,
            "grp_lvl_val_results")
        self.paths.grp_lvl_model_estimation_results_fn = os.path.join(
            self.paths.this_model_est_results,
            "grp_lvl_val_results"
        )

    def define_model_est_results_filename(self, sub_id: str):
        """Method to define the filename for the model estimation results for
        the experimental data of subject <sub_id>

        Args:
        -----
            sub_id (str): Subject ID
        """
        self.paths.this_sub_model_est_results_fn = os.path.join(
            self.paths.this_model_est_sub_lvl_results,
            f"val_results_sub-{sub_id}"
        )

    def define_sim_beh_output_paths(self, sub_id: str):
        """Method to define output paths and filename variables

        Args:
        -----
            sub_id (str): Subject ID
        """
        self.define_and_make_sim_beh_out(sub_id)
        self.define_beh_out_fn(sub_id)


class DataHandler:
    """Class to load/save data or results from/to disk

    Args
    ----------
        paths (Paths): Paths class instance
        exp_label (str): Name of experiment, i.e. task configuration

    Attributes
    ----------
        paths (Paths): Paths class instance
        exp_label (str): Name of experiment, i.e. task configuration
    """

    def __init__(self, paths: Paths, exp_label: str = ""):

        self.paths = paths
        self.exp_label = exp_label

        sim_proc_data = os.path.join(
            paths.data, 'processed_data', 'sim', f'sim_{exp_label}')

        # TODO: clean up class attributes to locas private variables
        self.ev_sim_run_fn = os.path.join(sim_proc_data,
                                          'sub-all_task-th_run-')
        self.ds_exp_fn = os.path.join(paths.descr_stats, 'exp',
                                      f'{exp_label}', 'descr_stats')
        self.grp_stats_exp_fn = os.path.join(
            paths.descr_stats, 'exp', f'{exp_label}', 'grp_lvl_stats')
        self.grp_stats_sim_fn = os.path.join(
            paths.descr_stats, 'sim', f'{exp_label}', 'grp_lvl_stats')
        self.tw_exp_fn = os.path.join(
            paths.descr_stats, 'exp', f'{exp_label}', 't_wise_stats')

    def create_matrix_fn(self, matrix_name: str,
                         n_nodes: int, n_hides: int) -> str:
        return os.path.join(
            self.paths.stoch_mats,
            f"{matrix_name}_{n_nodes}-nodes_{n_hides}-hides")

    def save_arrays(self, n_nodes: int, n_hides: int, **arrays):

        if not os.path.exists(self.paths.stoch_mats):
            os.makedirs(self.paths.stoch_mats)

        for key, array in arrays.items():

            # Define the output file name
            out_fn = self.create_matrix_fn(matrix_name=key,
                                           n_nodes=n_nodes,
                                           n_hides=n_hides)

            # Write the vectors to the TSV file
            with open(f"{out_fn}.csv", 'w', newline='', encoding="utf8") as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerows(array)

            with open(f"{out_fn}.pkl", "wb") as f:
                pickle.dump(array, f)

    def save_data_to_tsv(self, data: pd.DataFrame, filename: str):
        """Safe dataframe to a .tsv file

        Args:
        ----
            data (pd.DataFrame): (n_participants x n_variables)-Dataframe
                containing simulated behavioral data
            filename (str): Path to file on disk
        """
        with open(f"{filename}.tsv", "w", encoding="utf8") as tsv_file:
            tsv_file.write(data.to_csv(sep="\t", na_rep="nan", index=False))

    def load_sim_subj_lvl_stats(self) -> pd.DataFrame:
        """Load descriptive statstics from disk

        Returns:
        ------
            pd.DataFrame: (n_participants x n_measures)-Dataframe containing
              subject-level descriptive statistics
        """
        subj_lvl_stats_df = pd.read_pickle(
            f"{self.paths.subj_lvl_descr_stats_fn}.pkl")
        return subj_lvl_stats_df

    def create_list_of_files_in_folder(self, folder_path: str) -> list:
        """Create list of all files contained in given folder

        Args:
        -----
            folder_path (str): Path to folder on disk

        Returns:
        ------
            list: List of all filenames (full path to files)
        """
        return glob.glob(os.path.join(folder_path, "*"))

    def load_data_single_tsv(self, file_path: str) -> pd.DataFrame:
        """Load data from singe tsv file

        Args:
        -----
            folder_path (str): Path to file on disk

        Returns:
        -------
            pd.DataFrame: Data
        """
        return pd.read_csv(file_path, sep="\t", encoding="utf8")

    def load_data_in_one_folder(self, folder_path: str) -> pd.DataFrame:
        """Load data from all tsv files in one folder. All rowns from tsv files
        are concatenated to one dataframe.

        Note:
        -----
        Only works if all tsv files have same column labebels

        Args:
        -----
            folder_path (str): Path to folder on disk

        Returns:
        --------
            pd.DataFrame: Concatenated dataframe
        """
        file_list = self.create_list_of_files_in_folder(
            folder_path=folder_path)

        dataframe = pd.concat(
            ([pd.read_csv(f, sep="\t", encoding="utf8") for f in file_list]),
            ignore_index=True)
        return dataframe

    def load_proc_exp_events(self) -> pd.DataFrame:
        """Method to load dataframe containing events from all participants
        (processed data version) from disk.

        Returns:
        --------
            pd.DataFrame: ((n_participants * n_events) x n_variables)-Dataframe
                containing events
        """
        events_exp_fn = os.path.join(
            self.paths.data, 'processed_data', 'exp', f'{self.exp_label}',
            'sub-all_task-th_run-all_beh')
        return pd.read_pickle(f'{events_exp_fn}.pkl')

    def load_sim100_group_lvl_stats(self) -> pd.DataFrame:
        """Method to load group level descriptive statistics of the simulation
        with 100 different task configurations from disk

        Returns:
        --------
            pd.DataFrame: (n_agents x n_measures)-dataframe containing group
                level stats
        """
        grp_stats_sim_100_fn = os.path.join(
            self.paths.descr_stats, 'sim', 'sim_100_msc', 'grp_lvl_stats')
        return pd.read_pickle(f'{grp_stats_sim_100_fn}.pkl')

    def load_sim100_trialwise_stats(self) -> dict:
        """Method to load trialwise descriptive statistics of the simulation
        with 100 different task configurations from disk.

        Returns:
        -------
            dict: Dictiorany of dataframes. One entry per agent
        """
        tw_sim_100_fn = os.path.join(
            self.paths.descr_stats, 'sim', 'sim_100_msc', 't_wise_stats')
        tw_sim_100_aw = {}  # trial wise stats each agent over all blocks
        for agent in ['A1', 'A2', 'A3']:
            tw_sim_100_aw[agent] = pd.read_pickle(
                f'{tw_sim_100_fn}_agent-Agent {agent}.pkl')
        return tw_sim_100_aw


@dataclass
class GridConfigurationParameters:
    """A data class to store experimental parameters

    Attributes
    ----------
    n_blocks (int): number of blocks in one run.
    n_rounds (int): number of rounds in one block
    n_trials (int): number of trials in one run
    dim (int): size (i.e. no. of both, rows and columns) in the 2-dimensional
        grid
    n_hides (int): number of hiding spots in the grid world
    n_nodes (int): number of fields in the grid world
    """
    n_blocks: int = 1
    n_rounds: int = 10
    n_trials: int = 12
    dim: int = 5
    n_hides: int = 6
    n_nodes: int = field(init=False)

    def __post_init__(self):
        self.n_nodes = self.dim ** 2

class TaskConfigurator:
    """A Class to create task configurations given a set of
    experimental parameters (i.e. no. trials, dimension of the grid, etc.) or
    load existend task configurations from disk given the label of a task
    configuration.

    Configuration-specific task state values are stored in the instance
    attribute "states (dict)".
    Newly sampled task configuration are written to .npy files are written and
    save in config directory on disk.

    Args:
    -----
        path (Path): Instance of class Paths

    Attributes:
    ----------
        params (TaskDesignParameters): Instance of class TaskDesignParams
        states (dict of str: np.ndarray): Configuration-specific state values
            "s_1" : (n_blocks)x(n_rounds)-array of values indicating starting
                node positions of each round
            "s_3": (n_blocks)x(n_rounds)-array of values indicating treasure
                locations
            "hides_loc": (n_blocks)x(n_hides)-array of values indicating hiding
                spot locations
    """

    def __init__(self, path: Paths, dim: int = 5, n_hiding_spots: int = 6,
                 params: GridConfigurationParameters = GridConfigurationParameters()):
        self.paths = path
        self.states = {}
        self.params = params

    def get_user_input(self):
        """Method to get user input to create new task configurations"""

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

    def add_config_paths(self, config_label: str):
        """Add path to this task configurations config files dir to path obj

        Args:
        -----
            config_label (str): Name of task configuration, e.g. "exp_msc"
        """
        self.paths.this_config = os.path.join(
            self.paths.task_configs, config_label)

    def sample_hiding_spots(self):
        """Method to sample hiding spots from a discrete uniform distribution
        over all nodes (without replacement)"""
        hides_loc = np.empty((self.params.n_blocks,
                             self.params.n_hides), dtype=int)
        for block in range(self.params.n_blocks):
            hides_loc[block] = np.random.choice(
                np.arange(1, self.params.n_nodes + 1),
                self.params.n_hides,
                replace=False)
        self.states['hides'] = hides_loc

    def sample_start_pos(self):
        """Method to sample the starting position from a discrete uniform
        distribution over all nodes"""
        s_1 = np.empty((self.params.n_blocks,
                       self.params.n_rounds), dtype=int)
        for block in range(self.params.n_blocks):
            for round_ in range(self.params.n_rounds):
                s_1[block, round_] = int(np.random.choice(
                    np.arange(1, self.params.n_nodes + 1),
                    1))
        self.states['s_1'] = s_1

    def sample_treasure_loc(self):
        """Method to sample the tr location from a discrete uniform
        distribution over all hiding spots"""
        s_3 = np.empty((self.params.n_blocks,
                       self.params.n_rounds), dtype=int)
        for block in range(self.params.n_blocks):
            for round_ in range(self.params.n_rounds):
                # Set treasure to equal start position
                s_3[block, round_] = cp.deepcopy(
                    self.states['s_1'][block, round_])
                # Sample tr location until it's not the starting position s_0
                while s_3[block, round_] == self.states['s_1'][block, round_]:
                    s_3[block, round_] = int(np.random.choice(
                        self.states['hides'][block], 1))
        self.states['s_3'] = s_3

    def save_task_config(self):
        """Method to save newly sampled task states to task config directory"""
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
        """Method to sample all task states s1, s3 and s4 for all trials/rounds
        and return dict with states"""
        self.sample_hiding_spots()
        self.sample_start_pos()
        self.sample_treasure_loc()
        self.save_task_config()

    def load_task_config(self):
        """Method to load existing task configuration files from task config
        directory"""
        for item in ['s_1', 's_3', 'hides']:
            self.states[item] = np.load(
                os.path.join(self.paths.this_config, f'{item}.npy'))

    def get_config(self, config_label: str):
        """Method to create or load task configuration

        Args:
        -----
            config_label (str): Name of task configuration, e.g. "exp_msc"
            new_config_requested (bool, optional): If True, samples
                new task configurations and saves it under given label to disk.
                    Loads existing configuration otherwise. Defaults to False.

        Returns:
            TODO: TODO
        """
        n_blocks = 3
        self.add_config_paths(config_label)
        if not os.path.exists(self.paths.this_config):
            self.params.n_blocks = n_blocks
            self.sample_task_config()
        else:
            self.load_task_config()
            self.params.n_blocks = list(
                self.states.values())[0].shape[0]

        return self


def get_arguments():
    """Function to fetch arguments from environment, if script is executed from
    command line or called within a sehllscript, e.g. jobwrapper."""
    parser = argparse.ArgumentParser(description='Run model validation.')
    parser.add_argument('--parallel_computing', action="store_true")
    parser.add_argument('--version', type=str, default="")
    parser.add_argument('--repetition', type=int, nargs='+')
    parser.add_argument('--agent_model', type=str, nargs='+')
    parser.add_argument('--tau_gen', type=float, nargs='+')
    parser.add_argument('--lambda_gen', type=float, nargs='+')
    parser.add_argument('--tau_cand_res', type=int)
    parser.add_argument('--lambda_cand_res', type=int)
    parser.add_argument('--participant', type=int, nargs='+')
    args = parser.parse_args()
    return args
