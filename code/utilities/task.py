"""_This module contains the treasure hunt task class to simulate agent
task interaction."""
import os
import json
import pickle
import time
import logging
from pympler import asizeof
from dataclasses import dataclass, field
import copy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from math import factorial
import more_itertools
from .config import Paths, DataHandler, humanreadable_time
from .cardinality_set_O import cOb, cOg


@dataclass
class TaskNGridParameters:
    """A data class to store Parameters of the task and gridworld configuration

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


class TaskStatesConfigurator:
    """A Class to create task state values given a set of task and grid
    parameters (i.e. no. trials, dimension of the grid, etc.) or
    load existend task states from hard drive given the label of a task
    configuration.

    Configuration-specific task state values are stored in the instance
    attribute "states (dict)". Newly sampled task configuration are written to
    .npy files in state samples directory on hard drive.

    Args:
    -----
        path (Path): Instance of class Paths

    Attributes:
    ----------
        params (TaskDesignParameters): Instance of class TaskDesignParams
        states (dict of str: np.ndarray): Configuration-specific state values
            "s1" : (n_blocks)x(n_rounds)-array of values indicating starting
                node positions of each round
            "s2": (n_blocks)x(n_rounds)-array of values indicating treasure
                locations
            "s3": (n_blocks)x(n_hides)-array of values indicating hiding
                spot locations
    """

    def __init__(self,
                 path: Paths,
                 task_params: TaskNGridParameters):
        self.paths = path
        self.params = task_params
        self.state_values = {}

    def add_config_paths(self, config_label: str,
                         n_I: int, n_H: int):
        """Add path to this task configurations config files dir to path obj

        Args:
        -----
            config_label (str): Name of task configuration, e.g. "exp_msc"
        """
        self.paths.this_state_sample = os.path.join(
            self.paths.state_samples,
            f"{config_label}_{n_I}-nodes_{n_H}_hides")

    def sample_s1(self):
        """Method to sample the starting position from a discrete uniform
        distribution over all nodes"""
        s1 = np.empty((self.params.n_blocks,
                       self.params.n_rounds), dtype=int)
        for block in range(self.params.n_blocks):
            for round_ in range(self.params.n_rounds):
                s1[block, round_] = int(np.random.choice(
                    np.arange(1, self.params.n_nodes + 1),
                    1))
        self.state_values['s1'] = s1

    def sample_s3(self):
        """Method to sample hiding spots from a discrete uniform distribution
        over all nodes (without replacement)"""
        s3 = np.empty((self.params.n_blocks,
                       self.params.n_hides), dtype=int)
        for block in range(self.params.n_blocks):
            s3[block] = np.random.choice(
                np.arange(1, self.params.n_nodes + 1),
                self.params.n_hides,
                replace=False)
        self.state_values['s3'] = s3

    def sample_s2(self):
        """Method to sample the tr location from a discrete uniform
        distribution over all hiding spots"""
        s2 = np.empty((self.params.n_blocks,
                       self.params.n_rounds), dtype=int)

        for block in range(self.params.n_blocks):
            for round_ in range(self.params.n_rounds):

                # Set treasure to equal start position
                s2[block, round_] = cp.deepcopy(
                    self.state_values['s1'][block, round_])

                # Sample tr location until it's not the starting position s_0
                while s2[block, round_] == self.state_values['s1'][
                        block, round_]:
                    s2[block, round_] = int(np.random.choice(
                        self.state_values['s3'][block], 1))
        self.state_values['s2'] = s2

    def save_task_config(self):
        """Method to save newly sampled task states to task config directory"""
        os.makedirs(self.paths.this_state_sample)
        for key, value in self.state_values.items():
            np.save(os.path.join(self.paths.this_state_sample, f'{key}.npy'),
                    value)

        config_df_fn = os.path.join(self.paths.this_state_sample,
                                    'state_values.tsv')
        all_block_df = pd.DataFrame()
        for block_ in range(self.params.n_blocks):
            this_block_df = pd.DataFrame(
                index=range(0, self.params.n_rounds))
            this_block_df['block'] = block_ + 1
            this_block_df['round'] = range(1,
                                           self.params.n_rounds
                                           + 1)
            this_block_df['s3'] = np.full(
                self.params.n_rounds, np.nan)
            this_block_df['s3'] = this_block_df[
                's3'].astype('object')
            for round_ in range(self.params.n_rounds):
                this_block_df.at[
                    round_, 's3'] = self.state_values['s3'][block_]
            this_block_df['s1'] = self.state_values['s1'][block_]
            this_block_df['s2'] = self.state_values['s2'][block_]

            all_block_df = pd.concat([all_block_df,
                                      this_block_df],
                                     ignore_index=True)

        with open(config_df_fn, 'w', encoding="utf-8") as tsv_file:
            tsv_file.write(all_block_df.to_csv(sep='\t', index=False))

    def sample_task_states(self):
        """Method to sample all task states for all trials/rounds
        and return dict with states"""
        self.sample_s1()
        self.sample_s3()
        self.sample_s2()
        self.save_task_config()

    def load_task_states(self):
        """Method to load existing task configuration files from task config
        directory"""
        for item in ['s1', 's2', 's3']:
            self.state_values[item] = np.load(
                os.path.join(self.paths.this_state_sample, f'{item}.npy'))

    def get_task_state_values(self, config_label: str) -> dict:
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
        self.add_config_paths(config_label, self.params.n_nodes,
                              self.params.n_hides)
        if not os.path.exists(self.paths.this_state_sample):
            self.sample_task_states()
        else:
            self.load_task_states()
            self.params.n_blocks = list(
                self.state_values.values())[0].shape[0]
            self.params.n_rounds = len(self.state_values["s1"][0])
        return self.state_values


class TaskSetsNCardinalities:
    """_summary_
    """
    def __init__(self, task_params: TaskNGridParameters):
        self.params = task_params
        self.n = self.compute_S_cardinality_n()      # Cardinality of S       n
        self.m = self.compute_O_cardinality_m()      # Cardinality of O       m
        self.p = 5                                   # Cardinality of A       p
        self.log_shapes()
        self.S = np.full(                            # Set of states          S
            (self.n, 2 + self.params.n_hides),
            99)
        # self.S = sp.csr_matrix(
        #     (self.n, self.params.n_hides + 2),  # shape
        #     dtype=np.int8)
        # self.O_ = {
        #     0: sp.csr_matrix(
        #         (int(self.m / 2), self.params.n_nodes),
        #         dtype=np.int8
        #     ),
        #     1: sp.csr_matrix(
        #         (int(self.m / 2), self.params.n_nodes),
        #         dtype=np.int8
        #     )
        # }
        # self.O_ = sp.csr_matrix(                    # Set of observations      O
        #     (self.m, self.params.n_nodes + 1),  # shape
        #     dtype=np.int8
        #     )
        # # self.O_ = np.full(                         # Set of observations      O
        # #     (self.m, 1 + self.params.n_nodes),
        # #     99,
        # #     dtype=np.int8)

        self.A = np.array(                         # Set of actions           A
            [0, -self.params.dim, 1,
             self.params.dim, -1], dtype=np.int8)
        self.R = np.array([0, 1], dtype=np.int8)                  # Set of rewards           R


    def log_shapes(self):
        """Function to log calculated shapes of sets and stochastic matrices,
        before initializing sets, which bursts working memory"""
        logging.info("Calculated shapes:")
        logging.info("                 Value/Shape")
        logging.info(" Cardinality  n: %s",
                     self.n)
        logging.info(" Cardinality  m: %s",
                     self.m,)
        logging.info("          set S: (%s, %s)",
                     self.n, 2 + self.params.n_hides)
        logging.info("          set O: (%s, %s)",
                     self.m, 1 + self.params.n_nodes)
        logging.info("           beta: (%s, 1)",
                     self.n)
        logging.info("            Phi: (5, %s, %s)",
                     self.n, self.n)
        logging.info("          Omega: (2, %s, %s)",
                     self.n, self.m)

    def compute_or_load_sets(self):
        """Function to check if files of state and observation sets exist
        on disk and start compution of both individually, otherwise."""

        logging.info("                 Value/Shape           Size")
        logging.info(" Cardinality  n: %s",
                     self.n)
        logging.info("                                       %s",
                     asizeof.asizeof(self.n))
        logging.info(" Cardinality  m: %s",
                     self.m,)
        logging.info("                                       %s \n",
                     asizeof.asizeof(self.m))

        data_handler = DataHandler(paths=Paths())
        # ------ Set of states-------------------------------------------------
        set_S_path = data_handler.create_matrix_fn(
            matrix_name="set_S",
            n_nodes=self.params.n_nodes,
            n_hides=self.params.n_hides
            )

        if os.path.exists(f"{set_S_path}.pkl"):
            # Load matrices from hd for this task grid configuration
            logging.info(
                "Loading set S from %s.pkl",
                set_S_path
                )
            # print("Loading set S of states from disk for given task config ("
            #       f"{self.params.n_nodes} nodes and "
            #       f"{self.params.n_hides} hiding spots) ...")
            start = time.time()
            with open(f"{set_S_path}.pkl", "rb") as file:
                self.S = pickle.load(file)
            end = time.time()
            logging.info(
                "Time needed to load Set s: %s \n",
                humanreadable_time(end-start)
                )
            # print(f" ... finished loading. \n ... time:  "
            #       f"{humanreadable_time(end-start)}\n")

        else:
            # Compute for this task grid configuration and save to hd
            logging.info("Computing set S for given task config ...")
            # print("Computing set S for given task config ("
            #       f"{self.params.n_nodes} nodes and "
            #       f"{self.params.n_hides} hiding spots) ...")
            start = time.time()
            self.compute_set_S()
            end = time.time()
            logging.info("Time needed to compute set S: %s",
                         humanreadable_time(end-start)
                         )
            # print(f" ... finished computing S. \n ... time:  "
            #       f"{humanreadable_time(end-start)}\n")
            start = time.time()
            data_handler.save_arrays(
                n_nodes=self.params.n_nodes,
                n_hides=self.params.n_hides,
                set_S=self.S
                )
            end = time.time()
            logging.info(
                "Time needed to save set S to disk: %s \n",
                humanreadable_time(end-start)
                )
            # print(f" ... finisehd writing S to disk. \n ... time:  "
            #       f"{humanreadable_time(end-start)}\n"
            #       )
        logging.info("                 Value/Shape           Size")
        logging.info("          set S: %s",
                     self.S.shape)
        logging.info("                                       %s \n",
                     asizeof.asizeof(self.S))

        # ------ Set of observations O2-------------------------------------------
        set_O2_path = data_handler.create_matrix_fn(
            matrix_name="set_O2",
            n_nodes=self.params.n_nodes,
            n_hides=self.params.n_hides)

        if os.path.exists(f"{set_O2_path}.npz"):
            # Load matrices from hd for this task grid configuration
            logging.info(
                "Loading set O from %s.npz",
                set_O2_path
                )
            # print("Loading set O of observations from disk for given task config ("
            #       f"{self.params.n_nodes} nodes and "
            #       f"{self.params.n_hides} hiding spots) ...")
            start = time.time()
            with open(f"{set_O2_path}.npz", "rb") as file:
                self.O_ = sp.load_npz(file)
            end = time.time()
            logging.info(
                "Time needed to load set O2: %s \n",
                humanreadable_time(end-start)
                )
            # print(f" ... finished loading. \n ... time:  "
            #       f"{humanreadable_time(end-start)}\n")

        else:
            # Compute for this task grid configuration and save to hd
            logging.info("Computing set O2 for given task config ...")
            # print("Computing set O2 for given task config ("
            #       f"{self.params.n_nodes} nodes and "
            #       f"{self.params.n_hides} hiding spots) ...")
            start = time.time()
            self.compute_O2()
            end = time.time()
            logging.info("Time needed to compute set O: %s",
                         humanreadable_time(end-start)
                         )
            # print(f" ... finished computing S. \n ... time:  "
            #       f"{humanreadable_time(end-start)}\n")
            start = time.time()
            data_handler.save_arrays(
                n_nodes=self.params.n_nodes,
                n_hides=self.params.n_hides,
                set_O=self.O_
                )
            end = time.time()
            logging.info("Time needed to save set O2 to disk: %s \n",
                         humanreadable_time(end-start))
            # print(f" ... finisehd writing O to disk. \n ... time:  "
            #       f"{humanreadable_time(end-start)}\n"
            #       )
        logging.info("                 Value/Shape           Size")
        logging.info("          set O2: %s",
                     self.O2_.shape)
        logging.info("                                       %s \n",
                     asizeof.asizeof(self.O_))



        # ------ Set of observations-------------------------------------------
        set_O_path = data_handler.create_matrix_fn(
            matrix_name="set_O",
            n_nodes=self.params.n_nodes,
            n_hides=self.params.n_hides)

        if os.path.exists(f"{set_O_path}.npz"):
            # Load matrices from hd for this task grid configuration
            logging.info(
                "Loading set O from %s.npz",
                set_O_path
                )
            # print("Loading set O of observations from disk for given task config ("
            #       f"{self.params.n_nodes} nodes and "
            #       f"{self.params.n_hides} hiding spots) ...")
            start = time.time()
            with open(f"{set_O_path}.npz", "rb") as file:
                self.O_ = sp.load_npz(file)
            end = time.time()
            logging.info(
                "Time needed to load set O: %s \n",
                humanreadable_time(end-start)
                )
            # print(f" ... finished loading. \n ... time:  "
            #       f"{humanreadable_time(end-start)}\n")

        else:
            # Compute for this task grid configuration and save to hd
            logging.info("Computing set O for given task config ...")
            # print("Computing set O for given task config ("
            #       f"{self.params.n_nodes} nodes and "
            #       f"{self.params.n_hides} hiding spots) ...")
            start = time.time()
            self.compute_set_O()
            end = time.time()
            logging.info("Time needed to compute set O: %s",
                         humanreadable_time(end-start)
                         )
            # print(f" ... finished computing S. \n ... time:  "
            #       f"{humanreadable_time(end-start)}\n")
            start = time.time()
            data_handler.save_arrays(
                n_nodes=self.params.n_nodes,
                n_hides=self.params.n_hides,
                set_O=self.O_,
                sparse=True
                )
            end = time.time()
            logging.info("Time needed to save set O to disk: %s \n",
                         humanreadable_time(end-start))
            # print(f" ... finisehd writing O to disk. \n ... time:  "
            #       f"{humanreadable_time(end-start)}\n"
            #       )
        logging.info("                 Value/Shape           Size")
        logging.info("          set O: %s",
                     self.O_.shape)
        logging.info("                                       %s \n",
                     asizeof.asizeof(self.O_))

    def compute_S_cardinality_n(self):
        """Function to compute n = cardinality of set S, which is
        the number of possible current position times (the number of
        possible treasure locations X the ratio of hides to nodes."""
        n_S1 = self.params.n_nodes  # No. possible currents positions

        n_I = self.params.n_nodes
        n_H = self.params.n_hides

        # hide_combos = sorted(
        #     more_itertools.distinct_combinations(
        #         iterable=range(1, n_I + 1),
        #         r=n_H
        #         )
        #     )
        # n_h_combos_count = len(hide_combos)

        n_S3 = (factorial(n_I)
                / (factorial(n_I - n_H) * factorial(n_H))
                )

        # Compute the cardinality n of set S
        n = (n_S1 * n_H * n_S3)
        # n = 48  # TODO: QUICKFIX

        return int(n)  # TODO: hier weiter: geht alles nicht auf. ...

    def compute_set_S(self):
        """Method to compute the set of states"""
        n_nodes = self.params.n_nodes
        n_hides = self.params.n_hides

        hide_combos = sorted(
            more_itertools.distinct_combinations(
                iterable=range(1, n_nodes + 1),
                r=n_hides
                )
            )

        i = 0

        for possible_position in range(n_nodes):
            possible_position += 1  # Account for nodes = {1, ..., n_nodes}

            for possible_tr_loc in range(n_nodes):
                possible_tr_loc += 1

                for hiding_spot_combo in hide_combos:

                    # Include only states, that are possible, according to the
                    # rule that treasures can only be hid on hiding spots
                    if possible_tr_loc in hiding_spot_combo:

                        self.S[i, 0] = possible_position                  # s^1
                        self.S[i, 1] = possible_tr_loc                    # s^2
                        self.S[i, 2:] = hiding_spot_combo                 # s^3

                        i += 1

    def compute_O2(self) -> list:
        """Method to compute the set O2, that is the second component of the
        observation vector, which represents the the node colors of the grid.
        """
        n_nodes = self.params.n_nodes
        n_hides = self.params.n_hides

        # Create list of values for the urn model
        urn_values = [0] * n_nodes  # TODO: hier weiter, Permutationen berechnen
        urn_values.extend([1] * (n_nodes - n_hides))
        urn_values.extend([2] * n_hides)

        self.O2 = sorted(
            more_itertools.distinct_permutations(
                iterable=urn_values,  # All values in the Urn model
                r=n_nodes             # Number of items to be sampled
                )
            )

        #return O2

    def compute_O_cardinality_m(self):
        """"Function to compute m = cardinality of set O, which 2 x the number
        of node color combinations."""
        # self.compute_O2()  # TODO: doppelt!
        # n_node_color_combinations = len(self.O2)  # TODO: formula dafür?!?
        # m = 2 * n_node_color_combinations  # TP

        # von Dirks Skript:
        # --------------------------------
        colors = [0, 1, 2]
        n_o2c = (                      # cardinality of the unconstrained O2 set
            len(colors) ** self.params.n_nodes
            )

        # analytical solution
        n_o2b = cOb(                   # cardinality of violation set O2_b
            n_c=self.params.n_nodes,
            n_h=self.params.n_hides
            )
        n_o2g = cOg(                   # cardinality of violation set O2_g
            n_c=self.params.n_nodes,
            n_h=self.params.n_hides
            )
        n_o2 = n_o2c - n_o2b - n_o2g  # cardinality of the constrained O2 set 
        n_o = 2 * n_o2

        return n_o

    def compute_set_O(self):
        """Method to compute complete set of Observations"""
        self.compute_O2()  # Node color combinations
        O1 = [0, 1]  # Treasure flag
        i = 0

        rows = []
        cols = []
        data = []

        # Append row and col indices first o[0] = 1
        rows.extend(range(len(self.O2), len(self.O2) * 2))
        cols.extend([0] * len(self.O2))
        data.extend([1] * len(self.O2))

        for o1 in O1:  # Iterate treasure flags

            for o2 in self.O2:  # Iterate node color combinations

                for col_index, o2_enty in enumerate(o2):

                    if o2_enty != 0:
                        rows.append(i)
                        cols.append(col_index + 1)
                        data.append(o2_enty)

                    # self.O_[i, 0] = o1
                    # self.O_[i, 1:] = o2

                i += 1  # Move to next row of O2

        self.O_ = sp.csr_matrix(
            (data,  # data
             (rows, cols,)),  # index
            shape=(self.m, self.params.n_nodes + 1),  # shape
            dtype=np.int8
             )


class Task:
    """A class used to represent the treasure hunt task

    A task object can interact with an agent object within an
    agent-based behavioral modelling framework.
    """

    def __init__(self,
                 state_values: dict,
                 task_params: TaskNGridParameters,
                 task_sets_n_cardinalities: TaskSetsNCardinalities):
        """A class to represent the tresaure hunt task

        Args:
            states (dict): task state values e.g. hiding spots,
                treasure location, starting nodes etc
        """

        # Configuration specific task and gridworld components
        self.state_values: dict = state_values
        self.sets_n_cardins: TaskSetsNCardinalities = task_sets_n_cardinalities
        self.params = task_params
        self.node_colors = np.full(self.params.n_nodes, 0)
        self.shortest_dist_dic: dict = {}

        # Task model components
        self.T = range(0, self.params.n_trials + 1)  # Trials                 T
        self.C = range(1, self.params.n_rounds + 1)  # Rounds                 C

        # Dynamik, i.e. trial-specific model components
        self.t: int = 0  # Current trial
        self.c: int = 0  # Curret round
        self.s1_t = np.full(1, np.nan)             # Current posion       s^1_t
        self.s2_t = np.full(1, np.nan)             # Current treasure loc s^2_t
        self.s3_t = np.full(                       # Hiding spots         s^3_t
            (1, self.params.n_hides), np.nan)
        self.o_t = np.full(                        # Current observation    o_t
            (1 + self.params.n_nodes), np.nan)
        self.r_t: int = 0                          # Current reward         r_t

        # Compute or load shortest distance dict from hd
        self.compute_or_load_shortest_distances()

    def attach_sets(self, sets: TaskSetsNCardinalities):
        """Method to attach state sets object to task object

        Args:
            sets (TaskSets): object storing the set O of observations and 
                                 set S of states
        """

    def compute_or_load_shortest_distances(self):
        """Get the shortest distances between two nodes from json or evaluate
        save to json if not existent"""

        # Specify path for shortest_distances storage file
        paths = Paths()
        short_dist_fn = os.path.join(
            paths.code, 'utilities',
            f'shortest_dist_dim-{self.params.dim}.json')

        # Read in json file as dic if existent for given dimensionality
        if os.path.exists(short_dist_fn):
            with open(short_dist_fn, encoding="utf8") as json_file:
                self.shortest_dist_dic = json.load(json_file)

        # Create new json file if not yet existent and
        else:
            self.eval_shortest_distances()
            with open(short_dist_fn, 'w', encoding="utf8") as json_file:
                json.dump(self.shortest_dist_dic, json_file, indent=4)

    def eval_shortest_distances(self):
        """Evaluate the shortest distance between all nodes in grid world with
        dimension dim given all available actions in a_set.
        The shortest path is expressed in numbers of steps needed to
        reach the end node when standing on a given start node
        """
        # ------Initialize variables / objects--------------------------------
        n_nodes = self.params.n_nodes  # number of nodes in the graph
        dim = self.params.dim  # dimension of the grid world
        moves = self.sets_n_cardins.A[:4]  # possible moves / actions

        # ------Create adjacency matrix---------------------------------------
        adj_matrix = []  # Initialize adjacency matrix
        # Iterate over all fields and create row with ones for adjacent fields
        for i in range(n_nodes):
            row = np.full(n_nodes, 0)  # Initialize row with all zeros
            for move in moves:
                if ((i + move) >= 0) and ((i + move) < n_nodes):
                    if ((i % dim != 0) and move == -1) or \
                            ((((i + 1) % dim) != 0) and (move == 1)) or \
                            (move == self.params.dim
                             or move == -self.params.dim):
                        row[i + move] = 1
            adj_matrix.append(list(row))

        # ------Create adjacency list (dictionary)----------------------------
        adj_list = {}  # Initialize adjacency dictionary
        # Iterate over all fields and create dict. entry with adjacent fields
        for i in range(n_nodes):
            row_list = []
            for move in moves:
                if ((i + move) >= 0) and ((i + move) < n_nodes):
                    if ((i % dim != 0) and move == -1) or \
                            ((((i + 1) % dim) != 0) and (move == 1)) or \
                            (move in [self.params.dim,
                                      -self.params.dim]):
                        row_list.append(i + move)
                        row_list.sort()
            adj_list.update({i: row_list})

        # -------Iterate through starting nodes:-------
        for start_node in range(n_nodes):

            # ------Iterate through ending nodes:------
            for end_node in range(n_nodes):

                # Return zero if start_node equals end_node
                if start_node == end_node:
                    self.shortest_dist_dic[
                        f'{start_node + 1}_to_{end_node + 1}'] = 0
                    self.shortest_dist_dic[
                        f'{end_node + 1}_to_{start_node + 1}'] = 0

                else:
                    # Keep track of all visited nodes of a graph
                    explored = []
                    # keep track of all the paths to be checked
                    queue = [[start_node]]

                    # Keep looping until queue is empty
                    while queue:
                        # Pop the first path from the queue
                        path = queue.pop(0)
                        # Get the last node from path
                        node = path[-1]

                        if node not in explored:
                            neighbours = adj_list[node]

                            # Go through all neighbouring nodes, construct new
                            # path and push into queue
                            for neighbour in neighbours:
                                new_path = list(path)
                                new_path.append(neighbour)
                                queue.append(new_path)

                                # Return path if neighbour is end node
                                if neighbour == end_node:

                                    shortest_path = new_path
                                    shortest_distance = len(shortest_path)-1

                                    # Add the shortest path to dictionary
                                    self.shortest_dist_dic[
                                        f'{start_node + 1}_to_{end_node + 1}'
                                    ] = shortest_distance
                                    self.shortest_dist_dic[
                                        f'{end_node + 1}_to_{start_node + 1}'
                                    ] = shortest_distance
                                    queue = []
                                    break

                            # Mark node as explored
                            explored.append(node)

    def start_new_block(self, block_number):
        """Start new block with block-specific state values

        Parameters
        ----------
        block (int) : block number
        """
        self.s3_t = self.state_values["s3"][block_number]

    def start_new_round(self, block_number, round_number: int):
        """Fetch configuration-specific initial task states and reset
        dynamic states to initial values for a new round"""
        self.c = round_number
        self.s1_t = self.state_values["s1"][block_number, round_number]
        self.s2_t = self.state_values["s2"][block_number, round_number]
        self.r_t = 0  # reward

    def eval_obs_func_g(self):
        """Return observation, i.e. each node current status (color) and
        treasure disc (yes/no). This function maps action, reward and states
        s3 and s4 onto observation o_t, as specified in g
        """

        if self.r_t == 0:
            self.o_t[0] = 0
        else:
            self.o_t[0] = 1

        self.o_t[1:] = self.node_colors

    def eval_state_transition_f(self, action_t):
        """Perform the state transition function f. """
        # Move to new position (transition s1)
        self.s1_t += int(action_t)

        # After informative actions
        if action_t == 0:

            # Change node colors (transition)
            if self.s1_t not in self.s3_t:  # If s1 not hiding spot, set grey
                self.node_colors[self.s1_t] = 1
            elif self.s1_t in self.s3_t:  # Elif s1 is hiding spot, set blue
                self.node_colors[self.s1_t] = 2

    def eval_reward_r(self):
        """Evaluate whether new current position is the treasure location"""
        if self.s1_t == self.s2_t:  # if s1 equals treasure location
            self.r_t = 1

        else:
            self.r_t = 0

    def eval_action(self, action_t):
        """Evaluate beh_model action and update affected task states"""

        self.eval_state_transition_f(action_t)

        # If participant decides to take a step
        # -----------------------------------------------------
        if action_t != 0:

            # Evaluate whether new position is treasure location
            self.eval_reward_r()
