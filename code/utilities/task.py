"""_This module contains the treasure hunt task class to simulate agent
task interaction."""
import os
import json
import pickle
import time
from dataclasses import dataclass, field
import copy as cp
import numpy as np
import pandas as pd
from math import factorial as fac
import more_itertools
from .config import Paths, DataHandler, humanreadable_time, get_user_yes_no


class Task:
    """A class used to represent the treasure hunt task

    A task object can interact with an agent object within an
    agent-based behavioral modelling framework.
    """

    def __init__(self, task_configs,
                 grid_config: GridConfigParameters):
        """A class to represent the tresaure hunt task

        Args:
            task_configs (TaskConfigurator): configuration e.g. hiding spots,
                treasure location, starting nodes etc
        """
        self.paths: Paths = Paths()  # TODO:  lieber übergeben?
        self.task_configs: TaskConfigurator = task_configs

        # Observabale gridworld components
        self.grid: GridConfigParameters = grid_config
        self.node_colors = np.full(self.grid.n_nodes, 0)
        self.shortest_dist_dic: dict = {}

        # Task model components
        self.T = range(0, self.grid.n_trials + 1)  # Trials                   T
        self.C = range(1, self.grid.n_rounds + 1)  # Rounds                   C
        self.n = self.compute_S_cardinality_n()    # Cardinality of S         n
        self.m = self.compute_O_cardinality_m()    # Cardinality of O         m
        self.p = 5                                 # Cardinality of A         p
        self.S = np.full(                          # Set of states            S
            (self.n, 2 + self.grid.n_hides),
            np.nan)
        self.O_ = np.full(                         # Set of observations      O
            (self.m, 1 + self.grid.n_nodes),
            np.nan)
        self.A = np.array(                         # Set of actions           A
            [0, -self.grid.dim, 1,
             self.grid.dim, -1])
        self.R = np.array([0, 1])                  # Set of rewards           R

        # Dynamik model components
        self.t: int = 0  # Current trial
        self.c: int = 0  # Curret round
        self.s1_t = np.full(1, np.nan)             # Current posistopm    s^1_t
        self.s2_t = np.full(1, np.nan)             # Current treasure loc s^2_t
        self.s3_t = np.full(                       # Hiding spots         s^3_t
            (1, self.grid.n_hides), np.nan)
        self.o_t = np.full(                        # Current observation    o_t
            (1 + self.grid.n_nodes), np.nan)
        self.r_t: int = 0                          # Current reward         r_t

        # Compute or load sets and matrices of task model
        self.compute_or_load_set_matrices()
        self.compute_set_S()
        self.compute_set_O()
        # Compute or load shortest distances between nodes
        self.compute_or_load_shortest_distances()

    def compute_or_load_shortest_distances(self):
        """Get the shortest distances between two nodes from json or evaluate
        save to json if not existent"""

        # Specify path for shortest_distances storage file
        short_dist_fn = os.path.join(
            self.paths.code, 'utilities',
            f'shortest_dist_dim-{self.grid.dim}.json')
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
        n_nodes = self.grid.n_nodes  # number of nodes in the graph
        dim = self.grid.dim  # dimension of the grid world
        moves = self.A[:4]  # possible moves / actions

        # ------Create adjacency matrix---------------------------------------
        adj_matrix = []  # Initialize adjacency matrix
        # Iterate over all fields and create row with ones for adjacent fields
        for i in range(n_nodes):
            row = np.full(n_nodes, 0)  # Initialize row with all zeros
            for move in moves:
                if ((i + move) >= 0) and ((i + move) < n_nodes):
                    if ((i % dim != 0) and move == -1) or \
                            ((((i + 1) % dim) != 0) and (move == 1)) or \
                            (move == self.grid.dim
                             or move == -self.grid.dim):
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
                            (move in [self.grid.dim,
                                      -self.grid.dim]):
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

    def compute_or_load_set_matrices(self):
        """Function to check if files of state and observation sets exist
        on disk and start compution of both individually, otherwise."""

        data_handler = DataHandler(paths=self.paths)

        # ------ Set of states-------------------------------------------------
        set_S_path = data_handler.create_matrix_fn(matrix_name="set_S",
                                                   n_nodes=self.grid.n_nodes,
                                                   n_hides=self.grid.n_hides)

        if os.path.exists(f"{set_S_path}.pkl"):
            print("Loading set S of states from disk for given task config ("
                  f"{self.grid.n_nodes} nodes and "
                  f"{self.grid.n_hides} hiding spots) ...")
            start = time.time()
            with open(f"{set_S_path}.pkl", "rb") as file:
                self.S = pickle.load(file)
            end = time.time()
            print(f" ... finished loading. \n ... time:  "
                  f"{humanreadable_time(end-start)}\n")
        else:
            print("Computing set S for given task config ("
                  f"{self.grid.n_nodes} nodes and "
                  f"{self.grid.n_hides} hiding spots) ...")
            start = time.time()
            self.compute_set_S()
            end = time.time()
            print(f" ... finished computing S. \n ... time:  "
                  f"{humanreadable_time(end-start)}\n")
            start = time.time()
            data_handler.save_arrays(
                n_nodes=self.grid.n_nodes,
                n_hides=self.grid.n_hides,
                set_S=self.S
                )
            end = time.time()
            print(f" ... finisehd writing S to disk. \n ... time:  "
                  f"{humanreadable_time(end-start)}\n"
                  )

        # ------ Set of observations-------------------------------------------
        set_O_path = data_handler.create_matrix_fn(
            matrix_name="set_O",
            n_nodes=self.grid.n_nodes,
            n_hides=self.grid.n_hides)

        if os.path.exists(f"{set_O_path}.pkl"):
            print("Loading set S of states from disk for given task config ("
                  f"{self.grid.n_nodes} nodes and "
                  f"{self.grid.n_hides} hiding spots) ...")
            start = time.time()
            with open(f"{set_O_path}.pkl", "rb") as file:
                self.O_ = pickle.load(file)
            end = time.time()
            print(f" ... finished loading. \n ... time:  "
                  f"{humanreadable_time(end-start)}\n")
        else:
            print("Computing set O for given task config ("
                  f"{self.grid.n_nodes} nodes and "
                  f"{self.grid.n_hides} hiding spots) ...")
            start = time.time()
            self.compute_set_O()
            end = time.time()
            print(f" ... finished computing S. \n ... time:  "
                  f"{humanreadable_time(end-start)}\n")
            start = time.time()
            data_handler.save_arrays(
                n_nodes=self.grid.n_nodes,
                n_hides=self.grid.n_hides,
                set_O=self.O_
                )
            end = time.time()
            print(f" ... finisehd writing O to disk. \n ... time:  "
                  f"{humanreadable_time(end-start)}\n"
                  )

    def compute_S_cardinality_n(self):
        """Function to compute n = cardinality of set S, which is
        the number of possible current position times (the number of
        possible treasure locations X the ratio of hides to nodes."""
        n_pos = self.grid.n_nodes  # No. possible currents positions
        n_tr = self.grid.n_nodes  # No. possible treasure locations
        hide_to_node_ratio = (  # No. of hiding spots to No. nodes ratio
            self.grid.n_hides / self.grid.n_nodes)
        # TODO: Does this reflect the "treasure possibility given all hiding
        # spot combinations?"

        # Compute number of distinct comibinations for hiding spots according
        # to the binomial coefficient formula,
        # also known as "n choose r" or "combinations."
        n_it = self.grid.n_nodes       # length of input iterable
        r = self.grid.n_hides          # number of items taken from the
        # iterable to form combinations

        n_h_combos = (                        # number of distinct combinations
            fac(n_it) / fac(r) * fac(n_it - r)  # of hiding spots
                 )

        # Compute the cardinality n of set S
        n = (n_pos * n_tr * n_h_combos * hide_to_node_ratio)
        n = 48

        return int(n)  # TODO: hier weiter: geht alles nicht auf. ...

    def compute_set_S(self):
        """Method to compute the set of states"""
        n_nodes = self.grid.n_nodes
        n_hides = self.grid.n_hides

        hide_combos = sorted(
            more_itertools.distinct_combinations(
                range(1, n_nodes + 1), r=n_hides
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
                        # s_debug_list.append(i)
                        i += 1

        # a_test_n_hide_combos = len(hide_combos)
        # a_test_cardinality_formula = cardinality_S
        # a_test_list = len(s_debug_list)
        # bp = "here"

    def compute_O_cardinality_m(self):
        """"Function to compute m = cardinality of set O, which 2 x the number
        of node color combinations."""
        n_node_color_combinations = 63  # TODO: formula dafür?!?
        m = 2 * n_node_color_combinations  # TP
        return m

    def compute_node_color_combos(self) -> list:
        """Method to compute the node color components of observation set, i.e.
        the last 4 entries of each set element's vector"""
        n_nodes = self.grid.n_nodes
        n_hides = self.grid.n_hides

        obs_values = [0] * n_nodes
        obs_values.extend([1] * (n_nodes - n_hides))
        obs_values.extend([2] * n_hides)

        node_color_combos = sorted(
            more_itertools.distinct_permutations(obs_values, r=n_nodes))

        return node_color_combos

    def compute_set_O(self):
        """Method to compute complete set of Observations"""
        node_color_combos = self.compute_node_color_combos()

        i = 0
        for treasure_flag in [0, 1]:

            for color_combo in node_color_combos:

                # TODO: write observations
                self.O_[i, 0] = treasure_flag
                self.O_[i, 1:] = color_combo

                i += 1

    def start_new_block(self, block_number):
        """Start new block with new task_configuration

        Parameters
        ----------
        task_config : TaskConfigurator
        """
        self.s3_t = self.task_configs.states["hides"][block_number]

    def start_new_round(self, block_number, round_number: int):
        """Fetch configuration-specific initial task states and reset
        dynamic states to initial values for a new round"""
        self.c = round_number
        self.s1_t = self.task_configs.states["s_1"][block_number, round_number]
        self.s2_t = self.task_configs.states["s_3"][block_number, round_number]
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
        # Move to new position (transition s_1)
        self.s1_t += int(action_t)

        # After informative actions
        if action_t == 0:

            # Change node colors (transition)
            if self.s1_t not in self.s3_t:  # If s_1 not hiding spot, set grey
                self.node_colors[self.s1_t] = 1
            elif self.s1_t in self.s3_t:  # Elif s_1 is hiding spot, set blue
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


@dataclass
class GridConfigParameters:
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

    def __init__(self, path: Paths,
                 params=GridConfigParameters()):
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
