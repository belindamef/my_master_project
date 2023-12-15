"""_This module contains the treasure hunt task class to simulate agent
task interaction."""
import os
import json
import numpy as np
from math import factorial as fac
import more_itertools
from utilities.config import TaskConfigurator, TaskDesignParameters
from .config import Paths
from functools import reduce


class Task:
    """A class used to represent the treasure hunt task

    A task object can interact with an agent object within an
    agent-based behavioral modelling framework.
    """

    def __init__(self, task_configs,
                 task_params: TaskDesignParameters = TaskDesignParameters()):
        """A class to represent the tresaure hunt task

        Args:
            task_configs (TaskConfigurator): configuration e.g. hiding spots,
                treasure location, starting nodes etc
        """

        # Observabale gridworld components
        self.task_configs: TaskConfigurator = task_configs
        self.grid_config = task_params
        self.node_colors = np.full(self.grid_config.n_nodes, 0)

        # Task model components
        self.T = range(0, self.grid_config.n_trials + 1)   # Trials
        self.C = range(1, self.grid_config.n_rounds + 1)   # Rounds
        self.n = self.compute_S_cardinality_n()            # Cardinality of S
        self.m = self.compute_O_cardinality_m()            # Cardinality of O
        self.p = 5
        self.S = np.full(                                  # Set of states  S
            (self.n, 2 + self.grid_config.n_hides),
            np.nan)
        self.O = np.full(                                  # Set of observ. O
            (self.m, 1 + self.grid_config.n_nodes),
            np.nan)
        self.A = np.array(                                 # Set of actions A
            [0, -self.grid_config.dim, 1,
             self.grid_config.dim, -1])
        self.R = np.array([0, 1])                          # Set of rewards R

        # Dynamik model components
        self.t: int = 0  # Current trial
        self.c: int = 0  # Curret round
        self.s1_t = np.full(1, np.nan)                     # Current pos  s^1_t
        self.s2_t = np.full(1, np.nan)                     # Treasure loc s^2_t
        self.s3_t = np.full(                               # Hiding spots s^3_t
            (1, self.grid_config.n_hides), np.nan)
        self.o_t = np.full(                                # Observation    o_t
            (1 + self.grid_config.n_nodes), np.nan)
        self.r_t = 0                                       # Rewart         r_t

        # Compute or load sets and matrices of task model
        self.compute_set_S()
        self.compute_set_O()

        # Get the shortest distances between two nodes from json or evaluate
        # save to json if not existent
        # ---------------------------------------------------------------------
        # Initialize dictionary with the shortest distances
        self.shortest_dist_dic = {}

        # Specify path for shortest_distances storage file
        paths = Paths()
        short_dist_fn = os.path.join(
            paths.code, 'utilities',
            f'shortest_dist_dim-{self.grid_config.dim}.json')
        # Read in json file as dic if existent for given dimensionality
        if os.path.exists(short_dist_fn):
            with open(short_dist_fn, encoding="utf8") as json_file:
                self.shortest_dist_dic = json.load(json_file)
        # Create new json file if not yet existent and
        else:
            self.eval_shortest_distances()
            with open(short_dist_fn, 'w', encoding="utf8") as json_file:
                json.dump(self.shortest_dist_dic, json_file, indent=4)

    def compute_S_cardinality_n(self):
        """Function to compute n = cardinality of set S, which is
        the number of possible current position times (the number of
        possible treasure locations X the ratio of hides to nodes."""
        n_pos = self.grid_config.n_nodes  # No. possible currents positions
        n_tr = self.grid_config.n_nodes  # No. possible treasure locations
        hide_to_node_ratio = (  # No. of hiding spots to No. nodes ratio
            self.grid_config.n_hides / self.grid_config.n_nodes)
        # TODO: Does this reflect the "treasure possibility given all hiding
        # spot combinations?"

        # Compute number of distinct comibinations for hiding spots according
        # to the binomial coefficient formula,
        # also known as "n choose r" or "combinations."
        n_it = self.grid_config.n_nodes       # length of input iterable
        r = self.grid_config.n_hides          # number of items taken from the
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
        n_nodes = self.grid_config.n_nodes
        n_hides = self.grid_config.n_hides

        hide_combos = sorted(
            more_itertools.distinct_combinations(
                range(1, n_nodes + 1), r=n_hides
                )
        )

        # s_debug_list = []

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
        n_nodes = self.grid_config.n_nodes
        n_hides = self.grid_config.n_hides

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
                self.O[i, 0] = treasure_flag
                self.O[i, 1:] = color_combo

                i += 1

    def eval_shortest_distances(self):
        """Evaluate the shortest distance between all nodes in grid world with
        dimension dim given all available actions in a_set.
        The shortest path is expressed in numbers of steps needed to
        reach the end node when standing on a given start node
        """
        # ------Initialize variables / objects--------------------------------
        n_nodes = self.grid_config.n_nodes  # number of nodes in the graph
        dim = self.grid_config.dim  # dimension of the grid world
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
                            (move == self.grid_config.dim
                             or move == -self.grid_config.dim):
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
                            (move in [self.grid_config.dim,
                                      -self.grid_config.dim]):
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

    def start_new_trial(self, current_trial: int):
        """Reset dynamic states to initial values for a new trial"""
        self.t = current_trial

    def return_observation(self):
        """Return observation, i.e. each node current status (color) and
        treasure disc (yes/no). This function maps action, reward and states
        s3 and s4 onto observation o_t, as specified in g
        """

        if self.r_t == 0:
            self.o_t[0] = 0
        else:
            self.o_t[0] = 1

        self.o_t[1:] = self.node_colors

    def perform_state_transition_f(self, action_t):
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

    def return_reward_r(self):
        """Evaluate whether new current position is the treasure location"""
        if self.s1_t == self.s2_t:  # if s1 equals treasure location
            self.r_t = 1

        else:
            self.r_t = 0

    def eval_action(self, action_t):
        """Evaluate beh_model action and update affected task states"""

        self.perform_state_transition_f(action_t)

        # If participant decides to take a step
        # -----------------------------------------------------
        if action_t != 0:

            # Evaluate whether new position is treasure location
            self.return_reward_r()

