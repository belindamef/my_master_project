import numpy as np
import copy as cp
import os
import json


class Task:
    """A class used to represent the treasure hunt task model

    A task object can interact with an agent object within an
    agent-based behavioral modelling framework

    Attributes
    ----------
        #TODO
    """

    def __init__(self, t_init):
        """This function is the instantiation operation of the task class.

        Parameters
        ----------
        t_init         : initialization structure with fields
              .model   : behavioral model
              .model.a : action values
        """
        # Directory management
        self.working_dir = t_init.working_dir
        self.model = None

        # Initialize round and trial count
        self.c = np.nan  # round counter
        self.t = np.nan  # trial counter

        # Initialize experiment parameters based on init object
        self.rounds = t_init.rounds  # number of hunting rounds per task
        self.trials = t_init.trials  # number of task trials per round (i.e. available attempts)

        # Set task properties based on init object
        self.dim = t_init.dim  # dimension of the gird world
        self.n_hides = t_init.n_hides  # No. of hiding spots
        self.n_nodes = self.dim ** 2  # No. of fields

        # Initialize task states
        self.s_1 = np.full(1, np.nan)  # current position
        self.s_2_node_color = np.full(self.n_nodes, 0)  # node color {0, 1, 2}, initial value: 0 = black
        self.s_3_tr_loc = np.full(1, np.nan)  # hidden state, treasure location of current round
        self.s_4_hide_node = np.full(self.n_nodes, 0)  # hiding spot or not; init value: 0 = not

        # Initialize set of state-dependent action sets
        self.A = np.array([0, -self.dim, 1, self.dim, -1])  # set of actions

        # Initialize current trial observation
        self.o_t = np.full(1, np.nan)  # observation

        # Initialize variables for computations
        self.tr_disc = 0  # treasure discovery at current position, initial value = 0
        self.hides_loc = np.full(self.n_hides, np.nan)  # hidden state, hiding spots of current block/task
        self.n_black = cp.deepcopy(self.n_nodes)
        self.n_blue = 0
        self.n_grey = 0
        self.n_hides_left = cp.deepcopy(self.n_hides)
        self.drill_finding = np.nan
        self.tr_found_on_blue = np.nan

        # Get shortest distances between two nodes from json or evaluate save to json if not existent
        # ---------------------------------------------------------------------
        # Initialize dictionary with shortest distances
        self.shortest_dist_dic = {}

        # Specify path for shortest_distances storage file
        short_dist_fn = os.path.join(self.working_dir, 'utilities', f'shortest_dist_dim-{self.dim}.json')
        # Read in json file as dic if existent for given dimensionality
        if os.path.exists(short_dist_fn):
            with open(short_dist_fn) as json_file:
                self.shortest_dist_dic = json.load(json_file)
        # Create new json file if not yet existent and
        else:
            self.evaluate_shortest_distances()
            with open(short_dist_fn, 'w') as json_file:
                json.dump(self.shortest_dist_dic, json_file, indent=4)

    def evaluate_shortest_distances(self):
        """
        Evaluate the shortest distance between all nodes in grid world with
        dimension <dim> given all available actions in the task's action set.
        The shortest path is expressed in numbers of steps needed to
        reach the end node when standing on a given start node

        file creation
        -------------
            this function will create a json file containing the shortest distances
            between two nodes
        """
        # ------Initialize variables / objects--------------------------------
        n_nodes = cp.deepcopy(self.n_nodes)  # number of nodes in the graph
        dim = cp.deepcopy(self.dim)  # dimension of the grid world
        moves = cp.deepcopy(self.A[:4])  # possible moves / actions

        # ------Create adjacency matrix---------------------------------------
        adj_matrix = []  # Initialize adjacency matrix
        # Iterate over all fields and create row with ones for adjacent fields
        for i in range(n_nodes):
            row = np.full(n_nodes, 0)  # Initialize row with all zeros
            for move in moves:
                if ((i + move) >= 0) and ((i + move) < n_nodes):
                    if ((i % dim != 0) and move == -1) or \
                            ((((i + 1) % dim) != 0) and (move == 1)) or \
                            (move == self.dim or move == -self.dim):
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
                            (move == self.dim or move == -self.dim):
                        row_list.append(i + move)
                        row_list.sort()
            adj_list.update({i: row_list})

        # -------Iterate through starting nodes:-------
        for start_node in range(n_nodes):

            # ------Iterate through ending nodes:------
            for end_node in range(n_nodes):

                # Return zero if start_node equals end_node
                if start_node == end_node:
                    self.shortest_dist_dic[f'{start_node}_to_{end_node}'] = 0
                    self.shortest_dist_dic[f'{end_node}_to_{start_node}'] = 0

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
                        node = int(cp.deepcopy(path[-1]))

                        if node not in explored:
                            neighbours = cp.deepcopy(adj_list[node])
                            # not_visited_neighbours = cp.deepcopy(neighbours)
                            # for i in not_visited_neighbours:
                            #     if i in path:
                            #         not_visited_neighbours.remove(i)

                            # Go through all neighbouring nodes, construct new path and push into queue
                            for neighbour in neighbours:
                                new_path = list(path)
                                new_path.append(neighbour)
                                queue.append(new_path)

                                # Return path if neighbour is end node
                                if neighbour == end_node:

                                    shortest_path = new_path
                                    shortest_distance = len(shortest_path)-1

                                    # Add shortest path to dictionary
                                    self.shortest_dist_dic[f'{start_node}_to_{end_node}'] = shortest_distance
                                    self.shortest_dist_dic[f'{end_node}_to_{start_node}'] = shortest_distance
                                    queue = []
                                    break

                            # Mark node as explored
                            explored.append(node)

    def set_s_4(self):
        """Evaluate s_4 values"""
        # Set values in state s_4 according to hide_loc
        for node in self.hides_loc:
            self.s_4_hide_node[node] = 1

    def start_new_round(self):
        """Reset dynamic states to initial values for a new round"""
        self.tr_disc = 0  # treasure discovery
        self.tr_found_on_blue = np.nan

    def start_new_trial(self):
        """Reset dynamic states to initial values for a new trial"""
        self.drill_finding = np.nan

    def return_observation(self):
        """Return observation, i.e. each nodes current status (color) and treasure (yes/no).
        This function maps action, reward and states s3 and s4 onto observation o2, as specified in g"""

        # If node color black and no treasure
        if (self.s_2_node_color[int(self.s_1)] == 0) and (self.tr_disc == 0):
            self.o_t = 0

        # If node color = grey (always no treasure found)
        elif self.s_2_node_color[int(self.s_1)] == 1:
            self.o_t = 1

        # If node color = blue and no treasure
        elif (self.s_2_node_color[int(self.s_1)] == 2) and (self.tr_disc == 0):
            self.o_t = 2

        # If treasure found
        elif self.tr_disc == 1:
            self.o_t = 3

    def perform_state_transition_f(self):
        """Perform the state transition function f"""
        # Move to new position (transition s_1)
        self.s_1 = cp.deepcopy(self.s_1) + cp.deepcopy(self.model.a)

        # After informative actions
        if self.model.a == 0:

            # Change node colors (transition s_2)
            if self.s_4_hide_node[int(self.s_1)] == 0:  # If s_1 not hiding spot
                if self.s_2_node_color[int(self.s_1)] == 0:  # If node is (was) black
                    self.drill_finding = 0
                else:
                    self.drill_finding = 3  # Drill finding = 3, if drilled on unveiled spot (i.e. not black)
                self.s_2_node_color[int(self.s_1)] = 1  # Change color to grey (not a hiding spot)
            elif self.s_4_hide_node[int(self.s_1)] == 1:  # Elif s_1 is hiding spot
                if self.s_2_node_color[int(self.s_1)] == 0:  # If node is (was) black
                    self.drill_finding = 1
                else:
                    self.drill_finding = 3  # Drill finding = 3, if drilled on unveiled spot (i.e. not black)
                self.s_2_node_color[int(self.s_1)] = 2  # Change color to blue (hiding spot)

    def eval_whether_treasure(self):
        """Evaluate whether new current position is the treasure location"""
        if self.s_1 == self.s_3_tr_loc:  # if current position equals treasure location
            self.tr_disc = 1

            # Evaluate whether found on hide
            if self.s_2_node_color[int(self.s_1)] == 2:
                self.tr_found_on_blue = 1
            elif self.s_2_node_color[int(self.s_1)] == 0:
                self.tr_found_on_blue = 0
        else:
            self.tr_disc = 0

    def eval_action(self):
        """Evaluate model action and update affected task states"""

        self.perform_state_transition_f()

        # If participant decides to take a step
        # -----------------------------------------------------
        if self.model.a != 0:

            # Evaluate whether new position is treasure location
            self.eval_whether_treasure()

        # If participant decides to drill (a == 0)
        # -----------------------------------------------------
        else:

            # Update number of node colors
            self.n_black = np.count_nonzero(self.s_2_node_color == 0)
            self.n_grey = np.count_nonzero(self.s_2_node_color == 1)
            self.n_blue = np.count_nonzero(self.s_2_node_color == 2)

            # Update number of hides left
            self.n_hides_left = self.n_hides - self.n_blue
