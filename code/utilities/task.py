import numpy as np
import copy as cp
import os
import json


class Task:
    """A class used to represent the treasure hunt task model

    A task object can interact with an agent object within an
    agent-based behavioral modelling framework

    Parameters
    ----------
    working_dir : str
        Current working directory
    n_rounds : int
        Number of rounds within one block
    dim : int
        Dimensionality of the gridworld
    n_hides : int
        Number of hiding spots

    Attributes
    ----------
    working_dir : str
        Current working directory
    n_rounds : int
        Number of rounds within one block
    n_trials : int
        Number of trials within one round
    dim : int
        Dimensionality of the gridworld
    n_nodes : int
        Number of nodes in the gridworld
    n_hides : int
        Number of hiding spots
    model : object
        Object of class Model, directly linked (i.e. attributes will change, when model attributes change)
    s1_t : int
        State s1, scalar denoting the position in trial t
    s2_t : array_like
        State s2, (25x1)-dimensional array denoting each nodes color (0:black, 1:grey, 2:blue) in trial t
    s3_c : int
        State s3, scalar denoting the treasure location in round c
    s4 : array_like
        State s4, (25x1)-dimensional array denoting each nodes hiding spot status (0:non-hiding spot, 1:hiding spot)
    a_set : array_like
        Action set, (5x1)-dimensional array with all actions available in the treasure hunt task
    o_t : int
        Observation in trial t
    tr_disc_t : int
        Variable denoting whether or not the treasure was found in trial t (0: not found, 1:found)
    hides_loc : array_like
        Hiding spot locations (n_hides x 1)-dimensional array
    n_black : int
        Number of black nodes
    n_grey : int
        Number of grey nodes
    n_blue : int
        Number of blue nodes
    n_hides_left : int
        Number of hiding spots that haven't been unveiled in trial t
    drill_finding : any
        Variable denoting the result of drilling in trial t
    tr_found_on_blue : any
        Variable denoting whether the treasure (if found) was found on a blue node
    shortest_dist_dic : dict
        Shortest distance between two nodes

    Methods
    -------
    evaluate_shortest_distances()
        Evaluates the shortest distances between all nodes
    eval_s_4()
        Evaluates s_4 values according to hides_loc
    start_new_round()
        Resets trial-wise changing attributes to initial values for t=0 in a new round
    start_new_trial()
        Resets dynamic attributes to initial values for each trial t
    return_observations()
        Returns observation given states and action in trial t
    perform_state_transition_f()
        Performs state transitions given prior states and action in trial t
    eval_whether_treasure()
        Evaluates whether the new position s_{t+1} is the treasure location
    eval_action()
        Evaluates the action in trial t
    """

    def __init__(self, working_dir, n_rounds, n_trials, dim, n_hides):
        self.working_dir = working_dir
        self.n_rounds = n_rounds
        self.n_trials = n_trials
        self.dim = dim
        self.n_nodes = self.dim ** 2  # Number of fields
        self.n_hides = n_hides

        # Initialize empty attribute to embed model object of class Model
        self.model = None

        # Initialize task model components
        self.s1_t = np.nan  # Position in trial t
        self.s2_t = np.full(self.n_nodes, 0)  # Node colors in trial t, initial values: 0 = black
        self.s3_c = np.full(1, np.nan)  # Treasure location in round c
        self.s4 = np.full(self.n_nodes, 0)  # Each node's hiding spot status, initial value: 0 = non-hiding spot
        self.a_set = np.array([0, -self.dim, 1, self.dim, -1])  # Set of actions
        self.o_t = np.full(1, np.nan)  # Observation in trial t

        # Initialize variables for computations
        self.tr_disc_t = 0  # treasure discovery at current position, initial value: 0
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
            self.eval_shortest_distances()
            with open(short_dist_fn, 'w') as json_file:
                json.dump(self.shortest_dist_dic, json_file, indent=4)

    def eval_shortest_distances(self):
        """Evaluate the shortest distance between all nodes in grid world with
        dimension dim given all available actions in a_set.
        The shortest path is expressed in numbers of steps needed to
        reach the end node when standing on a given start node
        """
        # ------Initialize variables / objects--------------------------------
        n_nodes = cp.deepcopy(self.n_nodes)  # number of nodes in the graph
        dim = cp.deepcopy(self.dim)  # dimension of the grid world
        moves = cp.deepcopy(self.a_set[:4])  # possible moves / actions

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

    def eval_s_4(self):
        """Evaluate s_4 values according to hides_loc"""
        for node in self.hides_loc:
            self.s4[node] = 1

    def start_new_round(self):
        """Reset dynamic states to initial values for a new round"""
        self.tr_disc_t = 0  # treasure discovery
        self.tr_found_on_blue = np.nan

    def start_new_trial(self):
        """Reset dynamic states to initial values for a new trial"""
        self.drill_finding = np.nan

    def return_observation(self):
        """Return observation, i.e. each nodes current status (color) and treasure disc (yes/no).
        This function maps action, reward and states s3 and s4 onto observation o_t, as specified in g
        """
        # If node color black and no treasure
        if (self.s2_t[self.s1_t] == 0) and (self.tr_disc_t == 0):
            self.o_t = 0

        # If node color = grey (always no treasure found)
        elif self.s2_t[self.s1_t] == 1:
            self.o_t = 1

        # If node color = blue and no treasure
        elif (self.s2_t[self.s1_t] == 2) and (self.tr_disc_t == 0):
            self.o_t = 2

        # If treasure found
        elif self.tr_disc_t == 1:
            self.o_t = 3

    def perform_state_transition_f(self):
        """Perform the state transition function f"""
        # Move to new position (transition s_1)
        self.s1_t += int(self.model.a_t)

        # After informative actions
        if self.model.a_t == 0:

            # Change node colors (transition s_2)
            if self.s4[self.s1_t] == 0:  # If s_1 not hiding spot
                if self.s2_t[self.s1_t] == 0:  # If node is (was) black
                    self.drill_finding = 0
                else:
                    self.drill_finding = 3  # Drill finding = 3, if drilled on unveiled spot (i.e. not black)
                self.s2_t[self.s1_t] = 1  # Change color to grey (not a hiding spot)
            elif self.s4[self.s1_t] == 1:  # Elif s_1 is hiding spot
                if self.s2_t[self.s1_t] == 0:  # If node is (was) black
                    self.drill_finding = 1
                else:
                    self.drill_finding = 3  # Drill finding = 3, if drilled on unveiled spot (i.e. not black)
                self.s2_t[self.s1_t] = 2  # Change color to blue (hiding spot)

    def eval_whether_treasure(self):
        """Evaluate whether new current position is the treasure location"""
        if self.s1_t == self.s3_c:  # if current position equals treasure location
            self.tr_disc_t = 1

            # Evaluate whether found on hide
            if self.s2_t[self.s1_t] == 2:
                self.tr_found_on_blue = 1
            elif self.s2_t[self.s1_t] == 0:
                self.tr_found_on_blue = 0
        else:
            self.tr_disc_t = 0

    def eval_action(self):
        """Evaluate model action and update affected task states"""

        self.perform_state_transition_f()

        # If participant decides to take a step
        # -----------------------------------------------------
        if self.model.a_t != 0:

            # Evaluate whether new position is treasure location
            self.eval_whether_treasure()

        # If participant decides to drill (a == 0)
        # -----------------------------------------------------
        else:

            # Update number of node colors
            self.n_black = np.count_nonzero(self.s2_t == 0)
            self.n_grey = np.count_nonzero(self.s2_t == 1)
            self.n_blue = np.count_nonzero(self.s2_t == 2)

            # Update number of hides left
            self.n_hides_left = self.n_hides - self.n_blue
