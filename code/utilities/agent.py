"""Module classes to create an agent that can interact with the treasure
hunt task.
"""
import os
import time
import pickle
import numpy as np
from utilities.task import Task
from utilities.config import Paths, GridConfigurationParameters, humanreadable_time, DirectoryManager
from utilities.config import DataHandler
from matplotlib import pyplot
from utilities.very_plotter_new import VeryPlotter
from matplotlib.colors import ListedColormap


class HiddenMarkovModel:
    """A Class to compute Bayesian model components given a set of
    task parameters (i.e. no. trials, dimension of the grid, etc.) or
    load model components from disk if the task-parameter-specific components
    already exist on disk. For example, the file utilities/liklh_dim-5_h6.npy
    stores the likelihood array for task configuration with dimensionality 5
    and 6 hiding spots. If that file exists, the array will not be computed to
    save computation time.

    Newly computed model components are written to .npy files are written and
    save in code/utilities/ on disk.

    Attributes:
    -----------
        task_design_params (TaskDesignParameters): Instance of data
            class TaskDesignParameters.

        paths (Paths): Instance of data class Paths.

        s4_perms (list): All permutations of possible values for state s4

        s4_perm_node_indices(dict of int: list): node-specific indices of
            "self.s4_perms" of all those entries that have the value 1 at
                self.s4_perms[key], encoding that node i is a hiding spot
            .key (int): value indicating node in grid
            .value (list): list of indices for "self.s4_perms".

        n_s4_perms (int): number of state s4 permutations

        prior_c0 (np.ndarray): (n_nodes x n_s4_permutaions)-array of prior,
            i.e. initial belief state before any observation

        lklh (np.ndarray): (n_action_type x n_nodes x n_node_colors x
            n_observations x n_nodes x n_s4_permutations)-array of the
                likelihood, i.e. action- and state-dependent, state-conditional
                    observation distribution

    Args:
    -----
        task_design_params (TaskDesignParameters, optional): Instance of data
            class TaskDesignParameters.
    """

    def __init__(self, task_object,
                 grid_config_params=GridConfigurationParameters()):
        self.task: Task = task_object
        self.grid_config = grid_config_params
        self.paths: Paths = Paths()
        self.beta_0: np.ndarray = np.full((self.task.n, 1), np.nan)
        self.Omega: np.ndarray = np.array(np.nan)
        self.Phi: np.ndarray = np.array(np.nan)

    def compute_beta_0(self):
        """Method to evaluate the initial belief state beta in round 1,
        trial 1 given the first state current position"""

        # Set all states, that are in line with current position to 1
        self.beta_0[np.where(self.task.S[:, 0] == self.task.s1_t)[0], 0] = 1
        # Set alle remaining states zero
        self.beta_0[np.where(self.task.S[:, 0] != self.task.s1_t)[0], 0] = 0
        # Normalize belief state
        self.beta_0 = self.beta_0 / sum(self.beta_0)

    def get_comps_old(self):
        """Create or load Bayesian components, i.e. s4-permutations, prior and
        likelihood.
        """
        # Initialize and evaluate s_4 permutations
        s4_perms_fn_pkl = os.path.join(
            self.paths.code, "utilities",
            f"s4_perms_dim-{self.grid_config.dim}_"
            f"h{self.grid_config.n_hides}.pkl")

        if os.path.exists(s4_perms_fn_pkl):
            print("Loading s4_perms ...")
            start = time.time()
            with open(s4_perms_fn_pkl, "rb") as file:
                self.S4 = pickle.load(file)
            end = time.time()
            print(
                " ... finished loading s4_perms, timed needed: "
                f"{humanreadable_time(end-start)}"
                )
        else:
            print("Computing s4_perms ...")
            start = time.time()
            self.S4 = []
            self.eval_s4_perms()
            end = time.time()
            print(f" ... finished computing s4_perms, time needed: "
                  f"{humanreadable_time(end-start)}"
                  )
            print("Saving s4_perms ...")
            start = time.time()
            with open(s4_perms_fn_pkl, "wb") as file:
                pickle.dump(self.S4, file)
            end = time.time()
            print(f" ... finisehd saving s4_perms to files, time needed: "
                  f"{humanreadable_time(end-start)}"
                  )

        # Load/evaluate agent's initial belief state in 1. trial ---(Prior)---
        prior_fn = os.path.join(self.paths.code, "utilities",
                                f"prior_dim-{self.grid_config.dim}_"
                                f"h{self.grid_config.n_hides}.npy")
        if os.path.exists(prior_fn):
            print("Loading prior array from file ...")
            start = time.time()
            self.beta_0 = np.load(prior_fn)
            end = time.time()
            print(f" ... finished loading prior, time needed: "
                  f"{humanreadable_time(end-start)}"
                  )
            # sum_p_c0 = np.sum(self.prior_c0)
        else:
            print("Evaluating prior belief array for given task config ...")
            start = time.time()
            self.beta_0 = np.full((self.grid_config.n_nodes,
                                     self.n_S4), 0.0)
            self.eval_prior()
            end = time.time()
            print(f" ... finished evaluating prior, time needed: "
                  f"{humanreadable_time(end-start)}"
                  )
            print("Saving prior belief array to file ...")
            start = time.time()
            np.save(prior_fn, self.beta_0)
            end = time.time()
            print(f" ... finished saving prior to file, time needed:"
                  f"{humanreadable_time(end-start)}"
                  )

        # Load/eval action-dep. state-cond. obs distribution ---(Likelihood)---
        lklh_fn = os.path.join(
            self.paths.code,
            "utilities",
            f"lklh_dim-{self.grid_config.dim}_"
            f"h{self.grid_config.n_hides}.npy"
        )
        if os.path.exists(lklh_fn):
            print("Loading likelihood array from file ...")
            start = time.time()
            self.Omega = np.load(lklh_fn)
            end = time.time()
            print(f" ... finished loading likelihood array, time needed: "
                  f"{humanreadable_time(end-start)}\n")
        else:
            print("Computing likelihood array for given task config ...")
            self.Omega = np.zeros(
                (2, self.grid_config.n_nodes, 3, 4,
                 self.grid_config.n_nodes, self.n_S4),
                dtype=np.uint16)
            start = time.time()
            self.eval_likelihood()
            end = time.time()
            print(f" ... finished computing likelihood, time needed: "
                  f"{humanreadable_time(end-start)}")
            print("Saving likelihood array to file")
            start = time.time()
            np.save(lklh_fn, self.Omega)
            end = time.time()
            print(f" ... saved likelihood array to file, time needed: "
                  f"{humanreadable_time(end-start)}")
        # start = time.time()
        return self

    def compute_Omega(self):
        """Method to compute Omega"""

        self.Omega = np.full((self.task.n, self.task.m, self.task.p), 0)  # TODO init in init

        node_colors = {"black": 0,
                       "grey": 1,
                       "blue": 2}
        
        # Encode set A to either drill or step
        A = [0, 1]  # 0: drill, 1: step

        for i_a, a in enumerate(A):
            for i_s, s in enumerate(self.task.S):
                for i_o, o in enumerate(self.task.O_):
                    # Extract state components
                    current_pos = int(s[0])  # NOTE: set S[0] := {1, ..., n}
                    node_index_in_o_t = current_pos  # NOTE: bc o[0] is tr flag
                    tr_location = int(s[1])
                    hiding_spots = s[2:]

                    # Extract observation components
                    tr_flag = o[0]

                    # TODO: alle observations, wo node colors mit hiding spot
                    #  locations keinen sinn machen
                    # siehe Handwritten NOTE 24.11.

                    # -------After DRILL actions: -----------------------------
                    if a == 0:

                        # CONDITION:                        CORRESP MODEL VARIABLE:
                        # ---------------------------------------------------------
                        # if new position...                                s[1]
                        # ...IS NOT treasure location                       s[2]
                        # ...IS NOT hiding spot,                            s[3:]
                        # all observation, for which...
                        # ...tr_flag == 0,                                  o[1]
                        # ...and new node color == grey,                    o[2:]
                        #  = 1
                        if (
                                current_pos != tr_location
                                and current_pos not in hiding_spots
                                and tr_flag == 0
                                and o[node_index_in_o_t] == node_colors["grey"]):
                            self.Omega[i_s, i_o, i_a] = 1

                        # CONDITION:                        CORRESP MODEL VARIABLE:
                        # ---------------------------------------------------------
                        # if new position...                                s[1]
                        # ...IS NOT treasure location                       s[2]
                        # ...IS hiding spot,                                s[3:]
                        # all observation, for which...
                        # ...tr_flag == 0,                                  o[1]
                        # ...and new node color == blue,                    o[2:]
                        #  = 1
                        if (
                                current_pos != tr_location
                                and current_pos in hiding_spots
                                and tr_flag == 0
                                and o[node_index_in_o_t] == node_colors["blue"]):
                            self.Omega[i_s, i_o, i_a] = 1

                        # All other observaton probabilites remain 0 as initiated.

                    # -------After STEP actions: -----------------------------
                    else:  # if a != 0
                        # CONDITION:                        CORRESP MODEL VARIABLE:
                        # ---------------------------------------------------------
                        # if new position...                                s[1]
                        # ...IS NOT treasure location                       s[2]
                        # ...IS NOT hiding spot,                            s[3:]
                        # all observation, for which...
                        # ...tr_flag == 0,                                  o[1]
                        # ...and new node color in ["black", "grey"],       o[2:]
                        #  = 1
                        if (
                                current_pos != tr_location
                                and current_pos not in hiding_spots
                                and tr_flag == 0
                                and o[node_index_in_o_t] in [node_colors["black"],
                                                    node_colors["grey"]]):
                            self.Omega[i_s, i_o, i_a] = 1

                        # CONDITION:                        CORRESP MODEL VARIABLE:
                        # ---------------------------------------------------------
                        # if new position...                                s[1]
                        # ...IS NOT treasure location                       s[2]
                        # ...IS hiding spot,                                s[3:]
                        # all observation, for which...
                        # ...tr_flag == 0,                                  o[1]
                        # ...and new node color in ["black", "blue"],       o[2:]
                        #  = 1
                        if (
                                current_pos != tr_location
                                and current_pos in hiding_spots
                                and tr_flag == 0
                                and o[node_index_in_o_t] in [node_colors["black"],
                                                    node_colors["blue"]]):
                            self.Omega[i_s, i_o, i_a] = 1

                        # CONDITION:                        CORRESP MODEL VARIABLE:
                        # ---------------------------------------------------------
                        # if new position...                                s[1]
                        # ...IS treasure location                           s[2]
                        # ...IS hiding spot,                                s[3:]
                        # all observation, for which...
                        # ...tr_flag == 1,                                  o[1]
                        # ...and new node color in ["black", "blue"],       o[2:]
                        #  = 1
                        if (
                                current_pos == tr_location
                                and current_pos in hiding_spots
                                and tr_flag == 1
                                and o[node_index_in_o_t] in [node_colors["black"],
                                                    node_colors["blue"]]):
                            self.Omega[i_s, i_o, i_a] = 1

    def comput_Phi(self):
        "Method to compute Phi"

        action_set = [0,
                      - self.grid_config.dim, + 1,
                      + self.grid_config.dim, -1]

        self.Phi = np.full((self.task.n, self.task.n, len(action_set)), np.nan)

        # ---------------------------------------
        # NOTE:
        # s = (s_1, s_2, s_3, s_4)
        # p^{a_t = a}(s_{t+1} = s|s_t = s_tilde)
        # s_1 is the first component of s = s_{t+1}
        # ---------------------------------------

        for a_i, a in enumerate(action_set):

            # Iterate possible old states s_{t}
            for s_tilde_i, s_tilde in enumerate(self.task.S):

                # Iterate possible new states s_{t+1}
                for s_i, s in enumerate(self.task.S):

                    s_1 = s[0]              # current position in t + 1
                    s_1_tilde = s_tilde[0]  # current position in t
                    s_1_tilde_plus_a = s_1_tilde + a

                    # Filter out forbidden steps (wald outside border)
                    if (1 <= s_1_tilde_plus_a <= self.grid_config.n_nodes
                            and not ((((s_1_tilde - 1)
                                       % self.grid_config.dim) == 0)
                                     and a == -1)
                            and not ((((s_1_tilde)
                                       % self.grid_config.dim) == 0)
                                     and a == 1)
                            ):
                        if s_1 == s_1_tilde_plus_a:
                            self.Phi[s_tilde_i, s_i, a_i] = 1
                        
                        elif s_1 != s_1_tilde_plus_a:
                            self.Phi[s_tilde_i, s_i, a_i] = 0

                    # tODO: Alle unerlaubten action
                    # agent glaubt, dass er auf dem Feld stehen bleibt
                    else:
                        if s_1 == s_1_tilde:
                            self.Phi[s_tilde_i, s_i, a_i] = 1
                        elif s_1 != s_1_tilde:
                            self.Phi[s_tilde_i, s_i, a_i] = 0

    def plot_color_map(self, n_nodes, n_hides, **arrays):

        dir_mgr = DirectoryManager()

        for key, array in arrays.items():
            fig_fn = f"{key}-{n_nodes}-nodes_{n_hides}-hides"

            # Preapre figure
            plotter = VeryPlotter(paths=dir_mgr.paths)
            plt = pyplot

            rc_params = plotter.define_run_commands()
            plt = pyplot
            plt.rcParams.update(rc_params)
            fig, ax = plt.subplots(figsize=(11, 5))

            # Create a custom discrete colormap
            cmap = ListedColormap(['darkgrey', 'darkcyan'])
            image = ax.matshow(array, cmap=cmap)

            # Add colorbar
            cbar = plt.colorbar(image, ticks=[0, 1], shrink=0.4)

            # Save or display the plot
            plotter.save_figure(fig=fig, figure_filename=fig_fn)

    def compute_or_load_components(self):
        """Create or load aHMM components, i.e. matrices of probability distributions.
        """

        # TODO: load if existing
        # -----blueprint:-------------------------------------------------
        # if os.path.exists(s4_perms_fn_pkl):
        #     print("Loading s4_perms ...")
        #     start = time.time()
        #     with open(s4_perms_fn_pkl, "rb") as file:
        #         self.S4 = pickle.load(file)
        #     end = time.time()
        #     print(
        #         " ... finished loading s4_perms, timed needed: "
        #         f"{humanreadable_time(end-start)}"
        #         )


        # # Load/evaluate initial belief state beta_1_0 -------------------------
        # print("Computing beta_1_0 for given task config ...")
        # start = time.time()
        # self.compute_beta_0()
        # end = time.time()
        # print(f" ... finished computing beta_1_0, time needed: "
        #       f"{humanreadable_time(end-start)}"
        #       )
        # 
        # start = time.time()
        # self.save_arrays(beta_0_1=self.beta_0)
        # end = time.time()
        # print(f" ... finisehd saving beta_0_1 to files, time needed: "
        #       f"{humanreadable_time(end-start)}"
        #       )
        data_handler = DataHandler(paths=self.paths)

        # Load/evaluate Omega-------------------------
        print("Computing Omega for given task config ...")
        start = time.time()
        self.compute_Omega()
        end = time.time()
        print(f" ... finished computing Omega, time needed: "
              f"{humanreadable_time(end-start)}")
        start = time.time()
        data_handler.save_arrays(n_nodes=self.grid_config.n_nodes,
                    n_hides=self.grid_config.n_hides,
                    Omega_dill=self.Omega[:, :, 0])
        data_handler.save_arrays(n_nodes=self.grid_config.n_nodes,
                    n_hides=self.grid_config.n_hides,
                    Omega_step=self.Omega[:, :, 1])
        end = time.time()
        print(f" ... finisehd saving Omega to files, time needed: "
              f"{humanreadable_time(end-start)}"
              )
        self.plot_color_map(n_nodes=self.grid_config.n_nodes,
                            n_hides=self.grid_config.n_hides,
                            Omega_drill=self.Omega[:, :, 0],
                            Omega_step=self.Omega[:, :, 1])

        # Load/evaluate Phi-------------------------
        print("Computing Phi for given task config ...")
        start = time.time()
        self.comput_Phi()
        end = time.time()
        print(f" ... finished computing Phi, time needed: "
              f"{humanreadable_time(end-start)}")
        start = time.time()
        data_handler.save_arrays(n_nodes=self.grid_config.n_nodes,
                    n_hides=self.grid_config.n_hides,
                    Phi_drill=self.Phi[:, :, 0],
                    Phi_minus_dim=self.Phi[:, :, 1],
                    Phi_plus_one=self.Phi[:, :, 2],
                    Phi_plus_dim=self.Phi[:, :, 3],
                    Phi_minus_one=self.Phi[:, :, 4])
        end = time.time()
        print(f" ... finisehd saving Phi to files, time needed: "
              f"{humanreadable_time(end-start)}"
              )

        self.plot_color_map(n_nodes=self.grid_config.n_nodes,
                            n_hides=self.grid_config.n_hides,
                            Phi_drill=self.Phi[:, :, 0],
                            Phi_minus_dim=self.Phi[:, :, 1],
                            Phi_plus_one=self.Phi[:, :, 2],
                            Phi_plus_dim=self.Phi[:, :, 3],
                            Phi_minus_one=self.Phi[:, :, 4])

        return self


class AgentAttributes:
    """
    A class to store necessary agent-specific attributes. Instance of this
    class is needed to create an agent class instance

    Attributes:
    -----------
        name (str): Agent model name, e.g. "C1" or "A1"
        is_bayesian (bool): True if agent model is bayesian, i.e. belief state
            based. False otherwise.
        is_explorative (bool): True if agent is explorative. False otherwise.
    
    Args:
    -----
        agent_model_name (str): Agent model name, e.g. "C1" or "A1"
    """
    is_bayesian: bool
    is_explorative: bool

    def __init__(self, agent_model_name: str):
        self.name = agent_model_name
        self.define_attributes()

    def define_attributes(self):
        """Define agent specific attributes"""
        # Control models
        if self.name in ['C1', 'C2', 'C3']:
            self.is_bayesian = False
            self.is_explorative = False

        # Bayesian models
        elif self.name == 'A1':
            self.is_bayesian = True
            self.is_explorative = False

        # Bayesian models using explorative strategy
        elif self.name in ['A2', 'A3']:
            self.is_bayesian = True
            self.is_explorative = True


class Agent:
    """A class used to represent an agent model.
    An agent object can interact with a task object within an
    agent-based behavioral modelling framework

    Attributes:
    ----------
        agent_attr (AgentAttributes): Object storing agent attributes
        task_object (Task): Task object
        lambda_ (float): Scalar value representing this agent's weighting
            parameter, if applicable
        moves (int): Remaining number of moves

        a_s1 (np.ndarray): (1 x n_possible_actions)-array representing the
            state-dependent set of actions. Varying size.
        o_s2 (list): State-dependet set of possible observations. Varying size.

        valence_t (np.ndarray): (1 x n_a_s1)-array of action valences

        decision_t (np.ndarray): (1 x 1)-array of current decision

        marg_s3_b (np.ndarray): (1 x n_nodes)-array of marginal posterior
            belief state over possible values for s_3
        marg_s3_prior (np.ndarray): (1 x n_nodes)-array of marginal posterior
            belief state over possible values for s_3
        marg_s4_b (np.ndarray): (1 x n_nodes)-array representing each node's
            probability of being a hiding spot, based on posterior belief state
        marg_s4_prior (np.ndarray): (1 x n_nodes)-array representing each
            node's probability of being a hiding spot, based on prior belief
                state
        TODO: marg_s4_prior umbenennen. technically not the marginal distr.

        bayes_comps(BayesianModelComps): Bayesian model components

        prior_c (np.ndarray): (n_nodes x n_s4_permutations)-array representing
            the prior belief state.

        p_s_giv_o: (n_nodes x n_s4_permutations)-array representing
            the posterior belief state.

        max_s3_b_value (float): Scalar representing the currently maximum value
            of marginal s3 belief state.

        rounded_marg_s3_b (float): Scalar representing the rounded current
            maximum value of marginal s3 belief state.
        max_s3_b_nodes (TODO): all nodes for which marginal belief state equals
            the current max marginal s3 belief state value

        dist_to_max_s3_b_nodes (TODO): Distances to nodes with maximum marginal
            belief state 
        shortest_dist_to_max_s3_b (TODO): Shortest distance to one of the nodes
            with maximum belief state
        closest_max_s3_b_nodes_i_s (TODO): TODO
        closest_max_s3_b_nodes (TODO): TODO

        p_o_giv_o ((n_a_s1 x n_obs)-np.ndarray): (n_a_s1 x n_obs)-array representing the
            predictive posterior distribution
        kl_giv_a_o (np.ndarray): (n_a_s1 x n_obs)-array representing the KL
            divergences of respective virtual belief states and current
                posterior
        virt_b (dict of TODO): (TODO mapping) of virtual belief states

    Notes:
    -----
        belief state refers to the probability distribution p(s^3,s^4|o_{1:t}).
        likelihood (as part of bayesian model component) refers to the
        deterministic action- and state-dependent, state-conditional ob-
        servation distribution p(o|s^3, s^4)

    Args:
    ------
        agent_attr (AgentAttributes): Object storing agent attributes
        task_object (Task): Task object
        lambda_ (float): This agent's weighting parameter, if applicable
    """

    def __init__(self, agent_attr: AgentAttributes,
                 task_object: Task, lambda_):

        self.agent_attr = agent_attr
        self.task: Task = task_object
        self.lambda_ = lambda_

        # Initialize dynamic agent attributes
        self.a_s1: np.ndarray = np.array(np.nan)  # state-dependent action-set
        self.o_s2 = []  # state-dep observation-set

        # Initialize arrays for agent's decision valences and decisionsn
        self.valence_t = np.full(5, np.nan)  # decision valences
        self.decision_t = np.full(1, np.nan)  # decision

        # Initialize belief state objects
        self.marg_tr_belief = np.full(self.task.grid.n_nodes, np.nan)
        self.marg_hide_belief = np.full(self.task.grid.n_nodes, np.nan)
        self.marg_tr_belief_prior = np.full(self.task.grid.n_nodes, np.nan)
        self.marg_hide_belief_prior = np.full(self.task.grid.n_nodes, np.nan)

        if self.agent_attr.is_bayesian:
            # Unpack bayesian beh_model components
            self.hmm_matrices: HiddenMarkovModel = HiddenMarkovModel(task_object)

            # ---(Prior, c != 0)---
            self.p_s_giv_o_prior_new_c: np.ndarray = np.array(np.nan)
            # ---(Posterior)---
            self.p_s_giv_o_post: np.ndarray = np.array(np.nan)

        # Initialize closest max s3 node variables for computations
        self.max_s3_b_value = np.nan
        self.rounded_marg_s3_b = np.full(self.task.grid.n_nodes, np.nan)
        self.max_tr_b_node_indices = np.nan
        self.dist_to_max_s3_b_nodes = np.nan
        self.shortest_dist_to_max_s3_b = np.nan
        self.closest_max_s3_b_nodes_i_s: np.ndarray = np.array(np.nan)
        self.closest_max_s3_b_nodes: np.ndarray = np.array(np.nan)

        # Initialize p_o_giv_o and kl objects
        self.p_o_giv_o: np.ndarray = np.array(np.nan)
        self.kl_giv_a_o: np.ndarray = np.array(np.nan)
        self.virt_b = {}

    def attach_hmm_matrices(
            self,):
        """Method to load or create prior, likelihood and permutation lists etc

        Args:
        ------
        bayesian_comps (BayesianModelComps): Object storing bayesian model
            components
        """
        self.hmm_matrices = HiddenMarkovModel(self.task, self.task.grid)
        self.hmm_matrices.compute_or_load_components()

    def eval_prior_subs_rounds(self):
        """Reset belief states for s3 (treasure) based on marginal s4
        (hiding spot) beliefs. This is necessary so the prior of the first
        trial of a new round (that is not c=0) withholds information about
        currently already revealed hiding spots"""
        # TODO: major todo
        # Initialize all as zero
        self.p_s_giv_o_prior_new_c = np.full(
            (self.hmm_matrices.task.n, 1), 0.)

        # marg_s4_perm_b = np.full(self.n_s4_perms, np.nan)
        # for s4_perm in range(self.n_s4_perms):
        #     marg_s4_perm_b[s4_perm] = self.p_s_giv_o[:, s4_perm].sum()

        # TODO: how to evaluate marginal probs now??!
        # -------------------------------------------
        # THIS is the old code for marginal probs:
        # marg_s4_perm_b = self.p_s_giv_o_post.sum(axis=0)
        # sum_prob_s4_perm = marg_s4_perm_b[:].sum()

        # for s_3 in range(self.task.task_params.n_nodes):
        #     self.p_s_giv_o_prior[
        #         s_3,
        #         self.hmm_matrices.S4_incl_nodes_indices[s_3]
        #         ] = marg_s4_perm_b[
        #             self.hmm_matrices.S4_incl_nodes_indices[s_3]
        #             ] * (1 / self.task.task_params.n_hides)

        # Uncomment for DEBUGGING to track evaluation of posteriors (TODO: old!)
        # ------------------------------------------------------
        # # Evaluate marginal treasure distribution
        # marg_s3_b = np.full(self.task.n_nodes, np.nan)
        # for s3 in range(self.task.n_nodes):
        #     marg_s3_b[s3] = self.prior_c[s3, :].sum()
        # # sum_prob_tr = marg_s3_b[:].sum()
        #
        # # Evaluate marginal hiding spot distribution
        # marg_s4_b = np.full(self.task.n_nodes, np.nan)
        # for node in range(self.task.n_nodes):
        #     marg_s4_b[node] = self.prior_c[:, self.s4_perm_node_indices[
        #                                           node]].sum()
        # sum_prob_hides = marg_s4_b[:].sum()
        # set_debug_breakpoint = 'here'

    def eval_p_s_giv_o_post(self,
                            beta_prior,
                            a_t, o_t
                            ) -> np.ndarray:
        """Method to evaluate posterior belief state given prior, action, s1
        and observation

        Args:
        ----
            prior_belief_state (np.ndarray): (n_nodes x n_s4_perms)-array of
                the prior belief state.
            action (int): Current action
            s_1 (int): Current state s1 value
            s2_giv_s1 (int): Current state s2[s1] value, i.e. node color
                on current position s1
            obs (int): Current observation

        Returns:
        -------
            np.ndarray: (n_nodes x n_s4_perms)-array of
                the posterior belief state.
        """

        # Determine action type (step or drill)
        if a_t != 0:  # if step
            a_t_type = 1
        else:  # if drill
            a_t_type = 0
        a_t_type = int(a_t_type)

        # Determine observation dependent omega index j
        O_ = self.task.O_
        j = int(np.where(np.all(O_ == o_t, axis=1))[0])

        # Extract components  # TODO: kopieren kostet working memory
        Omega_j = self.hmm_matrices.Omega[:, j, a_t_type][:, np.newaxis]
        Phi_k = self.hmm_matrices.Phi[:, :, a_t]

        # TODO: still necessary?
        if np.sum(beta_prior * Omega_j) == 0:

            beta_post = (
                beta_prior * Omega_j)
            print('sum of prior * lklh = 0, leaving out normalization')
            # debug = 'here'

        else:
            # TODO: add state transition!

            beta_post = Omega_j * np.matmul(Phi_k.T, beta_prior)
            beta_post = beta_post / sum(beta_post)

        return beta_post

    def eval_marg_beliefs(self, belief):

        """Method to compute marginal posterior distributions for each node to
        the treasure location or hiding spot

        Args:
        -----
            belief (np.ndarray): (n_nodes x n_s4_permutations)-array
                representing a belief state distribution

        Returns:
        -------
            (np.ndarray, np.ndarray): marginal belief states

            TODO: marg_s4 not actually margianl distribution.
        """
        # Evaluate marginal treasure distribution
        marg_treasure_belief = np.full(self.task.grid.n_nodes, np.nan)
        for node in range(self.task.grid.n_nodes):
            possible_tr_loc = node + 1
            tr_index_in_state_vector = 1

            tr_spec_indices = np.where(
                self.task.S[
                    :, tr_index_in_state_vector
                    ] == possible_tr_loc
                )[0]

            marg_treasure_belief[node] = belief[tr_spec_indices, :].sum()

        # Evaluate marginal hiding spot distribution
        marg_hides_belief = np.full(self.task.grid.n_nodes, np.nan)
        for node in range(self.task.grid.n_nodes):

            possible_hide_loc = node + 1
            hide_indices_in_state_vector = range(
                2,  2 + self.task.grid.n_hides)

            hide_spec_indices = np.where(
                np.any(
                    self.task.S[
                        :, hide_indices_in_state_vector
                        ] == possible_hide_loc, axis=1
                    )
                )[0]
            marg_hides_belief[node] = belief[hide_spec_indices, :].sum()

        # Uncomment for debugging
        # sum_prob_hides = marg_s4_b[:].sum()  # should evaluate to ~ n_hides
        # sum_prob_treasure = marg_s3_b[:].sum()  # should evaluate to ~ 1

        return marg_treasure_belief, marg_hides_belief

    def start_new_trial(self):
        """Reset dynamic states to initial values for a new trial"""
        self.valence_t = np.full(5, np.nan)  # decision valences
        self.decision_t = np.full(1, np.nan)  # decision

    def start_new_round(self, round_number):
        """Reset dynamic states to initial values for a new round"""
        # Reset belief states if not first round, if bayesian agent
        if self.agent_attr.is_bayesian and round_number > 0:
            self.eval_prior_subs_rounds()

        # Uncomment for debugg
        # # Marginal prior distributions
        # if self.is_bayesian:
        #     if self.c == 0:
        #         self.marg_s3_prior, self.marg_s4_prior = \
        #             self.eval_marg_b(self.prior_c0)
        #     else:
        #         self.marg_s3_prior, self.marg_s4_prior = \
        #             self.eval_marg_b(self.prior_c)

    def update_belief_state(self, current_action):
        """Evaluate posterior belief state

        Args:
        -----
            current_action (int): current action
        """

        # Only update if agent is bayesian, TODO: outsouce?
        if self.agent_attr.is_bayesian:

            # ------ Define action and prior depending on current trial--------

            # If first trial of task (i.e. before any action) use initial prior
            # ...and step action (a=1).
            if self.task.c == 0 and self.task.t == 0:
                action = 1
                self.hmm_matrices.compute_beta_0()
                prior = self.hmm_matrices.beta_0
            
            # If first trial in a new round, i.e. before any action,  pior_c
            # ...which is the posterior of preceding round's last trial as
            # ...prior and step action (a=1).
            elif self.task.t == 0:
                action = 1
                prior = self.p_s_giv_o_prior_new_c

            # For all remaining trial use current action and posterior of
            else:
                action = current_action
                prior = self.p_s_giv_o_post

            # ------Evaluate posterior----------------------------
            self.p_s_giv_o_post = self.eval_p_s_giv_o_post(
                beta_prior=prior,
                a_t=action,
                o_t=self.task.o_t)

            # ------ Evaluate marginal distributions-------------
            # TODO: hier weiter!!
            start = time.time()
            self.marg_tr_belief, self.marg_hide_belief = self.eval_marg_beliefs(self.p_s_giv_o_post)
            end = time.time()
            print(
                f"time needed for marg_b: {humanreadable_time(end-start)}")

    def identify_a_giv_s1(self):
        """Identify state s1 dependent action set"""
        self.a_s1 = self.task.A

        for action in self.task.A:
            new_s1 = action + self.task.s1_t
            # Remove forbidden steps (walk outside border)
            if (not (1 <= new_s1 <= self.task.grid.n_nodes)
                    or (((self.task.s1_t - 1)
                         % self.task.task_configs.params.dim == 0)
                        and action == -1)
                    or (((self.task.s1_t)
                         % self.task.task_configs.params.dim == 0)
                        and action == 1)):

                self.a_s1 = self.a_s1[self.a_s1 != action]

    def identify_o_giv_s2_marg_s3(self, node, action):
        """Identify state s2 dependent observation set
        
        Args:
        -----
            node (int): scalar value representing the node of interest
            action (int): scalar value representing action
        """
        if action == 0:
            if self.task.node_colors[node] == 0:
                if np.around(self.marg_hide_belief[node], 10) == 0:
                    self.o_s2 = [1]
                elif np.around(self.marg_hide_belief[node], 10) == 1:
                    self.o_s2 = [2]
                else:
                    self.o_s2 = [1, 2]
            elif self.task.node_colors[node] == 1:
                self.o_s2 = [1]
            elif self.task.node_colors[node] == 2:
                self.o_s2 = [2]
        elif action == 1:
            if self.task.node_colors[node] == 0:
                if np.around(self.marg_tr_belief[node], 10) == 0:
                    self.o_s2 = [0]
                else:
                    self.o_s2 = [0, 3]
            elif self.task.node_colors[node] == 1:
                self.o_s2 = [1]
            elif self.task.node_colors[node] == 2:
                if np.around(self.marg_tr_belief[node], 10) == 0:
                    self.o_s2 = [2]
                else:
                    self.o_s2 = [2, 3]

    def eval_closest_max_s3_b_nodes(self):
        """Identify nodes with maximum s3 belief state values"""
        # Identify maximum s3 belief state value
        self.max_s3_b_value = np.around(np.amax(self.marg_tr_belief), 10)

        # Find all nodes with maximum belief state value
        self.rounded_marg_s3_b = np.around(self.marg_tr_belief, 10)
        self.max_tr_b_node_indices = np.where(
            self.rounded_marg_s3_b == self.max_s3_b_value)[0]

        # Evaluate shortest distances to max_s3_nodes
        self.dist_to_max_s3_b_nodes = np.full(len(self.max_tr_b_node_indices), np.nan)
        for index, node in np.ndenumerate(self.max_tr_b_node_indices):
            self.dist_to_max_s3_b_nodes[index] = \
                self.task.shortest_dist_dic[f'{int(self.task.s1_t)}_to_{node + 1}']
        self.shortest_dist_to_max_s3_b = np.amin(self.dist_to_max_s3_b_nodes)
        self.closest_max_s3_b_nodes_i_s = np.where(
            self.dist_to_max_s3_b_nodes == self.shortest_dist_to_max_s3_b)[0]
        self.closest_max_s3_b_nodes = self.max_tr_b_node_indices[
            self.closest_max_s3_b_nodes_i_s]

    def eval_p_o_giv_o(self):
        """Evaluate agent's belief state-dependent posterior predictive
        distribution
        
        Note:
        -----
            Resulting self.p_o_giv_o is a (n_a_s1 x n_obs)-array"""

        # Evaluate p_0 with likelihood
        self.p_o_giv_o = np.full((len(self.a_s1), 4), 0.)

        # Map action
        for i, action in enumerate(self.a_s1):

            new_s1 = self.task.s1_t + action

            # Convert action value to 1, if step action
            if action != 0:
                action = 1

            # Identify possible observations on new_s1
            self.identify_o_giv_s2_marg_s3(new_s1, action)

            for obs in self.o_s2:
                # product_a_o.shape = (25 x 177100)
                product_a_o = (self.p_s_giv_o_post
                               * self.hmm_matrices.Omega[action, new_s1,
                                                       self.task.node_colors[new_s1],
                                                       obs, :, :])

                sum_prod = np.sum(product_a_o)
                # self.p_o_giv_o[i, obs] = (self.p_s_giv_o
                #                * self.bayes_comps.lklh[action, new_s1,
                #                                        self.task.s2_t[new_s1],
                #                                        obs, :, :]).sum()
                self.p_o_giv_o[i, obs] = sum_prod

        # --- Tests ------------------------
        # # Try sparse tensor
        # sparse_lklh = torch.from_numpy(
        #     self.bayes_comps.lklh
        # ).to_sparse()

        # # Try tensor dot multiplication
        # test_tensor_dot_array = np.tensordot(
        #     self.p_s_giv_o, self.bayes_comps.lklh, axes=([0, 1], [4, 5]))
        stop = "here"

    @staticmethod
    def evaluate_kl_divergence(p_x, q_x):
        """Evaluate kl divergence between two distributions
        
        Args:
        -----
            p_x (np.ndarray): (n_nodes x n_s4_permutations)-array representing
                the virtual posterior belief state
            q_x (np.ndarray): (n_nodes x n_s4_permutations)-array representing
                the current posterior belief state.
        """
        # # Evaluate directly
        # start = time.time()
        # log = np.log(p_x / q_x)
        # # end_log = time.time()
        # # print(f'#compute log: {end_log - start}')
        # # start_check = time.time()
        # if np.any(np.isfinite(log)):
        #     # end_check = time.time()
        #     # print(f'Check if any finite: {end_check - start_check}')
        #     # start_masked = time.time()
        #     kl_mask = np.sum(p_x * np.ma.masked_invalid(log))
        #     # end_mask = time.time()
        #     # print(f'Compute masked kl: {end_mask-start_masked}')
        # else:
        #     kl_mask = 0.
        # end = time.time()
        # print(f'Using ma.masked total: {end - start}')

        # # Evaluate after replacing zeros with nans
        # start = time.time()
        # quotient_matrix = p_x / q_x
        # quotient_matrix[np.where(quotient_matrix == 0)] = np.nan Mask
        # zeros with nan
        # log_quot = np.nan_to_num(np.log(quotient_matrix))
        # kl = np.nansum(p_x * log_quot)
        # end = time.time()
        # print(f'After masking zeros: {end - start}')

        # Ensure KL is zero, if virtual postior and current belief state are
        # equal
        if np.all(np.around(p_x, 10) == np.around(q_x, 10)):
            kl = 0.

        # or sum over all dimensions of virtual belief for potential obs after
        # given action are zero, i.e. the agent is knows with certainty that
        # potential observation is inprobable (impossible), e.g. treasure on
        # grey field
        elif np.sum(np.sum(p_x)) == 0:
            kl = 0.

        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                kl = np.sum(p_x * np.ma.masked_invalid(np.log(p_x / q_x)))
                # TODO: masked value troublesome

        return kl

    def eval_kl(self):
        """Evaluate KL divergence between the agent's belief state on trial t
        and its virtual belief state on trial t + 1 given observation o_t
        and action

        NOTE:
        -----
        to save computation time kl is only evaluated if p_o_giv_o != 0,
        else, kl == 0"""

        self.kl_giv_a_o = np.full((len(self.a_s1), 4), 0.)
        self.virt_b = {0: {0: 0., 1: 0., 2: 0., 3: 0.}}

        for i, action in enumerate(self.a_s1):

            new_s1 = self.task.s1_t + action

            # Convert action value to 1, if step action
            if action != 0:
                action = 1

            self.virt_b[i] = {0: 0.}

            # Identify possible observations
            self.identify_o_giv_s2_marg_s3(new_s1, action)

            for obs in self.o_s2:
                # Evaluate virt belief state on t+1 giv potential o and a
                self.virt_b[i][obs] = self.eval_p_s_giv_o_post(  # type: ignore
                    self.p_s_giv_o_post, action, new_s1,
                    self.task.node_colors[new_s1], obs)

                # Evaluate KL divergence
                self.kl_giv_a_o[i, obs] = self.evaluate_kl_divergence(
                    self.virt_b[i][obs], self.p_s_giv_o_post)

    def evaluate_action_valences(self):
        """Evaluate action valences"""

        remaining_moves = self.task.grid.n_trials - self.task.t - 1
        # TODO: trial count weird?


        # 'C1' Valence for random choice agent
        # ---------------------------------------------------------------------
        if self.agent_attr.name == 'C1':
            # Allocate equal valences over all available actions
            self.valence_t[:] = 1 / len(self.a_s1)

        # 'C2' Valence for random exploiter agent
        # ---------------------------------------------------------------------
        if self.agent_attr.name == 'C2':
            # Allocate equal valences over all avail. actions (excl. drill)
            self.valence_t[:] = 1 / (len(self.a_s1) - 1)

            # Set valence for drill action to zero  # TODO: elegantere Lösung!
            self.valence_t[np.where(self.a_s1 == 0)] = 0

        # 'C3' Valence for 50% random exploit and 50% random explore beh_model
        # ---------------------------------------------------------------------
        if self.agent_attr.name == 'C3':
            # Allocate 50% times equal valences for all avail. actions (excl.
            # drill)
            self.valence_t[:] = 1 / 2 * (1 / (len(self.a_s1) - 1))

            # Set drill valence to 50%
            self.valence_t[np.where(self.a_s1 == 0)] = 1 / 2

        # 'A1' belief state-based, exploit, max. immediate reward (LOOK-AHEAD)
        # --------------------------------------------------------------------------
        if self.agent_attr.name == 'A1':

            self.valence_t[:] = 0

            # Iterate over possible actions (i.e. state-dependent actions)
            for index, action_i in enumerate(self.a_s1):
                # Anticipate new possible new position
                new_s1 = self.task.s1_t + action_i

                for close_max_s3_node in self.closest_max_s3_b_nodes:
                    current_dist_to_max_belief = self.task.shortest_dist_dic[
                        f'{int(self.task.s1_t)}_to_{close_max_s3_node + 1}']
                    new_dist_to_closest_max_beliefs = \
                        self.task.shortest_dist_dic[
                            f'{int(new_s1)}_to_{close_max_s3_node + 1}']
                    if remaining_moves >= new_dist_to_closest_max_beliefs \
                            < current_dist_to_max_belief:
                        self.valence_t[index] += self.marg_tr_belief[
                            close_max_s3_node]

            # Set drill action to minus value
            # ( --> to always have lower value than zero)
            # self.v[np.where(self.a_s1 == 0)] = -1  # Needed,
            # otherwise A1 will drill in last trials
            # Let agent stop drilling, if node is not black or if last round
            if self.task.c == (
                    self.task.task_configs.params.n_rounds - 1):
                # or self.task.s_2_node_color[self.task.s_1] != 0:
                self.valence_t[np.where(self.a_s1 == 0)] = -1

        # 'A2' pure explorer agent
        # --------------------------------------------------------------------------
        if self.agent_attr.name == 'A2':

            self.valence_t = (self.p_o_giv_o * self.kl_giv_a_o).sum(axis=1)

            # Let agent stop drilling, if node is not black or if last round
            if self.task.c == (
                    self.task.task_configs.params.n_rounds - 1):
                # or self.task.s_2_node_color[self.task.s_1] != 0:
                self.valence_t[np.where(self.a_s1 == 0)] = -1

        # 'A3' belief state based explorer-exploit agent (LOOK-AHEAD)
        # ---------------------------------------------------------------------
        if self.agent_attr.name == 'A3':

            self.valence_t[:] = 0

            # Iterate over possible actions (i.e. state-dependent actions)
            for i, action_i in enumerate(self.a_s1):
                # Anticipate new possible new position
                new_s1 = self.task.s1_t + action_i

                for close_max_s3_node in self.closest_max_s3_b_nodes:
                    current_dist_to_max_belief = self.task.shortest_dist_dic[
                        f'{int(self.task.s1_t)}_to_{close_max_s3_node + 1}']
                    new_dist_to_closest_max_beliefs = \
                        self.task.shortest_dist_dic[
                            f'{int(new_s1)}_to_{close_max_s3_node + 1}']
                    if remaining_moves >= new_dist_to_closest_max_beliefs \
                            < current_dist_to_max_belief:
                        self.valence_t[i] += (
                            1 - self.lambda_
                            ) * self.marg_tr_belief[close_max_s3_node]

            # Add information value

            for i, action_i in np.ndenumerate(self.a_s1):
                # Move to next loop, if a == 0, i.e. i == 0,
                # because p_o[0] is already filled
                # if action == 0:
                #     continue
                self.valence_t[i] += self.lambda_ * (
                    self.p_o_giv_o[i][0] * self.kl_giv_a_o[i][0] +
                    self.p_o_giv_o[i][1] * self.kl_giv_a_o[i][1] +
                    self.p_o_giv_o[i][2] * self.kl_giv_a_o[i][2] +
                    self.p_o_giv_o[i][3] * self.kl_giv_a_o[i][3])

            self.valence_t += (self.p_o_giv_o * self.kl_giv_a_o).sum(axis=1)

            # Let agent stop drilling, if node is not black or if last round
            if self.task.c == (
                    self.task.task_configs.params.n_rounds - 1):
                # or self.task.s_2_node_color[self.task.s_1] != 0:
                self.valence_t[np.where(self.a_s1 == 0)] = -1

    def evaluate_delta(self):
        """Implement the agent's decision function delta"""

        # Random choice agents C1, C2, C3
        # ---------------------------------------------------------------------
        if self.agent_attr.name in ['C1', 'C2', 'C3']:
            self.decision_t = np.random.choice(self.a_s1, 1, p=self.valence_t)

        # Belief state based
        # ---------------------------------------------------------------------
        if self.agent_attr.name in ['A1', 'A2', 'A3', 'A4']:
            self.decision_t = self.a_s1[np.argmax(self.valence_t)]

    def make_decision(self):
        """Let agent identify possible actions given s1, evaluate action
        valences and make decision.
        """

        # -------Identify state-dependent action set----------
        self.identify_a_giv_s1()

        # -------Initialize valence array for state dependent actions----------
        self.valence_t = np.full(len(self.a_s1), np.nan)

        # -------Identify closest nodes with maximum s3 belief values----------
        if self.agent_attr.is_bayesian:
            self.eval_closest_max_s3_b_nodes()

        if self.agent_attr.is_explorative:
            start = time.time()
            self.eval_p_o_giv_o()
            end = time.time()
            print(
                f"time needed for p_o_giv_o: {humanreadable_time(end-start)}")
            
            start = time.time()
            self.eval_kl()
            end = time.time()
            print(
                f"time needed for kl: {humanreadable_time(end-start)}")

        # -------Evaluate valence and decision function----------
        # start = time.time()
        self.evaluate_action_valences()
        # end = time.time()
        # print(f'Evaluate phi: {end - start}')
        self.evaluate_delta()
