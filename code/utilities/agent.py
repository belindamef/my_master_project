"""Module classes to create an agent that can interact with the treasure
hunt task.
"""
import os
import time
import pickle
import numpy as np
import more_itertools
from utilities.task import Task
from utilities.config import Paths, TaskDesignParameters, humanreadable_time


class BayesianModelComps:
    """A Class to create task configurations given a set of task parameters.
    Sampled task configuration npy files are written to output_dir

    TODO
    """

    def __init__(self, task_design_params=TaskDesignParameters()):
        self.task_design_params = task_design_params
        self.paths: Paths = Paths()
        self.s4_perms = []
        self.s4_perm_node_indices = {}
        self.n_s4_perms: int = 0
        self.prior_c0: np.ndarray = np.array(np.nan)
        self.lklh: np.ndarray = np.array(np.nan)

    def eval_s4_perms(self):
        """Evaluate permutations of s4 states"""
        s_4_values = [0] * (self.task_design_params.n_nodes -
                            self.task_design_params.n_hides)
        s_4_values.extend([1] * self.task_design_params.n_hides)
        self.s4_perms = sorted(
            more_itertools.distinct_permutations(s_4_values))

    def eval_prior(self):
        """Evaluate agent's state priors"""
        for s_3 in range(self.task_design_params.n_nodes):
            for index, s4_perm in enumerate(self.s4_perms):

                if s4_perm[s_3] == 1:
                    self.prior_c0[s_3, index] = 1 / (
                            self.n_s4_perms * self.task_design_params.n_hides)
                    # self.prior_c0[s3, index] = 1 / 1062600

    def eval_likelihood(self):
        """Evaluate action-dependent state-conditional observation
        distribution p(o|s) (likelihood), separately for
        action = 0 and action not 0"""

        # Loop through s4_permutations:
        for index, s4_perm in enumerate(self.s4_perms):

            # Loop through s1 values
            for s_1 in range(self.task_design_params.n_nodes):

                # ---------for all a = 0---------------

                # If s4[s1] == 0 (not hiding spot), lklh(o == 1 (grey)) = 1,
                #   else remain zero
                if s4_perm[s_1] == 0:

                    # for s2[s1] == 0 (black)
                    # -----------------------
                    s2_s1 = 0
                    self.lklh[0, s_1, s2_s1, 1, :, index] = 1
                    self.lklh[0, s_1, s2_s1, 1, s_1, index] = 0

                    # for s2[s1] == 1 (grey)
                    # -----------------------
                    s2_s1 = 1
                    self.lklh[0, s_1, s2_s1, 1, :, index] = 1
                    self.lklh[0, s_1, s2_s1, 1, s_1, index] = 0

                    # for s2[s1] == 2 (blue)
                    # -----------------------
                    # s2_s1 = 2
                    # bg color blue is impossible for s4_s1=0

                # If s4[s1] == 1 (hiding spot), lklh( o == 2 (blue)) = 1,
                    # else remain zero
                if s4_perm[s_1] == 1:

                    # for s2[s1] == 0 (black)
                    # -----------------------
                    s2_s1 = 0
                    # will deterministically turn to blue since s4_s1=1
                    self.lklh[0, s_1, s2_s1, 2, :, index] = 1
                    self.lklh[0, s_1, s2_s1, 2, s_1, index] = 0

                    # for s2[s1] == 1 (grey)
                    # -----------------------
                    # s2_s1 = 1
                    # grey node bg color impossible for s4_s1=1

                    # for s2[s1] == 2 (blue)
                    # -----------------------
                    # will return same color as already unveiled
                    s2_s1 = 2
                    self.lklh[0, s_1, s2_s1, 2, :, index] = 1
                    self.lklh[0, s_1, s2_s1, 2, s_1, index] = 0

                # ---------for all a = 1---------------

                # If s4[s1] == 0 (not hiding spot)
                if s4_perm[s_1] == 0:
                    # for s2[s1] == 0 (black)
                    # -----------------------
                    s2_s1 = 0

                    # For s3 == s1, lklh(o == 0 (black)) = 0,
                    # else lklh(o == 0 (black)) = 1
                    self.lklh[1, s_1, s2_s1, 0, :, index] = 1
                    self.lklh[1, s_1, s2_s1, 0, s_1, index] = 0

                    # all other observations ( o==1, o==2, o==3 remain 0)

                    # for s2[s1] == 1 (grey)
                    # -----------------------
                    s2_s1 = 1

                    # For s3 == s1, lklh(o == 1 (grey)) = 0,
                    # else lklh(o == 1 (grey)) = 1
                    self.lklh[1, s_1, s2_s1, 1, :, index] = 1
                    self.lklh[1, s_1, s2_s1, 1, s_1, index] = 0

                    # all other observations ( o==0, o==2, o==3 remain 0)

                    # for s2[s1] == 2 (blue)
                    # -----------------------
                    # s2_s1 = 2
                    # node color blue is impossible

                # If s4[s1] == 1 (node is a hiding spot)
                if s4_perm[s_1] == 1:
                    # for s2[s1] == 0 (black)
                    # -----------------------
                    s2_s1 = 0

                    # For s3 == s1, lklh(o == 0 (black)) = 0,
                    # else lklh(o == 0 (black)) = 1
                    self.lklh[1, s_1, s2_s1, 0, :, index] = 1
                    self.lklh[1, s_1, s2_s1, 0, s_1, index] = 0

                    # For s3 == 1, lklh(o == 3 (treasure)) = 1,
                    # else remain zero
                    self.lklh[1, s_1, s2_s1, 3, s_1, index] = 1

                    # all other observations ( o==1, o==2 remain 0)

                    # for s2[s1] == 1 (grey)
                    # -----------------------
                    # s2_s1 = 1

                    # observation grey impossible --> all zero

                    # for s2[s1] == 2 (blue)
                    # -----------------------
                    s2_s1 = 2

                    # For s3 == s1, lklh(o == 2 (blue)) = 0,
                    # else lklh(o==2 (blue) = 1
                    self.lklh[1, s_1, s2_s1, 2, :, index] = 1
                    self.lklh[1, s_1, s2_s1, 2, s_1, index] = 0

                    # For s3 == 1, lklh(o == 3 (treasure)) = 1,
                    # else remain zero
                    self.lklh[1, s_1, s2_s1, 3, s_1, index] = 1

    def get_comps(self):
        """Create or load Bayesian beh_model components (prior and likelihood)
        """
        # Initialize and evaluate s_4 permutations
        s4_perms_fn_pkl = os.path.join(
            self.paths.code, "utilities",
            f"s4_perms_dim-{self.task_design_params.dim}_"
            f"h{self.task_design_params.n_hides}.pkl")

        if os.path.exists(s4_perms_fn_pkl):
            print("Loading s4_perms ...")
            start = time.time()
            with open(s4_perms_fn_pkl, "rb") as file:
                self.s4_perms = pickle.load(file)
            end = time.time()
            print(
                " ... finished loading s4_perms, timed needed: "
                f"{humanreadable_time(end-start)}"
                )
        else:
            print("Computing s4_perms ...")
            start = time.time()
            self.s4_perms = []
            self.eval_s4_perms()
            end = time.time()
            print(f" ... finished somputing s4_perms, time needed: "
                  f"{humanreadable_time(end-start)}"
                  )
            print("Saving s4_perms ...")
            start = time.time()
            with open(s4_perms_fn_pkl, "wb") as file:
                pickle.dump(self.s4_perms, file)
            end = time.time()
            print(f" ... finisehd saving s4_perms to files, time needed: "
                  f"{humanreadable_time(end-start)}"
                  )
        # Create list with indices of all probs for each hide
        print("Computing s4_marg_indices ...")
        start = time.time()
        for node in range(self.task_design_params.n_nodes):
            self.s4_perm_node_indices[node] = [
                index for index, s4_perm in enumerate(self.s4_perms)
                if s4_perm[node] == 1
            ]
            # --> 25 X 42504 indices per hide ( if 25 nodes and 6 hides)
        end = time.time()
        print(f" ... finished computing s4_marg_indices, time needed: "
              f"{humanreadable_time(end-start)}"
              )

        # Evaluate number of s4 permutations
        self.n_s4_perms = len(self.s4_perms)

        # Load/evaluate agent's initial belief state in 1. trial ---(Prior)---
        prior_fn = os.path.join(self.paths.code, "utilities",
                                f"prior_dim-{self.task_design_params.dim}_"
                                f"h{self.task_design_params.n_hides}.npy")
        if os.path.exists(prior_fn):
            print("Loading prior array from file ...")
            start = time.time()
            self.prior_c0 = np.load(prior_fn)
            end = time.time()
            print(f" ... finished loading prior, time needed: "
                  f"{humanreadable_time(end-start)}"
                  )
            # sum_p_c0 = np.sum(self.prior_c0)
        else:
            print("Evaluating prior belief array for given task config ...")
            start = time.time()
            self.prior_c0 = np.full((self.task_design_params.n_nodes,
                                     self.n_s4_perms), 0.0)
            self.eval_prior()
            end = time.time()
            print(f" ... finished evaluating prior, time needed: "
                  f"{humanreadable_time(end-start)}"
                  )
            print("Saving prior belief array to file ...")
            start = time.time()
            np.save(prior_fn, self.prior_c0)
            end = time.time()
            print(f" ... finished saving prior to file, time needed:"
                  f"{humanreadable_time(end-start)}"
                  )

        # Load/eval action-dep. state-cond. obs distribution ---(Likelihood)---
        lklh_fn = os.path.join(
            self.paths.code,
            "utilities",
            f"lklh_dim-{self.task_design_params.dim}_"
            f"h{self.task_design_params.n_hides}.npy"
        )
        if os.path.exists(lklh_fn):
            print("Loading likelihood array from file ...")
            start = time.time()
            self.lklh = np.load(lklh_fn)
            end = time.time()
            print(f" ... finished loading likelihood array, time needed: "
                  f"{humanreadable_time(end-start)}")
        else:
            print("Computing likelihood array for given task config ...")
            self.lklh = np.zeros(
                (2, self.task_design_params.n_nodes, 3, 4,
                 self.task_design_params.n_nodes, self.n_s4_perms),
                dtype=np.uint16)
            start = time.time()
            self.eval_likelihood()
            end = time.time()
            print(f" ... finished computing likelihood, time needed: "
                  f"{humanreadable_time(end-start)}")
            print("Saving likelihood array to file")
            start = time.time()
            np.save(lklh_fn, self.lklh)
            end = time.time()
            print(f" ... saved likelihood array to file, time needed: "
                  f"{humanreadable_time(end-start)}")
        # start = time.time()
        return self


class AgentAttributes:
    """A class to create object that stores creates and stores all necessary
    information to create an agent class instance
    """
    is_bayesian: bool
    is_explorative: bool
    is_deterministic: bool

    def __init__(self, agent_model: str):
        self.name = agent_model
        self.def_attributes()

    def def_attributes(self):
        """Define agent attributes dependent on beh_model

        Returns
        -------
        AgentInitObject
        """
        # Control models
        if self.name in ['C1', 'C2', 'C3']:
            self.is_bayesian = False
            self.is_explorative = False
            self.is_deterministic = False

        # Bayesian models
        elif self.name == 'A1':
            self.is_bayesian = True
            self.is_explorative = False
            self.is_deterministic = True

        # Bayesian models using explorative strategy
        elif self.name in ['A2', 'A3']:
            self.is_bayesian = True
            self.is_explorative = True
            self.is_deterministic = True
        return self


class Agent:
    """A class used to represent an agent behavioral model.
    An agent object can interact with a task object within an
    agent-based behavioral modelling framework

    """

    def __init__(self, agent_attr: AgentAttributes,
                 task_object: Task, lambda_):
        self.agent_attr = agent_attr

        self.task: Task = task_object
        self.lambda_ = lambda_

        # Initialize dynamic agent attributes
        self.moves = self.task.task_configs.params.n_trials
        self.a_s1: np.ndarray = np.array(np.nan)  # state-dependent action-set
        self.o_s2 = []  # state-dep observation-set

        # Initialize arrays for agent's decision valences and decisionsn
        self.valence_t = np.full(5, np.nan)  # decision valences
        self.decision_t = np.full(1, np.nan)  # decision

        # Initialize belief state objects
        self.marg_s3_b = np.full(self.task.task_params.n_nodes, np.nan)
        self.marg_s4_b = np.full(self.task.task_params.n_nodes, np.nan)
        self.marg_s3_prior = np.full(self.task.task_params.n_nodes, np.nan)
        self.marg_s4_prior = np.full(self.task.task_params.n_nodes, np.nan)
        self.zero_sum_denominator = 0

        if self.agent_attr.is_bayesian:
            # Unpack bayesian beh_model components
            self.bayes_comps: BayesianModelComps = BayesianModelComps()

            # ---(Prior, c != 0)---
            self.prior_c: np.ndarray = np.array(np.nan)
            # ---(Posterior)---
            self.p_s_giv_o: np.ndarray = np.array(np.nan)

        # Initialize closest max s3 node variables for computations
        self.max_s3_b_value = np.nan
        self.rounded_marg_s3_b = np.full(self.task.task_params.n_nodes, np.nan)
        self.max_s3_b_nodes = np.nan
        self.dist_to_max_s3_b_nodes = np.nan
        self.shortest_dist_to_max_s3_b = np.nan
        self.closest_max_s3_b_nodes_i_s: np.ndarray = np.array(np.nan)
        self.closest_max_s3_b_nodes: np.ndarray = np.array(np.nan)

        # Initialize p_o_giv_o and kl objects
        self.p_o_giv_o: np.ndarray = np.array(np.nan)
        self.kl_giv_a_o: np.ndarray = np.array(np.nan)
        self.virt_b = {}

    def add_bayesian_model_components(self, bayesian_comps):
        """Load or create prior, likelihood and permutation lists etc"""
        self.bayes_comps = bayesian_comps

    def eval_prior_subs_rounds(self):
        """Reset belief states for s3 (treasure) based on marginal s4
        (hiding spot) beliefs"""

        # Initialize all as zero
        self.prior_c = np.full(
            (self.task.task_params.n_nodes, self.bayes_comps.n_s4_perms), 0.)

        # marg_s4_perm_b = np.full(self.n_s4_perms, np.nan)
        # for s4_perm in range(self.n_s4_perms):
        #     marg_s4_perm_b[s4_perm] = self.p_s_giv_o[:, s4_perm].sum()

        marg_s4_perm_b = self.p_s_giv_o.sum(axis=0)
        # sum_prob_s4_perm = marg_s4_perm_b[:].sum()

        for s_3 in range(self.task.task_params.n_nodes):
            self.prior_c[s_3,
                         self.bayes_comps.s4_perm_node_indices[s_3]
                         ] = marg_s4_perm_b[
                             self.bayes_comps.s4_perm_node_indices[s_3]
                             ] * (1 / self.task.task_params.n_hides)

        # uncomment for DEBUGGING
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
        # debug = 'here'

    def eval_posterior(self, prior_belief_state, action, s_1, s2_giv_s1, obs
                       ) -> np.ndarray:
        """Evaluate posterior belief state given prior, action, s1 and
        observation"""
        # Convert action value to 1, if step action
        if action != 0:
            action = 1
        action = int(action)

        if np.sum(prior_belief_state * self.bayes_comps.lklh[
                action, s_1, s2_giv_s1, obs, :, :]) == 0:
            post_belief_state = (
                prior_belief_state
                * self.bayes_comps.lklh[action, s_1, s2_giv_s1, obs, :, :])
            print('sum of prio * lklh = 0, leaving out normalization')
            # debug = 'here'
        else:
            post_belief_state = (
                prior_belief_state
                * self.bayes_comps.lklh[action, s_1, s2_giv_s1, obs, :, :]
                * (1 / np.sum(prior_belief_state
                              * self.bayes_comps.lklh[
                                  action, s_1, s2_giv_s1, obs, :, :])))

        return post_belief_state

    def eval_marg_b(self, belief):
        """Compute marginal posterior distributions for each node to the
        treasure location or hiding spot"""
        # Evaluate marginal treasure distribution
        marg_s3_b = np.full(self.task.task_params.n_nodes, np.nan)
        for s_3 in range(self.task.task_params.n_nodes):
            marg_s3_b[s_3] = belief[s_3, :].sum()

        # Evaluate marginal hiding spot distribution
        marg_s4_b = np.full(self.task.task_params.n_nodes, np.nan)
        for node in range(self.task.task_params.n_nodes):
            marg_s4_b[node] = belief[
                :, self.bayes_comps.s4_perm_node_indices[node]].sum()
        # sum_prob_hides = marg_s4_b[:].sum()

        return marg_s3_b, marg_s4_b

    def start_new_trial(self):
        """Reset dynamic states to initial values for a new trial"""
        self.valence_t = np.full(5, np.nan)  # decision valences
        self.decision_t = np.full(1, np.nan)  # decision
        self.zero_sum_denominator = 0

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
        """Update belief state"""
        if self.agent_attr.is_bayesian:
            if self.task.current_round == 0 and self.task.current_trial == 0:
                self.p_s_giv_o = self.eval_posterior(
                    prior_belief_state=self.bayes_comps.prior_c0,
                    action=1,
                    s_1=self.task.s1_t,
                    s2_giv_s1=self.task.s2_t[int(self.task.s1_t)],
                    obs=self.task.o_t)

            elif self.task.current_trial == 0:
                self.p_s_giv_o = self.eval_posterior(
                    prior_belief_state=self.prior_c,
                    action=1,
                    s_1=self.task.s1_t,
                    s2_giv_s1=self.task.s2_t[int(self.task.s1_t)],
                    obs=self.task.o_t)

            else:
                self.p_s_giv_o = self.eval_posterior(
                    prior_belief_state=self.p_s_giv_o,
                    action=current_action,
                    s_1=self.task.s1_t,
                    s2_giv_s1=self.task.s2_t[int(self.task.s1_t)],
                    obs=self.task.o_t)

            self.marg_s3_b, self.marg_s4_b = self.eval_marg_b(self.p_s_giv_o)

    def identify_a_giv_s1(self):
        """Identify state s1 dependent action set"""
        self.a_s1 = self.task.a_set

        for action in self.task.a_set:
            new_s1 = action + self.task.s1_t
            # Remove forbidden steps (walk outside border)
            if (not (0 <= new_s1 < self.task.task_params.n_nodes)
                    or ((self.task.s1_t
                         % self.task.task_configs.params.dim == 0)
                        and action == -1)
                    or (((self.task.s1_t + 1
                          ) % self.task.task_configs.params.dim == 0)
                        and action == 1)):

                self.a_s1 = self.a_s1[self.a_s1 != action]

    def identify_o_giv_s2_marg_s3(self, node, action):
        """Identify state s2 dependent observation set"""
        if action == 0:
            if self.task.s2_t[node] == 0:
                if np.around(self.marg_s4_b[node], 10) == 0:
                    self.o_s2 = [1]
                elif np.around(self.marg_s4_b[node], 10) == 1:
                    self.o_s2 = [2]
                else:
                    self.o_s2 = [1, 2]
            elif self.task.s2_t[node] == 1:
                self.o_s2 = [1]
            elif self.task.s2_t[node] == 2:
                self.o_s2 = [2]
        elif action == 1:
            if self.task.s2_t[node] == 0:
                if np.around(self.marg_s3_b[node], 10) == 0:
                    self.o_s2 = [0]
                else:
                    self.o_s2 = [0, 3]
            elif self.task.s2_t[node] == 1:
                self.o_s2 = [1]
            elif self.task.s2_t[node] == 2:
                if np.around(self.marg_s3_b[node], 10) == 0:
                    self.o_s2 = [2]
                else:
                    self.o_s2 = [2, 3]

    def eval_closest_max_s3_b_nodes(self):
        """Identify nodes with maximum s3 belief state values"""
        # Identify maximum s3 belief state value
        self.max_s3_b_value = np.around(np.amax(self.marg_s3_b), 10)

        # Find all nodes with maximum belief state value
        self.rounded_marg_s3_b = np.around(self.marg_s3_b, 10)
        self.max_s3_b_nodes = np.where(
            self.rounded_marg_s3_b == self.max_s3_b_value)[0]

        # Evaluate shortest distances to max_s3_nodes
        self.dist_to_max_s3_b_nodes = np.full(len(self.max_s3_b_nodes), np.nan)
        for index, node in np.ndenumerate(self.max_s3_b_nodes):
            self.dist_to_max_s3_b_nodes[index] = \
                self.task.shortest_dist_dic[f'{int(self.task.s1_t)}_to_{node}']
        self.shortest_dist_to_max_s3_b = np.amin(self.dist_to_max_s3_b_nodes)
        self.closest_max_s3_b_nodes_i_s = np.where(
            self.dist_to_max_s3_b_nodes == self.shortest_dist_to_max_s3_b)[0]
        self.closest_max_s3_b_nodes = self.max_s3_b_nodes[
            self.closest_max_s3_b_nodes_i_s]

    def eval_p_o_giv_o(self):
        """Evaluate agent's belief state-dependent posterior predictive
        distribution"""

        # Evaluate p_0 with likelihood
        self.p_o_giv_o = np.full((len(self.a_s1), 4), 0.)

        for i, action in enumerate(self.a_s1):

            new_s1 = self.task.s1_t + action

            # Convert action value to 1, if step action
            if action != 0:
                action = 1

            # Identify possible observations on new_s1
            self.identify_o_giv_s2_marg_s3(new_s1, action)

            for obs in self.o_s2:
                product_a_o = (self.p_s_giv_o
                               * self.bayes_comps.lklh[action, new_s1,
                                                       self.task.s2_t[new_s1],
                                                       obs, :, :])
                sum_prod = np.sum(product_a_o)
                self.p_o_giv_o[i, obs] = sum_prod

    @staticmethod
    def evaluate_kl_divergence(p_x, q_x):
        """Evaluate kl divergence between two distributions"""
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

        NOTE: to save computation time kl is only evaluated if p_o_giv_o != 0,
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
                self.virt_b[i][obs] = self.eval_posterior(  # type: ignore
                    self.p_s_giv_o, action, new_s1,
                    self.task.s2_t[new_s1], obs)

                # Evaluate KL divergence
                kl_value = self.evaluate_kl_divergence(
                    self.virt_b[i][obs], self.p_s_giv_o)
                
                self.kl_giv_a_o[i, obs] = self.evaluate_kl_divergence(
                    self.virt_b[i][obs], self.p_s_giv_o)

                stop = "here"

    def evaluate_action_valences(self):
        """Evaluate action valences"""

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
                        f'{int(self.task.s1_t)}_to_{close_max_s3_node}']
                    new_dist_to_closest_max_beliefs = \
                        self.task.shortest_dist_dic[
                            f'{int(new_s1)}_to_{close_max_s3_node}']
                    if self.task.moves >= new_dist_to_closest_max_beliefs \
                            < current_dist_to_max_belief:
                        self.valence_t[index] += self.marg_s3_b[
                            close_max_s3_node]

            # Set drill action to minus value
            # ( --> to always have lower value than zero)
            # self.v[np.where(self.a_s1 == 0)] = -1  # Needed,
            # otherwise A1 will drill in last trials
            # Let agent stop drilling, if node is not black or if last round
            if self.task.current_round == (
                    self.task.task_configs.params.n_rounds - 1):
                # or self.task.s_2_node_color[self.task.s_1] != 0:
                self.valence_t[np.where(self.a_s1 == 0)] = -1

        # 'A2' pure explorer agent
        # --------------------------------------------------------------------------
        if self.agent_attr.name == 'A2':

            # Iterate over possible actions (i.e. state-dependent actions)
            for i, action_i in np.ndenumerate(self.a_s1):
                self.valence_t[i] = \
                    self.p_o_giv_o[i, 0] * self.kl_giv_a_o[i, 0] + \
                    self.p_o_giv_o[i, 1] * self.kl_giv_a_o[i, 1] + \
                    self.p_o_giv_o[i, 2] * self.kl_giv_a_o[i, 2] + \
                    self.p_o_giv_o[i, 3] * self.kl_giv_a_o[i, 3]

            # Let agent stop drilling, if node is not black or if last round
            if self.task.current_round == (
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
                        f'{int(self.task.s1_t)}_to_{close_max_s3_node}']
                    new_dist_to_closest_max_beliefs = \
                        self.task.shortest_dist_dic[
                            f'{int(new_s1)}_to_{close_max_s3_node}']
                    if self.task.moves >= new_dist_to_closest_max_beliefs \
                            < current_dist_to_max_belief:
                        self.valence_t[i] += (
                            1 - self.lambda_
                            ) * self.marg_s3_b[close_max_s3_node]

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

            # Let agent stop drilling, if node is not black or if last round
            if self.task.current_round == (
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
        """Let agent make decision"""

        # -------Identify state-dependent action set----------
        self.identify_a_giv_s1()

        # -------Initialize valence array for state dependent actions----------
        self.valence_t = np.full(len(self.a_s1), np.nan)

        # -------Identify closest nodes with maximum s3 belief values----------
        if self.agent_attr.is_bayesian:
            self.eval_closest_max_s3_b_nodes()

        if self.agent_attr.is_explorative:
            self.eval_p_o_giv_o()
            self.eval_kl()

        # -------Evaluate valence and decision function----------
        # start = time.time()
        self.evaluate_action_valences()
        # end = time.time()
        # print(f'Evaluate phi: {end - start}')
        self.evaluate_delta()
