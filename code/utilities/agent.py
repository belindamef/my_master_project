import numpy as np
import copy as cp
import time


class Agent:
    """A class used to represent an agent model.
    An agent object can interact with a task object within an
    agent-based behavioral modelling framework

    ...

    Parameters
    ----------
    agent : str
        Agent model
    bayesian : bool
        True if agent model is bayesian, False otherwise
    exploring : bool
        True if agent model is uses exploring strategy, False otherwise
    model_comps : any
        Object of class ModelComponents, containing prior, s4_permutations, s4_permutation_indices and likelihood
    task_object : any
        Object of class Task

    Attributes
    ----------
    agent : str
        Agent model
    bayesian : bool
        True if agent model is bayesian, False otherwise
    exploring : bool
        True if agent model is uses exploring strategy, False otherwise
    task : any
        Object of class Task
    c : int
        Current round index
    t: int
        Current trial index
    moves : int
        Number of moves left in round c
    a_s1 : array_like
        State-dependent action set
    o_s2 : array_like
        State-dependent observation-set
    v : array_like
        (a_s1 x 1)-dimensional array denoting decision valences for each availabe action
    d : float
        Agent's decision
    s4_perms : list of tuple
        All possible permutations for state s_4
    s4_perm_node_indices : dict of list
        Dictionary with one list for each node. Each list contains indices corresponding to s4_permutations in which
        respective nodes are a hiding spot (TODO: add example to make it clearer?)
    prior_c0 : array_like
        (n_nodes x n_s4_perms)-dimensional array denoting the agent's initial belief state (subjective uncertainty
        over the non-observable states s3_c and s4 at trial t=0 in round c=0
    prior_c : array_like
        (n_nodes x n_s4_perms)-dimensional array denoting the agent's initial belief state (subjective uncertainty)
        over the non-observable states s3_c and s4 at trial t=0 in round c=0
    marg_s3_prior : array_like
        (n_nodes x 1)-dimensional array denoting the agent's marginal initial belief state (subjective uncertainty)
        over the non-observable states s3_c at trial t=0 in round c=0
    marg_s4_prior : array_like
        (n_nodes x 1)-dimensional array denoting the agent's marginal initial belief state (subjective uncertainty)
        over the non-observable states s4 at trial t=0 in round c=0
    lklh : array_like
        (2 x n_nodes x 3 x 4 x n_nodes x n_s4_pers)-dimensional array denoting the state and action-dependent
        state-conditional observation distribution (prob. distribution of o_t on node s1 with node color s3_{s1} given
        the non-observable states s3_c and s4 and action a_t
    p_s_giv_o : array_like
        (n_nodes x n_s4_perms)-dimensional array denoting the agent's (posterior) belief state (subjective uncertainty)
        over the non-observable states s3_c and s4 at trial t given the history of observations and actions
    marg_s3_b : array_like
        (n_nodes x 1)-dimensional array denoting the agent's marginal (posterior) belief state (subjective uncertainty)
        over the non-observable states s3_c at trial t given the history of observations and actions
    marg_s4_b : array_like
        (n_nodes x 1)-dimensional array denoting the agent's marginal (posterior) belief state (subjective uncertainty)
        over the non-observable states s4 at trial t given the history of observations and actions
    zero_sum_denominator : int
        Variable denoting whether or not a zero value was encountered in denominator during belief state updat

    TODO
    self.max_s3_b_value = np.nan
    self.rounded_marg_s3_b = np.full(self.task.n_nodes, np.nan)
    self.max_s3_b_nodes = np.nan
    self.dist_to_max_s3_b_nodes = np.nan
    self.shortest_dist_to_max_s3_b = np.nan
    self.closest_max_s3_b_nodes_i_s = np.nan
    self.closest_max_s3_b_nodes = np.nan

    # Initialize p_o_giv_o and kl objects
    self.p_o_giv_o = np.nan
    self.kl_giv_a_o = np.nan
    self.virt_b = np.nan

    """

    def __init__(self, agent, bayesian, exploring, model_comps, task_object):
        self.agent = agent
        self.bayesian = bayesian
        self.exploring = exploring
        self.task = task_object
        self.model = None

        # Initialize dynamic agent attributes
        self.c = np.nan  # hunting round counter
        self.t = np.nan  # this_trial counter
        self.moves = cp.deepcopy(self.task.n_trials)
        self.a_s1 = np.nan  # state-dependent action-set
        self.o_s2 = np.nan  # state-dependent observation-set

        # Initialize arrays for agent's decision valences and decisions
        self.v = np.full(5, np.nan)  # decision valences
        self.d = np.full(1, np.nan)  # decision

        # Initialize belief state objects
        self.marg_s3_b = np.full(self.task.n_nodes, np.nan)
        self.marg_s4_b = np.full(self.task.n_nodes, np.nan)
        self.marg_s3_prior = np.full(self.task.n_nodes, np.nan)
        self.marg_s4_prior = np.full(self.task.n_nodes, np.nan)
        self.zero_sum_denominator = 0

        if self.bayesian:
            # Fetch model components
            self.s4_perms = model_comps.s4_perms
            self.s4_perm_node_indices = model_comps.s4_perm_node_indices
            self.n_s4_perms = model_comps.n_s4_perms
            self.prior_c0 = model_comps.prior_c0  # ---(Prior, c == 0)---
            self.lklh = model_comps.lklh

            # Initialize posterior object
            self.prior_c = np.full((self.task.n_nodes, self.n_s4_perms), 0.)  # ---(Prior, c != 0)---
            self.p_s_giv_o = np.full((self.task.n_nodes, self.n_s4_perms), 0.)  # ---(Posterior)---

        # Initialize closest max s3 node variables for computations
        self.max_s3_b_value = np.nan
        self.rounded_marg_s3_b = np.full(self.task.n_nodes, np.nan)
        self.max_s3_b_nodes = np.nan
        self.dist_to_max_s3_b_nodes = np.nan
        self.shortest_dist_to_max_s3_b = np.nan
        self.closest_max_s3_b_nodes_i_s = np.nan
        self.closest_max_s3_b_nodes = np.nan

        # Initialize p_o_giv_o and kl objects
        self.p_o_giv_o = np.nan
        self.kl_giv_a_o = np.nan
        self.virt_b = np.nan

    def eval_prior_subs_rounds(self):
        """Reset belief states for s3 (treasure) based on marginal s4 (hiding spot) beliefs"""

        # Initialize all as zero
        self.prior_c[:, :] = 0.

        marg_s4_perm_b = np.full(self.n_s4_perms, np.nan)
        for s4_perm in range(self.n_s4_perms):
            marg_s4_perm_b[s4_perm] = self.p_s_giv_o[:, s4_perm].sum()
        sum_prob_s4_perm = marg_s4_perm_b[:].sum()

        for s3 in range(self.task.n_nodes):
            self.prior_c[s3, self.s4_perm_node_indices[s3]] = \
                marg_s4_perm_b[self.s4_perm_node_indices[s3]] * (1 / self.task.n_hides)

        # Evaluate marginal treasure distribution
        marg_s3_b = np.full(self.task.n_nodes, np.nan)
        for s3 in range(self.task.n_nodes):
            marg_s3_b[s3] = self.prior_c[s3, :].sum()
        sum_prob_tr = marg_s3_b[:].sum()

        # Evaluate marginal hiding spot distribution
        marg_s4_b = np.full(self.task.n_nodes, np.nan)
        for node in range(self.task.n_nodes):
            marg_s4_b[node] = self.prior_c[:, self.s4_perm_node_indices[node]].sum()
        sum_prob_hides = marg_s4_b[:].sum()
        debug = 'here'

    def eval_posterior(self, prior_belief_state, a, s1, s2_s1, o, record_zerosum=False):
        """Evaluate posterior belief state given prior, action, s1 and observation"""
        # Convert action value to 1, if step action
        if a != 0:
            a = 1

        if np.sum(prior_belief_state * self.lklh[a, s1, s2_s1, o, :, :]) == 0:
            if record_zerosum:
                self.zero_sum_denominator = 1
            post_belief_state = prior_belief_state * self.lklh[a, s1, s2_s1, o, :, :]
            print('zero_sum occurred')
            debug = 'here'
        else:
            post_belief_state = prior_belief_state * self.lklh[a, s1, s2_s1, o, :, :] \
                                * (1 / np.sum(prior_belief_state * self.lklh[a, s1, s2_s1, o, :, :]))

        return post_belief_state

    def eval_marg_b(self, belief):
        """Compute marginal posterior distributions for each node to e treasure location or hiding spot"""
        # Evaluate marginal treasure distribution
        marg_s3_b = np.full(self.task.n_nodes, np.nan)
        for s3 in range(self.task.n_nodes):
            marg_s3_b[s3] = belief[s3, :].sum()

        # Evaluate marginal hiding spot distribution
        marg_s4_b = np.full(self.task.n_nodes, np.nan)
        for node in range(self.task.n_nodes):
            marg_s4_b[node] = belief[:, self.s4_perm_node_indices[node]].sum()
        sum_prob_hides = marg_s4_b[:].sum()

        return marg_s3_b, marg_s4_b

    def start_new_trial(self):
        """Reset dynamic states to initial values for a new trial"""
        self.moves -= 1
        self.v = np.full(5, np.nan)  # decision valences
        self.d = np.full(1, np.nan)  # decision
        self.zero_sum_denominator = 0

    def start_new_round(self):
        """Reset dynamic states to initial values for a new round"""
        self.moves = cp.deepcopy(self.task.n_trials)
        # Reset belief states if not first round, if bayesian agent
        if self.bayesian and self.c > 0:
            self.eval_prior_subs_rounds()

        # Marginal prior distributions
        if self.bayesian:
            if self.c == 0:
                self.marg_s3_prior, self.marg_s4_prior = self.eval_marg_b(self.prior_c0)
            else:
                self.marg_s3_prior, self.marg_s4_prior = self.eval_marg_b(self.prior_c)

    def update_belief_state(self):
        """Update belief state self"""
        if self.bayesian:
            if self.c == 0 and self.t == 0:
                self.p_s_giv_o = self.eval_posterior(self.prior_c0, 1, self.task.s1_t,
                                                     self.task.s2_t[self.task.s1_t], self.task.o_t,
                                                     record_zerosum=True)
            elif self.t == 0:
                self.p_s_giv_o = self.eval_posterior(self.prior_c, 1, self.task.s1_t,
                                                     self.task.s2_t[self.task.s1_t], self.task.o_t,
                                                     record_zerosum=True)

            else:
                self.p_s_giv_o = self.eval_posterior(self.p_s_giv_o, self.model.a_t, self.task.s1_t,
                                                     self.task.s2_t[self.task.s1_t], self.task.o_t,
                                                     record_zerosum=True)

            self.marg_s3_b, self.marg_s4_b = self.eval_marg_b(self.p_s_giv_o)

    def identify_a_giv_s1(self):
        """Identify state s1 dependent action set"""
        self.a_s1 = cp.deepcopy(self.task.a_set)
        for action in np.nditer(self.task.a_set):

            # Remove forbidden steps (walk outside border)
            if ((self.task.s1_t + action) < 0) or (self.task.s1_t + action) >= self.task.n_nodes or \
                    ((self.task.s1_t % self.task.dim == 0) and action == -1) or \
                    (((self.task.s1_t + 1) % self.task.dim == 0) and action == 1):
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
        self.max_s3_b_nodes = np.where(self.rounded_marg_s3_b == self.max_s3_b_value)[0]

        # Evaluate shortest distances to max_s3_nodes
        self.dist_to_max_s3_b_nodes = np.full(len(self.max_s3_b_nodes), np.nan)
        for index, node in np.ndenumerate(self.max_s3_b_nodes):
            self.dist_to_max_s3_b_nodes[index] = self.task.shortest_dist_dic[f'{int(self.task.s1_t)}_to_{node}']
        self.shortest_dist_to_max_s3_b = np.amin(self.dist_to_max_s3_b_nodes)
        self.closest_max_s3_b_nodes_i_s = np.where(self.dist_to_max_s3_b_nodes == self.shortest_dist_to_max_s3_b)[0]
        self.closest_max_s3_b_nodes = self.max_s3_b_nodes[self.closest_max_s3_b_nodes_i_s]

    def eval_p_o_giv_o(self):
        """Evaluate agent's belief state-dependent posterior predictive distribution"""

        # Evaluate p_0 with likelihood
        self.p_o_giv_o = np.full((len(self.a_s1), 4), 0.)

        for i, a in np.ndenumerate(self.a_s1):

            new_s1 = self.task.s1_t + a

            # Convert action value to 1, if step action
            if a != 0:
                a = 1

            # Identify possible observations on new_s1
            self.identify_o_giv_s2_marg_s3(new_s1, a)

            for o in self.o_s2:
                product_a_o = self.p_s_giv_o * self.lklh[a, new_s1, self.task.s2_t[new_s1], o, :, :]
                sum_prod = np.sum(product_a_o)
                self.p_o_giv_o[i, o] = sum_prod

    @staticmethod
    def kl(p_x, q_x):
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
        # quotient_matrix[np.where(quotient_matrix == 0)] = np.nan  # Mask zeros with nan
        # log_quot = np.nan_to_num(np.log(quotient_matrix))
        # kl = np.nansum(p_x * log_quot)
        # end = time.time()
        # print(f'After masking zeros: {end - start}')

        if np.all(np.around(p_x, 10) == np.around(q_x, 10)):
            kl_mask = 0.

        else:
            # start = time.time()
            diffs = np.where(p_x != q_x)
            kl_mask = np.sum(p_x * np.ma.masked_invalid(np.log(p_x / q_x)))
            # end = time.time()
            # print(f'Using ma.masked all_in_one: {end - start}')

        return kl_mask

    def eval_kl(self):
        """Evaluate KL divergence between the agent's belief state on trial t and its virtual belief state on
        trial t + 1 given observation o_t and action

        NOTE: to save computation time kl is only evaluated if p_o_giv_o != 0, else, kl == 0"""

        self.kl_giv_a_o = np.full((len(self.a_s1), 4), 0.)
        self.virt_b = {0: {0: 0., 1: 0., 2: 0., 3: 0.}}

        for a, action in np.ndenumerate(self.a_s1):

            new_s1 = self.task.s1_t + action

            # Convert action value to 1, if step action
            if action != 0:
                action = 1

            self.virt_b[a] = {0: 0.}  # , 1: 0., 2: 0., 3: 0.}  # TODO: solved?

            # Identify possible observations
            self.identify_o_giv_s2_marg_s3(new_s1, action)

            for o in self.o_s2:

                # Evaluate virtual belief state on t+1 given potential observations and actions
                self.virt_b[a][o] = self.eval_posterior(self.p_s_giv_o, action, new_s1,
                                                        self.task.s2_t[new_s1], o)

                # Evaluate KL divergence
                self.kl_giv_a_o[a, o] = self.kl(self.virt_b[a][o], self.p_s_giv_o)

    def evaluate_phi(self):
        """Evaluate action valences"""

        # 'C1' Valence for random choice agent
        # --------------------------------------------------------------------------
        if self.agent == 'C1':
            # Allocate equal valences over all available actions
            self.v[:] = 1 / len(self.a_s1)

        # 'C2' Valence for random exploiter agent
        # --------------------------------------------------------------------------
        if self.agent == 'C2':
            # Allocate equal valences over all available step action (minus drill action)
            self.v[:] = 1 / (len(self.a_s1) - 1)

            # Set valence for drill action to zero
            self.v[np.where(self.a_s1 == 0)] = 0

        # 'C3' Valence for 50% random exploit and 50% random explore model
        # --------------------------------------------------------------------------
        if self.agent == 'C3':
            # Allocation 50% times equal valences over all available step actions (minus drill)
            self.v[:] = 1 / 2 * (1 / (len(self.a_s1) - 1))

            # Set drill valence to 50%
            self.v[np.where(self.a_s1 == 0)] = 1 / 2

        # 'A1' belief state based, exploiter, maximize immediate reward ---LOOK-AHEAD---
        # --------------------------------------------------------------------------
        if self.agent == 'A1':

            self.v[:] = 0

            # Iterate over possible actions (i.e. state-dependent actions)
            for index, action in np.ndenumerate(self.a_s1):
                # Anticipate new possible new position
                new_s1 = self.task.s1_t + action

                for close_max_s3_node in self.closest_max_s3_b_nodes:
                    current_dist_to_max_belief = self.task.shortest_dist_dic[
                        f'{int(self.task.s1_t)}_to_{close_max_s3_node}']
                    new_dist_to_closest_max_beliefs = self.task.shortest_dist_dic[
                        f'{int(new_s1)}_to_{close_max_s3_node}']
                    if self.moves >= new_dist_to_closest_max_beliefs < current_dist_to_max_belief:
                        self.v[index] += self.marg_s3_b[close_max_s3_node]

            # Set drill action to minus value ( --> to always have lower value than zero)
            #self.v[np.where(self.a_s1 == 0)] = -1  # Needed, otherwise A1 will drill in last trials
            # Let agent stop drilling, if node is not black or if it's last round
            if self.c == (self.task.n_rounds - 1):  # or self.task.s_2_node_color[self.task.s_1] != 0:
                self.v[np.where(self.a_s1 == 0)] = -1

        # 'A2' pure explorer agent
        # --------------------------------------------------------------------------
        if self.agent == 'A2':

            # Iterate over possible actions (i.e. state-dependent actions)
            for a, action in np.ndenumerate(self.a_s1):
                self.v[a] = self.p_o_giv_o[a, 0] * self.kl_giv_a_o[a, 0] + \
                            self.p_o_giv_o[a, 1] * self.kl_giv_a_o[a, 1] + \
                            self.p_o_giv_o[a, 2] * self.kl_giv_a_o[a, 2] + \
                            self.p_o_giv_o[a, 3] * self.kl_giv_a_o[a, 3]

            # Let agent stop drilling, if node is not black or if it's last round
            if self.c == (self.task.n_rounds - 1):  # or self.task.s_2_node_color[self.task.s_1] != 0:
                self.v[np.where(self.a_s1 == 0)] = -1

        # 'A3' belief state based explorer-exploit agent ---LOOK-AHEAD---
        # --------------------------------------------------------------------------
        if self.agent == 'A3':

            self.v[:] = 0

            # Iterate over possible actions (i.e. state-dependent actions)
            for a, action in np.ndenumerate(self.a_s1):
                # Anticipate new possible new position
                new_s1 = self.task.s1_t + action

                for close_max_s3_node in self.closest_max_s3_b_nodes:
                    current_dist_to_max_belief = self.task.shortest_dist_dic[
                        f'{int(self.task.s1_t)}_to_{close_max_s3_node}']
                    new_dist_to_closest_max_beliefs = self.task.shortest_dist_dic[
                        f'{int(new_s1)}_to_{close_max_s3_node}']
                    if self.moves >= new_dist_to_closest_max_beliefs < current_dist_to_max_belief:
                        self.v[a] += self.marg_s3_b[close_max_s3_node]

            # Add information value
            for a, action in np.ndenumerate(self.a_s1):
                # Move to next loop, if a == 0, i.e. i == 0, because p_o[0] is already filled
                # if action == 0:
                #     continue
                self.v[a] += self.p_o_giv_o[a][0] * self.kl_giv_a_o[a][0] + \
                             self.p_o_giv_o[a][1] * self.kl_giv_a_o[a][1] + \
                             self.p_o_giv_o[a][2] * self.kl_giv_a_o[a][2] + \
                             self.p_o_giv_o[a][3] * self.kl_giv_a_o[a][3]

            # Let agent stop drilling, if node is not black or if it's last round
            if self.c == (self.task.n_rounds - 1):  # or self.task.s_2_node_color[self.task.s_1] != 0:
                self.v[np.where(self.a_s1 == 0)] = -1

    def evaluate_delta(self):
        """Implement the agent's decision function delta"""

        # Random choice agents C1, C2, C3
        # --------------------------------------------------------------------------
        if self.agent in ['C1', 'C2', 'C3']:
            self.d = np.random.choice(self.a_s1, 1, p=self.v)

        # Belief state based
        # --------------------------------------------------------------------------
        if self.agent in ['A1', 'A2', 'A3', 'A4']:
            self.d = self.a_s1[np.argmax(self.v)]  # TODO: will always choose first value...??

    def make_decision(self):
        """Let agent make decision"""

        # -------Identify state-dependent action set----------
        self.identify_a_giv_s1()

        # -------Initialize valence array for state dependent actions----------
        self.v = np.full(len(self.a_s1), np.nan)  # Better initialize with zeros?

        # -------Identify closest nodes with maximum s3 belief values------------
        if self.bayesian:
            self.eval_closest_max_s3_b_nodes()

        if self.exploring:
            self.eval_p_o_giv_o()
            self.eval_kl()

        # -------Evaluate valence and decision function----------
        # start = time.time()
        self.evaluate_phi()
        # end = time.time()
        # print(f'Evaluate phi: {end - start}')
        self.evaluate_delta()
