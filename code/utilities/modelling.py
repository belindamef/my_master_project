from dataclasses import dataclass
import copy as cp
import os
import time
import pickle
import numpy as np
import more_itertools


@dataclass
class AgentInitObject:
    agent_model: str
    is_bayesian: bool
    is_explorative: bool
    is_deterministic: bool


def define_agent_attributes(agent_model):
    """Returns an agent initialization object

    Parameters
    ----------
    agent_model : str

    Returns
    -------
    agent_attributes : AgentInitObject
        obj with agent attributes
    """
    # Control models
    if agent_model in ['C1', 'C2', 'C3']:
        agent_attributes = AgentInitObject(agent_model=agent_model,
                                           is_bayesian=False,
                                           is_explorative=False,
                                           is_deterministic=False)

    # Bayesian models
    elif agent_model == 'A1':
        agent_attributes = AgentInitObject(agent_model=agent_model,
                                           is_bayesian=True,
                                           is_explorative=False,
                                           is_deterministic=True)

    # Bayesian models using explorative strategy
    elif agent_model in ['A2', 'A3']:
        agent_attributes = AgentInitObject(agent_model=agent_model,
                                           is_bayesian=True,
                                           is_explorative=True,
                                           is_deterministic=True)
    else:
        agent_attributes = AgentInitObject(agent_model=agent_model,
                                           is_bayesian=False,
                                           is_explorative=False,
                                           is_deterministic=False)

    # agent_attributes.bayesian_model_comps = bayesian_model_comps
    return agent_attributes


class BayesianModelComps:
    """A Class to create task configurations given a set of task parameters.
    Sampled task configuration npy files are written to output_dir

    TODO
    """
    s4_perms = []
    s4_perm_node_indices = {}
    n_s4_perms = np.nan
    prior_c0 = np.nan
    lklh = np.nan

    def __init__(self, path, task_design_params):
        self.paths = path
        self.task_design_params = task_design_params

    def eval_s4_perms(self):
        """Evaluate permutations of s4 states"""
        s_4_values = [0] * (self.task_design_params.n_nodes -
                            self.task_design_params.n_hides)
        s_4_values.extend([1] * self.task_design_params.n_hides)
        self.s4_perms = sorted(
            more_itertools.distinct_permutations(s_4_values))

    def eval_prior(self):
        """Evaluate agent's state priors"""
        for s3 in range(self.task_design_params.n_nodes):
            for index, s4_perm in enumerate(self.s4_perms):

                if s4_perm[s3] == 1:
                    self.prior_c0[s3, index] = 1 / (
                            self.n_s4_perms * self.task_design_params.n_hides)
                    # self.prior_c0[s3, index] = 1 / 1062600

    def eval_likelihood(self):
        """Evaluate action-dependent state-conditional observation
        distribution p(o|s) (likelihood), separately for
        action = 0 and action not 0"""

        # Loop through s4_permutations:
        for index, s4_perm in enumerate(self.s4_perms):

            # Loop through s1 values
            for s1 in range(self.task_design_params.n_nodes):

                # ---------for all a = 0---------------

                # If s4[s1] == 0 (not hiding spot), lklh(o == 1 (grey)) = 1,
                #   else remain zero
                if s4_perm[s1] == 0:

                    # for s2[s1] == 0 (black)
                    # -----------------------
                    s2_s1 = 0
                    self.lklh[0, s1, s2_s1, 1, :, index] = 1
                    self.lklh[0, s1, s2_s1, 1, s1, index] = 0

                    # for s2[s1] == 1 (grey)
                    # -----------------------
                    s2_s1 = 1
                    self.lklh[0, s1, s2_s1, 1, :, index] = 1
                    self.lklh[0, s1, s2_s1, 1, s1, index] = 0

                    # for s2[s1] == 2 (blue)
                    # -----------------------
                    # s2_s1 = 2
                    # bg color blue is impossible for s4_s1=0

                # If s4[s1] == 1 (hiding spot), lklh( o == 2 (blue)) = 1,
                    # else remain zero
                if s4_perm[s1] == 1:

                    # for s2[s1] == 0 (black)
                    # -----------------------
                    s2_s1 = 0
                    # will deterministically turn to blue since s4_s1=1
                    self.lklh[0, s1, s2_s1, 2, :, index] = 1
                    self.lklh[0, s1, s2_s1, 2, s1, index] = 0

                    # for s2[s1] == 1 (grey)
                    # -----------------------
                    # s2_s1 = 1
                    # grey node bg color impossible for s4_s1=1

                    # for s2[s1] == 2 (blue)
                    # -----------------------
                    # will return same color as already unveiled
                    s2_s1 = 2
                    self.lklh[0, s1, s2_s1, 2, :, index] = 1
                    self.lklh[0, s1, s2_s1, 2, s1, index] = 0

                # ---------for all a = 1---------------

                # If s4[s1] == 0 (not hiding spot)
                if s4_perm[s1] == 0:
                    # for s2[s1] == 0 (black)
                    # -----------------------
                    s2_s1 = 0

                    # For s3 == s1, lklh(o == 0 (black)) = 0,
                    # else lklh(o == 0 (black)) = 1
                    self.lklh[1, s1, s2_s1, 0, :, index] = 1
                    self.lklh[1, s1, s2_s1, 0, s1, index] = 0

                    # all other observations ( o==1, o==2, o==3 remain 0)

                    # for s2[s1] == 1 (grey)
                    # -----------------------
                    s2_s1 = 1

                    # For s3 == s1, lklh(o == 1 (grey)) = 0,
                    # else lklh(o == 1 (grey)) = 1
                    self.lklh[1, s1, s2_s1, 1, :, index] = 1
                    self.lklh[1, s1, s2_s1, 1, s1, index] = 0

                    # all other observations ( o==0, o==2, o==3 remain 0)

                    # for s2[s1] == 2 (blue)
                    # -----------------------
                    # s2_s1 = 2
                    # node color blue is impossible

                # If s4[s1] == 1 (node is a hiding spot)
                if s4_perm[s1] == 1:
                    # for s2[s1] == 0 (black)
                    # -----------------------
                    s2_s1 = 0

                    # For s3 == s1, lklh(o == 0 (black)) = 0,
                    # else lklh(o == 0 (black)) = 1
                    self.lklh[1, s1, s2_s1, 0, :, index] = 1
                    self.lklh[1, s1, s2_s1, 0, s1, index] = 0

                    # For s3 == 1, lklh(o == 3 (treasure)) = 1,
                    # else remain zero
                    self.lklh[1, s1, s2_s1, 3, s1, index] = 1

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
                    self.lklh[1, s1, s2_s1, 2, :, index] = 1
                    self.lklh[1, s1, s2_s1, 2, s1, index] = 0

                    # For s3 == 1, lklh(o == 3 (treasure)) = 1,
                    # else remain zero
                    self.lklh[1, s1, s2_s1, 3, s1, index] = 1

    def create_or_load_components(self):
        """Create or load Bayesian model components (prior and likelihood)"""
        # Initialize and evaluate s_4 permutations
        s4_perms_fn_pkl = os.path.join(
            self.paths.code, "utilities",
            f"s4_perms_dim-{self.task_design_params.dim}_"
            f"h{self.task_design_params.n_hides}.pkl")

        if os.path.exists(s4_perms_fn_pkl):
            start = time.time()
            with open(s4_perms_fn_pkl, "rb") as file:
                self.s4_perms = pickle.load(file)
            end = time.time()
            print(f" ...loaded s4_perms, timed needed: "
                  f"{round((end - start), ndigits=2)}")
        else:
            start = time.time()
            self.s4_perms = []
            self.eval_s4_perms()
            end = time.time()
            print(f" ... computed s4_perms, time needed: "
                  f"{round((end - start), ndigits=2)}")
            start = time.time()
            with open(s4_perms_fn_pkl, "wb") as file:
                pickle.dump(self.s4_perms, file)
            end = time.time()
            print(f" ... saved s4_perms as pickle, time needed: "
                  f"{round((end-start), ndigits=2)}")

        # Create list with indices of all probs for each hide
        start = time.time()
        for node in range(self.task_design_params.n_nodes):
            self.s4_perm_node_indices[node] = [
                index for index, s4_perm in enumerate(self.s4_perms)
                if s4_perm[node] == 1
            ]
            # --> 25 X 42504 indices per hide ( if 25 nodes and 6 hides)
        end = time.time()
        print(f" ... computed s4_marg_indices, time needed: "
              f"{round((end - start), ndigits=2)}")

        # Evaluate number of s4 permutations
        self.n_s4_perms = len(self.s4_perms)

        # Load/evaluate agent's initial belief state in 1. trial ---(Prior)---
        prior_fn = os.path.join(self.paths.code, "utilities",
                                f"prior_dim-{self.task_design_params.dim}_"
                                f"h{self.task_design_params.n_hides}.npy")
        if os.path.exists(prior_fn):
            start = time.time()
            self.prior_c0 = np.load(prior_fn)
            end = time.time()
            print(f" ... loaded prior, time needed: "
                  f"{round((end - start), ndigits=2)}")
            # sum_p_c0 = np.sum(self.prior_c0)
        else:
            start = time.time()
            self.prior_c0 = np.full((self.task_design_params.n_nodes,
                                     self.n_s4_perms), 0.0)
            self.eval_prior()
            end = time.time()
            print(f" ... evaluated prior, time needed: "
                  f" {round((end - start), ndigits=2)}")
            start = time.time()
            np.save(prior_fn, self.prior_c0)
            end = time.time()
            print(f" ... saved prior, time needed:"
                  f" {round((end - start), ndigits=2)}")

        # Load/eval action-dep. state-cond. obs distribution ---(Likelihood)---
        lklh_fn = os.path.join(
            self.paths.code,
            "utilities",
            f"lklh_dim-{self.task_design_params.dim}_"
            f"h{self.task_design_params.n_hides}.npy"
        )
        if os.path.exists(lklh_fn):
            start = time.time()
            self.lklh = np.load(lklh_fn)
            end = time.time()
            print(f" ... loaded lkhl, time needed: "
                  f"{round((end - start), ndigits=2)}")
        else:
            self.lklh = np.zeros(
                (2, self.task_design_params.n_nodes, 3, 4,
                 self.task_design_params.n_nodes, self.n_s4_perms),
                dtype=np.uint16)
            start = time.time()
            self.eval_likelihood()
            end = time.time()
            print(f" ... computed lkhl, time needed: "
                  f"{round((end - start), ndigits=2)}")
            start = time.time()
            np.save(lklh_fn, self.lklh)
            end = time.time()
            print(f" ... saved lkhl, time needed: "
                  f"{round((end - start), ndigits=2)}")
        # start = time.time()


class BehavioralModel:
    """A class used to represent the behavioral model

    Attributes
    ----------
    agent : object
        Object of class agent
    a_t : array_like
        Action value in trial t
    """

    def __init__(self):
        # Initialize empty attribute to embed agent object of class Agent
        self.agent = None

        # Initialize action attribute
        self.a_t = np.full(1, np.nan)  # agent action

    def return_action(self):
        """This function returns the action value given agent's decision."""
        # probability action given decision of 1
        self.a_t = cp.deepcopy(self.agent.d)
