import numpy as np
import more_itertools
import os
import time
import pickle


class ModelComps:
    """A Class to create task configurations given a set of task parameters.
    Sampled task configuration npy files are written to output_dir

    ...

    Attributes
    ----------
    working_dir : str
        Current working directory
    dim : int
        Dimensionality of the gridworld
    n_hides : int
        Number of hiding spots
    TODO

    Methods
    -------
    TODO
    """

    def __init__(self, working_dir, dim, n_hides):
        """This function is the instantiation operation of the model component class"""

        self.working_dir = working_dir
        self.dim = dim
        self.n_nodes = dim**2
        self.n_hides = n_hides

        # Initialize and evaluate s_4 permutations
        start = time.time()
        s4_perms_fn_pkl = os.path.join(
            self.working_dir,
            "utilities",
            f"s4_perms_dim-{self.dim}_h{self.n_hides}.pkl",
        )

        if os.path.exists(s4_perms_fn_pkl):
            with open(s4_perms_fn_pkl, "rb") as file:
                self.s4_perms = pickle.load(file)
            end = time.time()
            print(f"ModelComps loading s4_perms: {end - start}")
        else:
            self.s4_perms = []
            self.eval_s4_perms()
            end = time.time()
            print(f"ModelComps computing s4_perms: {end - start}")
            with open(s4_perms_fn_pkl, "wb") as file:
                pickle.dump(self.s4_perms, file)
            end = time.time()
            print(f"ModelComps saving s4_perms as pickle: {end - start}")

        # Create list with indices of all probs for each hide
        start = time.time()
        self.s4_perm_node_indices = {}
        for node in range(self.n_nodes):
            self.s4_perm_node_indices[node] = [
                index
                for index, s4_perm in enumerate(self.s4_perms)
                if s4_perm[node] == 1
            ]
            # --> 25 X 42504 indices per hide ( if 25 nodes and 6 hides)
        end = time.time()
        print(f"ModelComps computing s4_marg_indices: {end - start}")

        # Evaluate number of s4 permutations
        self.n_s4_perms = len(self.s4_perms)

        # Load or evaluate agent's initial belief state in first trials ---(Prior)---
        start = time.time()
        prior_fn = os.path.join(
            self.working_dir, "utilities", f"prior_dim-{self.dim}_h{self.n_hides}.npy"
        )
        if os.path.exists(prior_fn):
            self.prior_c0 = np.load(prior_fn)
            end = time.time()
            print(f"ModelComps loading prior: {end - start}")
            sum_p_c0 = np.sum(self.prior_c0)
        else:
            self.prior_c0 = np.full((self.n_nodes, self.n_s4_perms), 0.0)
            self.eval_prior()
            np.save(prior_fn, self.prior_c0)
            end = time.time()
            print(f"ModelComps computing prior: {end - start}")

        # Load or evaluate action-dependent state-conditional observation distribution ---(Likelihood)---
        start = time.time()
        lklh_fn = os.path.join(
            self.working_dir, "utilities", f"lklh_dim-{self.dim}_h{self.n_hides}.npy"
        )
        if os.path.exists(lklh_fn):
            self.lklh = np.load(lklh_fn)
            end = time.time()
            print(f"ModelComps loading likhl: {end - start}")
        else:

            self.lklh = np.full(
                (2, self.n_nodes, 3, 4, self.n_nodes, self.n_s4_perms), 0.0
            )
            self.eval_likelihood()  # TODO: gets killed; figure out alternative
            np.save(lklh_fn, self.lklh)
            end = time.time()
            print(f"ModelComps computing likhl: {end - start}")
        start = time.time()

    def eval_s4_perms(self):
        """Evaluate permutations of s4 states"""
        s_4_values = [0] * (self.n_nodes - self.n_hides)
        s_4_values.extend([1] * self.n_hides)
        self.s4_perms = sorted(more_itertools.distinct_permutations(s_4_values))

    def eval_prior(self):
        """Evaluate agent's state priors"""

        for s3 in range(self.n_nodes):
            for index, s4_perm in enumerate(self.s4_perms):

                if s4_perm[s3] == 1:
                    self.prior_c0[s3, index] = 1 / (self.n_s4_perms * self.n_hides)
                    # self.prior_c0[s3, index] = 1 / 1062600

        sum = np.sum(self.prior_c0)
        stop = 4

    def eval_likelihood(self):
        """Evaluate action-dependent state-conditional observation distribution p(o|s) (likelihood),
        seperately for action = 0 and action not 0"""

        # # Loop through s4_permutations:
        # for index, s4_perm in enumerate(self.s4_perms):
        #
        #     # Loop through through s1 values
        #     for s1 in range(self.n_nodes):
        #
        #         # ---------for all a = 0---------------
        #
        #         # If s4[s1] == 0 (not hiding spot), lklh(o == 1 (grey)) = 1, else remain zero
        #         if s4_perm[s1] == 0:
        #             self.lklh[0, s1, 1, :, index] = 1
        #
        #         # If s4[s1] == 1 (hiding spot), lklh( o == 2 (blue)) = 1, else remain zero
        #         if s4_perm[s1] == 1:
        #             self.lklh[0, s1, 2, :, index] = 1
        #
        #         # ---------for all a = 1---------------
        #
        #         # For s3 == s1, lklh(o == 0 (black)) = 0, else lklh(o == 0 (black)) = 1
        #         self.lklh[1, s1, 0, :, index] = 1
        #         self.lklh[1, s1, 0, s1, index] = 0
        #
        #         # For s3 == s1, lklh(o == 1 (grey)) = 0, else lklh(o == 1 (grey)) = 1
        #         self.lklh[1, s1, 1, :, index] = 1
        #         self.lklh[1, s1, 1, s1, index] = 0
        #
        #         # For s3 == s1, liklh(o == 2 (blue)) = 0, else lklh(o==2 (blue) = 1
        #         self.lklh[1, s1, 2, :, index] = 1
        #         self.lklh[1, s1, 2, s1, index] = 0
        #
        #         # For s3 == 1, liklh(o == 3 (treasure)) = 1, else remain zero
        #         self.lklh[1, s1, 3, s1, index] = 1

        # Loop through s4_permutations:
        for index, s4_perm in enumerate(self.s4_perms):

            # Loop through through s1 values
            for s1 in range(self.n_nodes):

                # ---------for all a = 0---------------

                # If s4[s1] == 0 (not hiding spot), lklh(o == 1 (grey)) = 1, else remain zero
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
                    s2_s1 = 2
                    # bg color blue is impossible for s4_s1=0

                # If s4[s1] == 1 (hiding spot), lklh( o == 2 (blue)) = 1, else remain zero
                if s4_perm[s1] == 1:

                    # for s2[s1] == 0 (black)
                    # -----------------------
                    s2_s1 = 0
                    # will deterministically turn to blue since s4_s1=1
                    self.lklh[0, s1, s2_s1, 2, :, index] = 1
                    self.lklh[0, s1, s2_s1, 2, s1, index] = 0

                    # for s2[s1] == 1 (grey)
                    # -----------------------
                    s2_s1 = 1
                    # grey node bg color impossible for s4_s1=1

                    # for s2[s1] == 2 (blue)
                    # -----------------------
                    # will return same color as already unvealed
                    s2_s1 = 2
                    self.lklh[0, s1, s2_s1, 2, :, index] = 1
                    self.lklh[0, s1, s2_s1, 2, s1, index] = 0

                # ---------for all a = 1---------------

                # If s4[s1] == 0 (not hiding spot)
                if s4_perm[s1] == 0:
                    # for s2[s1] == 0 (black)
                    # -----------------------
                    s2_s1 = 0

                    # For s3 == s1, lklh(o == 0 (black)) = 0, else lklh(o == 0 (black)) = 1
                    self.lklh[1, s1, s2_s1, 0, :, index] = 1
                    self.lklh[1, s1, s2_s1, 0, s1, index] = 0

                    # all other observations ( o==1, o==2, o==3 remain 0)

                    # for s2[s1] == 1 (grey)
                    # -----------------------
                    s2_s1 = 1

                    # For s3 == s1, lklh(o == 1 (grey)) = 0, else lklh(o == 1 (grey)) = 1
                    self.lklh[1, s1, s2_s1, 1, :, index] = 1
                    self.lklh[1, s1, s2_s1, 1, s1, index] = 0

                    # all other observations ( o==0, o==2, o==3 remain 0)

                    # for s2[s1] == 2 (blue)
                    # -----------------------
                    s2_s1 = 2
                    # node color blue is impossible

                # If s4[s1] == 1 (node is a hiding spot)
                if s4_perm[s1] == 1:
                    # for s2[s1] == 0 (black)
                    # -----------------------
                    s2_s1 = 0

                    # For s3 == s1, lklh(o == 0 (black)) = 0, else lklh(o == 0 (black)) = 1
                    self.lklh[1, s1, s2_s1, 0, :, index] = 1
                    self.lklh[1, s1, s2_s1, 0, s1, index] = 0

                    # For s3 == 1, liklh(o == 3 (treasure)) = 1, else remain zero
                    self.lklh[1, s1, s2_s1, 3, s1, index] = 1

                    # all other observations ( o==1, o==2 remain 0)

                    # for s2[s1] == 1 (grey)
                    # -----------------------
                    s2_s1 = 1

                    # observation grey impossible --> all zero

                    # for s2[s1] == 2 (blue)
                    # -----------------------
                    s2_s1 = 2

                    # For s3 == s1, liklh(o == 2 (blue)) = 0, else lklh(o==2 (blue) = 1
                    self.lklh[1, s1, s2_s1, 2, :, index] = 1
                    self.lklh[1, s1, s2_s1, 2, s1, index] = 0

                    # For s3 == 1, liklh(o == 3 (treasure)) = 1, else remain zero
                    self.lklh[1, s1, s2_s1, 3, s1, index] = 1
