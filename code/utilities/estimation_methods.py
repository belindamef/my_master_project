"""This script contains classes and methods to evaluate Maximum Likelihood
estimations of model parameters

Author: Belinda Fleischmann
"""
import time
import numpy as np
from utilities.simulation_methods import Simulator


class ParameterEstimator:
    """A class to evaluate Maximum Likelihood estimations parameters"""

    def __init__(self, exp_data, task_configs, bayesian_comps):
        """

        Parameters
        ----------
        sim_object: Simulator
        """
        self.data = exp_data
        self.sim_object = Simulator(task_configs, bayesian_comps)

    def eval_llh_function(self, parameter_space):
        """Evaluate log_likelihood function for given parameter space and
        simulated dataset.
        """

        loglikelihood_function = np.full(len(parameter_space), np.nan)

        for i, tau_i in np.ndenumerate(parameter_space):

            this_tau_s_llh = self.sim_object.sim_interaction_to_eval_llh(tau_i)

            loglikelihood_function[i] = this_tau_s_llh

        return loglikelihood_function

    def eval_brute_force_est(self) -> float:
        """Evaluate the maximum likelihood estimation of the decision noise
        parameter based on dataset of one participant with brute force method.
        """
        print("Starting brute-force estimation")
        start_est_total = time.time()
        tau_candidate_space = np.linspace(0.01, 2, 50)
        loglikelihood_function = self.eval_llh_function(
            parameter_space=tau_candidate_space)

        # Identify tau with maximum likelihood, i.e. min. neg. log likelihood
        neg_llh_function = - loglikelihood_function
        maximum_likelihood_tau = tau_candidate_space[np.argmin(
            neg_llh_function)]
        end_est_total = time.time()
        print(f"Finined estimation in "
              f"{round(end_est_total - start_est_total,ndigits=2)} sec.")
        return maximum_likelihood_tau

    def estimate_tau(self, method: str) -> float:
        """Evaluate ML estimate of tau parameter as average over all trials"""

        if method == "brute_force":
            tau_estimate = self.eval_brute_force_est()

        return tau_estimate
