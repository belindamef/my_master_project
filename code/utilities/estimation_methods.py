"""This script contains classes and methods to evaluate Maximum Likelihood
estimations of model parameters

Author: Belinda Fleischmann
"""
import time
import numpy as np
from utilities.simulation_methods import Simulator


class ParameterEstimator:
    """A class to evaluate Maximum Likelihood parameters estimations"""
    sim_object: Simulator

    def instantiate_sim_obj(self, exp_data, task_configs, bayesian_comps):
        """
        Parameters
        ----------
        sim_object: Simulator
        """
        self.sim_object = Simulator(task_configs, bayesian_comps)
        self.sim_object.data = exp_data

    def eval_llh_function_tau(self, parameter_space):
        """Evaluate log_likelihood function for given parameter space and
        simulated dataset.
        """

        loglikelihood_function = np.full(len(parameter_space), np.nan)

        for i, tau_i in np.ndenumerate(parameter_space):
            # TODO: where define lambda?
            this_tau_s_llh = self.sim_object.sim_to_eval_llh(tau_i,
                                                             0.5)

            loglikelihood_function[i] = this_tau_s_llh

        return loglikelihood_function

    def eval_llh_function_lambda(self, parameter_space):
        """Evaluate log_likelihood function for given parameter space and
        simulated dataset.
        """

        loglikelihood_function = np.full(len(parameter_space), np.nan)

        for i, lambda_i in np.ndenumerate(parameter_space):
            # TODO: where to define tau? hardcoded here
            this_lambda_s_llh = self.sim_object.sim_to_eval_llh(1.0,
                                                                lambda_i)

            loglikelihood_function[i] = this_lambda_s_llh

        return loglikelihood_function

    def eval_brute_force_est_tau(self) -> float:
        """Evaluate the maximum likelihood estimation of the decision noise
        parameter based on dataset of one participant with brute force method.
        """
        print("Starting brute-force estimation for tau")
        start_est_total = time.time()
        tau_candidate_space = np.linspace(0.01, 2, 20)
        loglikelihood_function = self.eval_llh_function_tau(
            parameter_space=tau_candidate_space)

        # Identify tau with maximum likelihood, i.e. min. neg. log likelihood
        neg_llh_function = - loglikelihood_function
        maximum_likelihood_tau = tau_candidate_space[np.argmin(
            neg_llh_function)]
        end_est_total = time.time()
        print(f"Finined estimation in "
              f"{round(end_est_total - start_est_total,ndigits=2)} sec.")
        return maximum_likelihood_tau

    def eval_brute_force_est_lambda(self) -> float:
        print("Starting brute-force estimation for lambda")
        start_est_total = time.time()
        lambda_candidate_space = np.linspace(0.1, 0.9, 20)
        loglikelihood_function = self.eval_llh_function_lambda(
            parameter_space=lambda_candidate_space)

        # Identify tau with maximum likelihood, i.e. min. neg. log likelihood
        neg_llh_function = - loglikelihood_function
        maximum_likelihood_lambda = lambda_candidate_space[np.argmin(
            neg_llh_function)]
        end_est_total = time.time()
        print(f"Finined estimation in "
              f"{round(end_est_total - start_est_total,ndigits=2)} sec.")
        return maximum_likelihood_lambda

    def estimate_tau(self, method: str) -> float:

        if method == "brute_force":
            tau_estimate = self.eval_brute_force_est_tau()

        return tau_estimate

    def estimate_lambda(self, method: str) -> float:

        if method == "brute_force":
            lambda_estimate = self.eval_brute_force_est_lambda()
        return lambda_estimate

    def eval_brute_force_estimates(self):
        print("Starting brute-force estimation")
        #tau_candidate_space = 

    def estimate_parameters(self, method:str):
        if method == "brute_force":
            self.eval_brute_force_estimates()