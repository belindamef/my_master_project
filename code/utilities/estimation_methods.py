"""This script contains classes and methods to evaluate Maximum Likelihood
estimations of model parameters

Author: Belinda Fleischmann
"""
import time
import numpy as np
from utilities.simulation_methods import Simulator


class EstimationParams:
    tau_bf_cand_space = np.arange(0.1, 2., 0.4)
    lambda_bf_cand_space = np.linspace(0.1, 0.9, 5)
    current_tau_analyze: float = None
    current_lambda_analyze: float = None
    tau_analyze_if_fixed = 0.1
    lambda_analyze_if_fixed = 0.5

    def get_params_from_args(self):
        return self


class ParameterEstimator:
    """A class to evaluate Maximum Likelihood parameters estimations"""
    est_params: EstimationParams = EstimationParams()
    sim_object: Simulator

    def instantiate_sim_obj(self, exp_data, task_configs, bayesian_comps):
        """
        Parameters
        ----------
        sim_object: Simulator
        """
        self.sim_object = Simulator(task_configs, bayesian_comps)
        self.sim_object.data = exp_data

    def eval_llh_function_tau(self):
        """Evaluate log_likelihood function for given tau parameter space, a 
        fixed lambda value and simulated dataset.
        """

        loglikelihood_function = np.full(
            len(self.est_params.tau_bf_cand_space), np.nan)

        for i, tau_i in np.ndenumerate(self.est_params.tau_bf_cand_space):
            this_tau_s_llh = self.sim_object.sim_to_eval_llh(
                tau_i,
                self.est_params.lambda_analyze_if_fixed)

            loglikelihood_function[i] = this_tau_s_llh

        return loglikelihood_function

    def eval_llh_function_lambda(self):
        """Evaluate log_likelihood function for given lambda parameter space,
         fixed tau value and simulated dataset.
        """

        loglikelihood_function = np.full(
            len(self.est_params.lambda_bf_cand_space), np.nan)

        for i, lambda_i in np.ndenumerate(self.est_params.lambda_bf_cand_space):
            this_lambda_s_llh = self.sim_object.sim_to_eval_llh(
                self.est_params.tau_analyze_if_fixed,
                lambda_i)

            loglikelihood_function[i] = this_lambda_s_llh

        return loglikelihood_function

    def eval_brute_force_est_tau(self) -> float:
        """Evaluate the maximum likelihood estimation of the decision noise
        parameter tau  based on dataset of one participant with brute force
        method.
        """
        start_est_total = time.time()

        loglikelihood_function = self.eval_llh_function_tau()

        # Identify tau with maximum likelihood, i.e. min. neg. log likelihood
        neg_llh_function = - loglikelihood_function
        maximum_likelihood_tau = self.est_params.tau_bf_cand_space[
            np.argmin(neg_llh_function)]
        end_est_total = time.time()
        print(f"Finined estimation in "
              f"{round(end_est_total - start_est_total,ndigits=2)} sec.")
        return maximum_likelihood_tau

    def eval_brute_force_est_lambda(self) -> float:
        print("Starting brute-force estimation for lambda")
        start_est_total = time.time()
        lambda_candidate_space = self.est_params.lambda_bf_cand_space
        loglikelihood_function = self.eval_llh_function_lambda()

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

    def estimate_parameters(self, method: str):
        if method == "brute_force":
            self.eval_brute_force_estimates()