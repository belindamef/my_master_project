"""This script contains classes and methods to evaluate Maximum Likelihood
estimations of model parameters

Author: Belinda Fleischmann
"""
import time
import numpy as np
from utilities.simulation_methods import Simulator
from utilities.modelling import BayesianModelComps
from utilities.config import TaskConfigurator



class RecoveryParameters:
    agent_model_candidate_space = ["C1", "C2", "C3", "A1", "A2", "A3"]
    tau_bf_cand_space = np.arange(0.01, 0.51, 0.1)  # TODO: more elegant way!
    lambda_bf_cand_space = np.linspace(0.1, 0.9, 5)
    current_tau_analyze: float = None
    current_lambda_analyze: float = None
    tau_analyze_if_fixed = 0.1

    def get_params_from_args(self):
        return self


class ParamAndModelRecoverer:
    """A class to evaluate Maximum Likelihood parameters estimations"""
    est_params: RecoveryParameters = RecoveryParameters()
    sim_object: Simulator

    def instantiate_sim_obj(self, exp_data, task_configs: TaskConfigurator,
                            bayesian_comps: BayesianModelComps):
        """
        Parameters
        ----------
        sim_object: Simulator
        """
        self.sim_object = Simulator(task_configs, bayesian_comps)
        self.sim_object.data = exp_data

    def eval_llh_function_tau(self):
        """Evaluate log_likelihood function for given tau parameter space, and
        when lambda is not applicable.
        """

        loglikelihood_function = np.full(
            len(self.est_params.tau_bf_cand_space), np.nan)

        for i, tau_i in np.ndenumerate(self.est_params.tau_bf_cand_space):
            this_tau_s_llh = self.sim_object.sim_to_eval_llh(
                candidate_tau=tau_i,
                candidate_lambda=np.nan)

            loglikelihood_function[i] = this_tau_s_llh

        return loglikelihood_function

    # def eval_llh_function_lambda(self):
    #     """Evaluate log_likelihood function for given lambda parameter space,
    #      fixed tau value and simulated dataset.
    #     """

    #     loglikelihood_function = np.full(
    #         len(self.est_params.lambda_bf_cand_space), np.nan)

    #     for i, lambda_i in np.ndenumerate(self.est_params.lambda_bf_cand_space):
    #         this_lambda_s_llh = self.sim_object.sim_to_eval_llh(
    #             self.est_params.tau_analyze_if_fixed,
    #             lambda_i)

    #         loglikelihood_function[i] = this_lambda_s_llh

    #     return loglikelihood_function

    def eval_llh_function_tau_and_lambda(self):
        """Evaluate log_likelihood function for given 2-dimdensional tau and
        lambda space."""

        loglikelihood_function = np.full(
            (len(self.est_params.tau_bf_cand_space),
             len(self.est_params.lambda_bf_cand_space)),
            np.nan)

        for i_tau, tau_i in np.ndenumerate(self.est_params.tau_bf_cand_space):

            for i_lambda, lambda_i in np.ndenumerate(
                self.est_params.lambda_bf_cand_space):

                this_theta_s_llh = self.sim_object.sim_to_eval_llh(
                    candidate_tau=tau_i,
                    candidate_lambda=lambda_i
                )

                loglikelihood_function[i_tau, i_lambda] = this_theta_s_llh
        
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
            np.argmin(neg_llh_function)]  # TODO möglich, mehrere Minima zu haben? 
        end_est_total = time.time()
        print(f"Finined estimation in "
              f"{round(end_est_total - start_est_total,ndigits=2)} sec.")
        return maximum_likelihood_tau

    def eval_brute_force_tau_lambda(self) -> tuple[float, float]:
        """Evaluate the maximum likelihood estimation of the decision noise
        parameter tau and weighting parameter lambda based on dataset of one
        participant with brute force method.
        """

        loglikelihood_function = self.eval_llh_function_tau_and_lambda()

        # Identify theta=(tau,lambda) with max likelihood, i.e. min neg logL
        neg_llh_function = -loglikelihood_function
        min_neg_llh_two_dim_index = np.unravel_index(
            neg_llh_function.argmin(), neg_llh_function.shape)
        min_neg_llh_tau_index = min_neg_llh_two_dim_index[0]
        min_neg_llh_lambda_index = min_neg_llh_two_dim_index[1]

        max_llh_tau = self.est_params.tau_bf_cand_space[min_neg_llh_tau_index]
        max_llh_lambda = self.est_params.lambda_bf_cand_space[
            min_neg_llh_lambda_index]

        return max_llh_tau, max_llh_lambda

    # def eval_brute_force_est_lambda(self) -> float:
    #     print("Starting brute-force estimation for lambda")
    #     start_est_total = time.time()
    #     lambda_candidate_space = self.est_params.lambda_bf_cand_space
    #     loglikelihood_function = self.eval_llh_function_lambda()

    #     # Identify tau with maximum likelihood, i.e. min. neg. log likelihood
    #     neg_llh_function = - loglikelihood_function
    #     maximum_likelihood_lambda = lambda_candidate_space[np.argmin(
    #         neg_llh_function)]
    #     end_est_total = time.time()
    #     print(f"Finined estimation in "
    #           f"{round(end_est_total - start_est_total,ndigits=2)} sec.")
    #     return maximum_likelihood_lambda

    def estimate_tau(self, method: str) -> float:

        if method == "brute_force":
            tau_estimate = self.eval_brute_force_est_tau()

        return tau_estimate

    def estimate_tau_lambda(self, method: str):
        """Estimate two-dimensional parameter vektor, tau and lambda"""

        if method == "brute_force":
            tau_estimate, lambda_estimate = self.eval_brute_force_tau_lambda()

        return tau_estimate, lambda_estimate

    def evaluate_BICs(self):

        for agent_model in self.est_params.agent_model_candidate_space:

            stop = "here"
