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
    agent_model_candidate_space = ["C1", "C2", "C3"] #, "A1", "A2", "A3"]
    tau_bf_cand_space = np.linspace(0.01, 0.5, 5)
    lambda_bf_cand_space = np.linspace(0.1, 0.9, 5)


class ParamAndModelRecoverer:
    """A class to evaluate Maximum Likelihood parameters estimations"""
    est_params: RecoveryParameters = RecoveryParameters()
    sim_object: Simulator

    current_cand_agent: str
    tau_est_result_gen_agent: float = np.nan
    tau_est_result_current_cand_agent: float = np.nan
    lambda_est_result_gen_agent: float = np.nan
    lambda_est_result_current_cand_agent: float = np.nan
    llh_theta_hat_gen_agent: float = np.nan
    llh_theta_hat_current_cand_agent: float = np.nan

    def instantiate_sim_obj(self, exp_data, task_configs: TaskConfigurator,
                            bayesian_comps: BayesianModelComps):
        """
        Parameters
        ----------
        sim_object: Simulator
        """
        self.sim_object = Simulator(task_configs, bayesian_comps)
        self.sim_object.data = exp_data

    def reset_result_variables_to_nan(self):
        self.tau_est_result_gen_agent = np.nan
        self.tau_est_result_current_cand_agent = np.nan
        self.lambda_est_result_gen_agent = np.nan
        self.lambda_est_result_current_cand_agent = np.nan
        self.llh_theta_hat_gen_agent = np.nan
        self.llh_theta_hat_current_cand_agent = np.nan

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
            np.argmin(neg_llh_function)]
        end_est_total = time.time()
        print(f"Finined estimation in "
              f"{round(end_est_total - start_est_total,ndigits=2)} sec.")

        if self.current_cand_agent == self.sim_object.data.loc["agent"][0]:
            self.tau_est_result_gen_agent = maximum_likelihood_tau
            self.llh_theta_hat_gen_agent = np.min(neg_llh_function)
        else:
            self.tau_est_result_current_cand_agent = maximum_likelihood_tau
            self.llh_theta_hat_current_cand_agent = np.min(neg_llh_function)

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

        if self.current_cand_agent == self.sim_object.data.iloc[0]["agent"]:
            self.tau_est_result_gen_agent = max_llh_tau
            self.lambda_est_result_gen_agent = max_llh_lambda
            self.llh_theta_hat_gen_agent = np.min(neg_llh_function)
        else:
            self.tau_est_result_current_cand_agent = max_llh_tau
            self.lambda_est_result_current_cand_agent = max_llh_lambda
            self.llh_theta_hat_current_cand_agent = np.min(
                neg_llh_function)

    def estimate_tau(self, method: str):

        if method == "brute_force":
            self.eval_brute_force_est_tau()

    def estimate_tau_lambda(self, method: str):
        """Estimate two-dimensional parameter vektor, tau and lambda"""

        if method == "brute_force":
            self.eval_brute_force_tau_lambda()

    def estimate_parameters(self, method: str):
        if (np.isnan(self.sim_object.sim_params.current_tau_gen)
                and np.isnan(self.sim_object.sim_params.current_lambda_gen)):
            pass  # TODO check, if before already np.nan

        elif np.isnan(self.sim_object.sim_params.current_lambda_gen):
            self.estimate_tau(method=method)

        else:
            self.estimate_tau_lambda(method=method)

    def eval_bic_giv_theta_hat(self,
                               llh_theta_hat: float,
                               n_params: int,
                               n_valid_actions: int):

        this_bic = llh_theta_hat - n_params/2 * np.log(n_valid_actions)
        return this_bic

    def evaluate_bic_s(self, est_method: str):

        agent_specific_bic_s = np.ful(
            len(self.est_params.agent_model_candidate_space),
            np.nan)

        for i, agent_model in enumerate(
                self.est_params.agent_model_candidate_space):
            self.current_cand_agent = agent_model

            if "C" in agent_model:
                n_params = 0
            elif agent_model == "A3":
                n_params = 2
            else:
                n_params = 1

            n_valid_choices = self.sim_object.data.a.count()

            if agent_model == self.sim_object.data.iloc[0]["agent"]:
                llh_theta_hat = self.llh_theta_hat_gen_agent

            else:
                self.estimate_parameters(method=est_method)
                llh_theta_hat = self.llh_theta_hat_current_cand_agent

            agent_specific_bic_s[i] = self.eval_bic_giv_theta_hat(
                llh_theta_hat=llh_theta_hat,
                n_params=n_params,
                n_valid_actions=n_valid_choices)

        return agent_specific_bic_s
