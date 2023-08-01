"""This script contains classes and methods to evaluate Maximum Likelihood
estimations of model parameters

Author: Belinda Fleischmann
"""
import numpy as np
from utilities.simulation_methods import Simulator
from utilities.modelling import BayesianModelComps
from utilities.config import TaskConfigurator


class RecoveryParameters:
    agent_candidate_space: list
    tau_bf_cand_space: list
    lambda_bf_cand_space: list

    def def_params_manually(self, agent_candidate_space: list = None,
                            tau_bf_cand_space: list = None,
                            lambda_bf_cand_space: list = None):

        if agent_candidate_space is None:
            self.agent_candidate_space = ["C1", "C2", "C3", "A1", "A2", "A3"]
        else:
            self.agent_candidate_space = agent_candidate_space

        if tau_bf_cand_space is None:
            self.tau_bf_cand_space = np.linspace(0.01, 0.5, 20).tolist()
        else:
            self.tau_bf_cand_space = tau_bf_cand_space

        if lambda_bf_cand_space is None:
            self.lambda_bf_cand_space = np.linspace(0, 1, 20).tolist()
        else:
            self.lambda_bf_cand_space = lambda_bf_cand_space


class ParamAndModelRecoverer:
    """A class to evaluate Maximum Likelihood parameters estimations"""
    recov_params: RecoveryParameters = RecoveryParameters()
    sim_object: Simulator

    current_cand_agent: str
    tau_est_result_gen_agent: float = np.nan
    tau_est_result_current_cand_agent: float = np.nan
    lambda_est_result_gen_agent: float = np.nan
    lambda_est_result_current_cand_agent: float = np.nan
    llh_noparam_current_cand_agent: float = np.nan
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
        self.llh_noparam_current_cand_agent = np.nan
        self.llh_theta_hat_gen_agent = np.nan
        self.llh_theta_hat_current_cand_agent = np.nan

    def eval_llh_data_no_params(self, candidate_agent):
        llh = self.sim_object.sim_to_eval_llh(
            candidate_tau=np.nan,
            candidate_lambda=np.nan,
            candidate_agent=candidate_agent
        )
        self.llh_noparam_current_cand_agent = llh

    def eval_llh_function_tau(self, candidate_agent: str):
        """Evaluate log_likelihood function for given tau parameter space, and
        when lambda is not applicable.
        """

        loglikelihood_function = np.full(
            len(self.recov_params.tau_bf_cand_space), np.nan)

        for i, tau_i in np.ndenumerate(self.recov_params.tau_bf_cand_space):
            this_tau_s_llh = self.sim_object.sim_to_eval_llh(
                candidate_tau=tau_i,
                candidate_lambda=np.nan,
                candidate_agent=candidate_agent)

            loglikelihood_function[i] = this_tau_s_llh

        return loglikelihood_function

    def eval_llh_function_tau_and_lambda(self, candidate_agent: str):
        """Evaluate log_likelihood function for given 2-dimdensional tau and
        lambda space."""

        loglikelihood_function = np.full(
            (len(self.recov_params.tau_bf_cand_space),
             len(self.recov_params.lambda_bf_cand_space)),
            np.nan)

        for i_tau, tau_i in np.ndenumerate(
                self.recov_params.tau_bf_cand_space):

            for i_lambda, lambda_i in np.ndenumerate(
                    self.recov_params.lambda_bf_cand_space):

                this_theta_s_llh = self.sim_object.sim_to_eval_llh(
                    candidate_tau=tau_i,
                    candidate_lambda=lambda_i,
                    candidate_agent=candidate_agent
                )

                loglikelihood_function[i_tau, i_lambda] = this_theta_s_llh

        return loglikelihood_function

    def eval_brute_force_est_tau(self, candidate_agent: str) -> float:
        """Evaluate the maximum likelihood estimation of the decision noise
        parameter tau  based on dataset of one participant with brute force
        method.
        """

        llh_function = self.eval_llh_function_tau(
            candidate_agent=candidate_agent)

        # Identify tau with maximum likelihood
        maximum_likelihood_tau = self.recov_params.tau_bf_cand_space[
            np.argmax(llh_function)]

        if self.current_cand_agent == self.sim_object.data.iloc[0]["agent"]:
            self.tau_est_result_gen_agent = maximum_likelihood_tau
            self.llh_theta_hat_gen_agent = np.max(llh_function)
        else:
            self.tau_est_result_current_cand_agent = maximum_likelihood_tau
            self.llh_theta_hat_current_cand_agent = np.max(llh_function)

    def eval_brute_force_tau_lambda(self, candidate_agent: str):
        """Evaluate the maximum likelihood estimation of the decision noise
        parameter tau and weighting parameter lambda based on dataset of one
        participant with brute force method.
        """

        llh_function = self.eval_llh_function_tau_and_lambda(
            candidate_agent=candidate_agent)

        # Identify theta=(tau,lambda) with max likelihood
        max_llh_two_dim_index = np.unravel_index(
            llh_function.argmax(), llh_function.shape)
        max_llh_tau_index = max_llh_two_dim_index[0]
        max_llh_lambda_index = max_llh_two_dim_index[1]

        max_llh_tau = self.recov_params.tau_bf_cand_space[max_llh_tau_index]
        max_llh_lambda = self.recov_params.lambda_bf_cand_space[
            max_llh_lambda_index]

        if self.current_cand_agent == self.sim_object.data.iloc[0]["agent"]:
            self.tau_est_result_gen_agent = max_llh_tau
            self.lambda_est_result_gen_agent = max_llh_lambda
            self.llh_theta_hat_gen_agent = np.max(llh_function)
        else:
            self.tau_est_result_current_cand_agent = max_llh_tau
            self.lambda_est_result_current_cand_agent = max_llh_lambda
            self.llh_theta_hat_current_cand_agent = np.min(llh_function)

    def estimate_tau(self, method: str, candidate_agent: str):
        # TODO: llh comutation is a side product! NOT NICE, no no

        if method == "brute_force":
            self.eval_brute_force_est_tau(candidate_agent=candidate_agent)

    def estimate_tau_lambda(self, method: str, candidate_agent: str):
        """Estimate two-dimensional parameter vektor, tau and lambda"""

        if method == "brute_force":
            self.eval_brute_force_tau_lambda(candidate_agent=candidate_agent)

    def estimate_parameters(self, method: str, candidate_agent: str):
        if (np.isnan(self.sim_object.sim_params.current_tau_gen)
                and np.isnan(self.sim_object.sim_params.current_lambda_gen)):
            pass

        elif np.isnan(self.sim_object.sim_params.current_lambda_gen):
            self.estimate_tau(method=method, candidate_agent=candidate_agent)

        else:
            self.estimate_tau_lambda(method=method,
                                     candidate_agent=candidate_agent)

    def eval_llh_data(self, candidate_agent, method: str):

        if "C" in candidate_agent:
            self.eval_llh_data_no_params(
                candidate_agent=candidate_agent)

        elif candidate_agent in ["A1", "A2"]:
            self.estimate_tau(method=method,
                              candidate_agent=candidate_agent)

        elif candidate_agent == "A3":
            self.estimate_tau_lambda(method=method,
                                     candidate_agent=candidate_agent)

    def eval_bic_giv_theta_hat(self,
                               llh_theta_hat: float,
                               n_params: int,
                               n_valid_actions: int):

        this_bic = llh_theta_hat - n_params/2 * np.log(n_valid_actions)
        return this_bic

    def evaluate_bic_s(self, est_method: str) -> dict:
        agent_specific_bic_s = {
            "BIC_C1": np.nan, "BIC_C2": np.nan, "BIC_C3": np.nan,
            "BIC_A1": np.nan, "BIC_A2": np.nan, "BIC_A3": np.nan
        }

        for agent_model in self.recov_params.agent_candidate_space:
            self.current_cand_agent = agent_model

            if "C" in agent_model:
                n_params = 0
            elif agent_model == "A3":
                n_params = 2
            else:
                n_params = 1

            n_valid_choices = self.sim_object.data.a.count()

            if "C" in agent_model:

                self.eval_llh_data(candidate_agent=agent_model,
                                   method=est_method)
                llh_data = self.llh_noparam_current_cand_agent

            elif "A" in agent_model:

                if agent_model == self.sim_object.data.iloc[0]["agent"]:
                    llh_data = self.llh_theta_hat_gen_agent

                else:
                    self.eval_llh_data(candidate_agent=agent_model,
                                       method=est_method)
                    llh_data = self.llh_theta_hat_current_cand_agent

            agent_specific_bic_s[
                f"BIC_{agent_model}"] = self.eval_bic_giv_theta_hat(
                llh_theta_hat=llh_data,
                n_params=n_params,
                n_valid_actions=n_valid_choices)

        return agent_specific_bic_s
