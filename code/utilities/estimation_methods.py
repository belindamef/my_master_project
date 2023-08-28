"""This script contains classes and methods to evaluate Maximum Likelihood
estimations of model parameters

Author: Belinda Fleischmann
"""
import numpy as np
import pandas as pd
from utilities.simulation_methods import Simulator
from utilities.agent import BayesianModelComps
from utilities.config import TaskConfigurator


class EstimationParameters:
    """Class to store and manage candidate model and parameter spaces for model
    recorvery
    """
    agent_candidate_space: list
    tau_bf_cand_space: list
    lambda_bf_cand_space: list

    def get_params_from_args(self, args):
        """Method to fetch simulation parameters from command line or bash
        script arguments."""
        tau_cand_res = args.tau_cand_res
        lambda_cand_res = args.lambda_cand_res

        #self.agent_candidate_space = args.agent_model
        self.tau_bf_cand_space = np.linspace(0.01, 0.5, tau_cand_res).tolist()
        self.lambda_bf_cand_space = np.linspace(0, 1, lambda_cand_res).tolist()
        return self

    def def_params_manually(self, agent_candidate_space=None,
                            tau_bf_cand_space=None,
                            lambda_bf_cand_space=None):
        """_summary_

        Args:
            agent_candidate_space (list, optional): _description_. Defaults to
                None.
            tau_bf_cand_space (list, optional): _description_.
                Defaults to None.
            lambda_bf_cand_space (list, optional): _description_. Defaults to
                None.
        """
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


class Estimator:
    """A class to evaluate Maximum Likelihood parameters estimations"""
    sim_object: Simulator

    current_cand_agent: str
    tau_est_result_gen_agent: float = np.nan
    tau_est_result_current_cand_agent: float = np.nan
    lambda_est_result_gen_agent: float = np.nan
    lambda_est_result_current_cand_agent: float = np.nan
    max_llh_current_gen_agent: float = np.nan
    max_llh_current_cand_agent: float = np.nan

    def __init__(self, estim_params: EstimationParameters):
        self.est_params = estim_params

    def instantiate_sim_obj(self, task_configs: TaskConfigurator,
                            bayesian_comps: BayesianModelComps):
        """
        Parameters
        ----------
        sim_object: Simulator
        """
        self.sim_object = Simulator(task_configs, bayesian_comps)

    def reset_result_variables_to_nan(self):
        """Name says it all
        """
        self.tau_est_result_gen_agent = np.nan
        self.tau_est_result_current_cand_agent = np.nan
        self.lambda_est_result_gen_agent = np.nan
        self.lambda_est_result_current_cand_agent = np.nan
        self.max_llh_current_gen_agent = np.nan
        self.max_llh_current_cand_agent = np.nan

    def eval_llh_data_no_params(self, candidate_agent: str,
                                data: pd.DataFrame):
        """Method to ecalute the "likelihood" of data if no parameters given

        Args:
            candidate_agent (str): Agent name
            data (pd.DataFrame): Behavioral data, trialwise events
        """
        llh = self.sim_object.sim_to_eval_llh(
            candidate_tau=np.nan,
            candidate_lambda=np.nan,
            candidate_agent=candidate_agent,
            data=data
        )
        self.max_llh_current_cand_agent = llh

    def eval_llh_function_tau(self, candidate_agent: str, data: pd.DataFrame):
        """Evaluate log_likelihood function for given tau parameter space, and
        when lambda is not applicable.
        """

        loglikelihood_function = np.full(
            len(self.est_params.tau_bf_cand_space), np.nan)

        for i, tau_i in np.ndenumerate(self.est_params.tau_bf_cand_space):
            this_tau_s_llh = self.sim_object.sim_to_eval_llh(
                candidate_tau=tau_i,
                candidate_lambda=np.nan,
                candidate_agent=candidate_agent,
                data=data)

            loglikelihood_function[i] = this_tau_s_llh

        return loglikelihood_function

    def eval_llh_function_tau_and_lambda(self, candidate_agent: str,
                                         data: pd.DataFrame):
        """Evaluate log_likelihood function for given 2-dimdensional tau and
        lambda space."""

        loglikelihood_function = np.full(
            (len(self.est_params.tau_bf_cand_space),
             len(self.est_params.lambda_bf_cand_space)),
            np.nan)

        for i_tau, tau_i in np.ndenumerate(
                self.est_params.tau_bf_cand_space):

            for i_lambda, lambda_i in np.ndenumerate(
                    self.est_params.lambda_bf_cand_space):

                this_theta_s_llh = self.sim_object.sim_to_eval_llh(
                    candidate_tau=tau_i,
                    candidate_lambda=lambda_i,
                    candidate_agent=candidate_agent,
                    data=data
                )

                loglikelihood_function[i_tau, i_lambda] = this_theta_s_llh

        return loglikelihood_function

    def eval_brute_force_est_tau(self, candidate_agent: str,
                                 data: pd.DataFrame):
        """Evaluate the maximum likelihood estimation of the decision noise
        parameter tau  based on dataset of one participant with brute force
        method.
        """

        llh_function = self.eval_llh_function_tau(
            candidate_agent=candidate_agent, data=data)

        # Identify tau with maximum likelihood
        ml_tau = self.est_params.tau_bf_cand_space[
            np.argmax(llh_function)]

        if ("agent" in data.columns and
                self.current_cand_agent == data.iloc[0]["agent"]):
            self.tau_est_result_gen_agent = ml_tau
            self.max_llh_current_gen_agent = np.max(llh_function)

        else:
            self.tau_est_result_current_cand_agent = ml_tau
            self.max_llh_current_cand_agent = np.max(llh_function)

    def eval_brute_force_tau_lambda(self, candidate_agent: str,
                                    data: pd.DataFrame):
        """Evaluate the maximum likelihood estimation of the decision noise
        parameter tau and weighting parameter lambda based on dataset of one
        participant with brute force method.
        """

        llh_function = self.eval_llh_function_tau_and_lambda(
            candidate_agent=candidate_agent, data=data)

        # Identify theta=(tau,lambda) with max likelihood
        ml_two_dim_index = np.unravel_index(llh_function.argmax(),
                                            llh_function.shape)
        ml_tau_index = ml_two_dim_index[0]
        ml_lambda_index = ml_two_dim_index[1]

        ml_tau = self.est_params.tau_bf_cand_space[ml_tau_index]
        ml_lambda = self.est_params.lambda_bf_cand_space[ml_lambda_index]

        if ("agent" in data.columns and
                self.current_cand_agent == data.iloc[0]["agent"]):
            self.tau_est_result_gen_agent = ml_tau
            self.lambda_est_result_gen_agent = ml_lambda
            self.max_llh_current_gen_agent = np.max(llh_function)
        else:
            self.tau_est_result_current_cand_agent = ml_tau
            self.lambda_est_result_current_cand_agent = ml_lambda
            self.max_llh_current_cand_agent = np.max(llh_function)

    def estimate_tau(self, method: str, candidate_agent: str,
                     data: pd.DataFrame):
        """Estimate tau parameter value with given dataset. Brute-force
        estimation method will first evaluate likelihood function for different
        candidate parameter values.

        Args:
            method (str): _description_
            candidate_agent (str): _description_
            data (pd.DataFrame): _description_
        """

        if method == "brute_force":
            self.eval_brute_force_est_tau(candidate_agent=candidate_agent,
                                          data=data)

    def estimate_tau_lambda(self, method: str, candidate_agent: str,
                            data: pd.DataFrame):
        """Estimate two-dimensional parameter vektor, tau and lambda"""

        if method == "brute_force":
            self.eval_brute_force_tau_lambda(candidate_agent=candidate_agent,
                                             data=data)

    def estimate_parameters(self, data: pd.DataFrame,
                            method: str, candidate_agent: str,
                            task_configs: TaskConfigurator,
                            bayesian_comps: BayesianModelComps):
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            method (str): _description_
            candidate_agent (str): _description_
            task_configs (TaskConfigurator): _description_
            bayesian_comps (BayesianModelComps): _description_
            sim_params (SimulationParameters): _description_
        """
        self.instantiate_sim_obj(
            task_configs=task_configs,
            bayesian_comps=bayesian_comps
            )

        # Set candidate agent model to generating agent
        self.current_cand_agent = candidate_agent

        self.reset_result_variables_to_nan()

        if "C" in candidate_agent:
            pass

        elif candidate_agent in ["A1", "A2"]:
            self.estimate_tau(method=method, candidate_agent=candidate_agent,
                              data=data)

        else:  # if candidate_agent == "A3"
            self.estimate_tau_lambda(method=method,
                                     candidate_agent=candidate_agent,
                                     data=data)

    def eval_llh_data(self, candidate_agent: str, method: str,
                      data: pd.DataFrame):
        """_summary_

        Args:
            candidate_agent (str): _description_
            method (str): _description_
            data (pd.DataFrame): _description_
        """
        if "C" in candidate_agent:
            self.eval_llh_data_no_params(
                candidate_agent=candidate_agent,
                data=data)

        elif candidate_agent in ["A1", "A2"]:
            self.estimate_tau(method=method,
                              candidate_agent=candidate_agent,
                              data=data)

        elif candidate_agent == "A3":
            self.estimate_tau_lambda(method=method,
                                     candidate_agent=candidate_agent,
                                     data=data)

    def determine_n_parameters(self, agent_model_name: str) -> int:
        """_summary_

        Args:
            agent_model_name (str): _description_

        Returns:
            int: _description_
        """
        if "C" in agent_model_name:
            n_params = 0
        elif agent_model_name == "A3":
            n_params = 2
        else:
            n_params = 1
        return n_params

    def eval_bic_giv_theta_hat(self,
                               llh_theta_hat: float,
                               n_params: int,
                               n_valid_actions: int):
        """_summary_

        Args:
            llh_theta_hat (float): _description_
            n_params (int): _description_
            n_valid_actions (int): _description_

        Returns:
            float: BIC value for model validation with given parameters values
        """
        this_bic = llh_theta_hat - n_params/2 * np.log(n_valid_actions)
        return this_bic

    def evaluate_bic_s(self, data: pd.DataFrame, est_method: str,
                       data_type: str) -> dict:
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            est_method (str): _description_
            data_type (str): _description_

        Returns:
            dict: _description_
        """
        agent_specific_bic_s = {
            "BIC_C1": np.nan, "BIC_C2": np.nan, "BIC_C3": np.nan,
            "BIC_A1": np.nan, "BIC_A2": np.nan, "BIC_A3": np.nan
        }

        for agent_model in self.est_params.agent_candidate_space:
            self.current_cand_agent = agent_model

            n_params = self.determine_n_parameters(agent_model)
            n_valid_choices = data.a.count()

            if data_type == "sim":
                if ("A" in agent_model and
                        agent_model == data.iloc[0]["agent"]):
                    llh_data = self.max_llh_current_gen_agent
                else:
                    self.eval_llh_data(candidate_agent=agent_model,
                                       method=est_method,
                                       data=data)
                    llh_data = self.max_llh_current_cand_agent

            else:  # if data_type == "exp"
                self.eval_llh_data(candidate_agent=agent_model,
                                   method=est_method,
                                   data=data)
                llh_data = self.max_llh_current_cand_agent

            agent_specific_bic_s[
                f"BIC_{agent_model}"] = self.eval_bic_giv_theta_hat(
                llh_theta_hat=llh_data,
                n_params=n_params,
                n_valid_actions=n_valid_choices)

        return agent_specific_bic_s
