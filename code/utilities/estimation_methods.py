"""This script contains classes and methods to evaluate Maximum Likelihood
estimations of model parameters
"""
import numpy as np
import pandas as pd
import xarray as xr
import utilities.abm_bmc as abm_bmc
from utilities.simulation_methods import Simulator
from utilities.agent import BayesianModelComps
from utilities.config import TaskConfigurator, custom_sort_key


class EstimationParameters:
    """Class to store and manage candidate model and parameter spaces for model
    recorvery

    Attributes:
    ----------
        agent_candidate_space (list): Candidate agent model space
        tau_bf_cand_space (list): Candidate tau parameter space for brute-force
            estimation method
        lambda_bf_cand_space (list): Candidate lambda parameter space for
            brute-force parameter estimation
    """
    agent_candidate_space: list
    tau_bf_cand_space: list
    lambda_bf_cand_space: list

    def get_params_from_args(self, args):

        """Method to fetch simulation parameters from command line or bash
        script arguments."""
        tau_cand_res = args.tau_cand_res
        lambda_cand_res = args.lambda_cand_res

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


class BMCObject:
    """A class to store Bayesian model comparison input data as needed
    for abm_bmc methods.

    TODO:
    -----
    !

    Attributes:
    -----------
        mll (np.ndarray): participants x models array of maximized log likelihoods
        n (np.ndarray): participants x models array of observation numbers
        k  (np.ndarray): participants x models array of free parameter numbers

        mll_avg (np.ndarray): 1 x models array of maximized log likelihood averages
        bic (np.ndarray): participants x models array of BICs
        bic_sum (np.ndarray): 1 x models array of BIC sum values
        win (np.ndarray): p x 1 array indexing participant-specific winning models
        gmi (Bunch): group model inference structure
        phi (np.ndarray): 1 x models array of protected exceedance probabilities
        p_eta_0 (np.ndarray): posterior probability of null hypothesis ("Bayes omnibus risk")
        """

    def __init__(self, mll, bic, n: np.ndarray, k: np.ndarray):  # TODO: what type is xarray?
        self.mll = mll
        self.bic = bic
        self.n = n
        self.k = k
        # self.phi = np.full(n_models, np.nan)
        # self.mll_avg = np.full(n_models, np.nan)
        # self.bic = np.full(n_models, np.nan)
        # self.bic_sum = np.full(n_models, np.nan)
        # self.win = np.full(1, np.nan)  # TODO: 1 because n_part = 1
        # self.gmi = None  # TODO: später Bunch()
        # self.p_eta_0 = np.full(1, np.nan)


class Estimator:
    """A class to evaluate Maximum Likelihood parameters estimations

    Attributes:
    ---------
        mll (np.ndarray): (1 x n_agents)-array of maximum log likelihoods. 
        
    Args:
    -----
        estim_params (EstimationParameters): _description_
    """
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
        self.mll_results = np.full((1,
                                    len(estim_params.agent_candidate_space)),
                                    np.nan)

    def instantiate_sim_obj(self, task_configs: TaskConfigurator,
                            bayesian_comps: BayesianModelComps):
        """
        Method to instantiate simulation object

        Args:
        ----------
        sim_object (Simulator): Object to simulate agent-task-interaction
        """
        self.sim_object = Simulator(task_configs=task_configs,
                                    bayesian_comps=bayesian_comps)

    def reset_result_variables(self):
        """Method to reset result variables to nan values"""
        self.tau_est_result_gen_agent = np.nan
        self.tau_est_result_current_cand_agent = np.nan
        self.lambda_est_result_gen_agent = np.nan
        self.lambda_est_result_current_cand_agent = np.nan
        self.max_llh_current_gen_agent = np.nan
        self.max_llh_current_cand_agent = np.nan

    def eval_llh_data_no_params(self, candidate_agent: str,
                                data: pd.DataFrame):
        """Method to evaluate the "likelihood" of data. For control models
        without parameters

        Args:
        -----
            candidate_agent (str): Candidate agent name
            data (pd.DataFrame): (n_events x n_meausures)-dataframe of
                behavioral data used for llh evaluation
        """
        llh = self.sim_object.sim_to_eval_llh(
            candidate_tau=np.nan,
            candidate_lambda=np.nan,
            candidate_agent=candidate_agent,
            data=data
        )
        self.max_llh_current_cand_agent = llh

    def eval_llh_function_tau(self, candidate_agent: str, data: pd.DataFrame):
        """Method to evaluate log likelihood function for given tau parameter
        space, if lambda is not applicable.

        Args:
        ----
            candidate_agent (str): Current candidate agent
            data (pd.DataFrame): (n_events x n_meausures)-dataframe of
                behavioral data used for llh evaluation

        Returns:
        --------
            np.ndarray: (1 x n_tau_candidates)-array of tau-specific
                log likelihood values
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
        """Method to evaluate the log likelihood function for given
        2-dimdensional parameter (i.e. tau and lambda) space.

        Args:
        -----
            candidate_agent (str): Current candidate agent
            data (pd.DataFrame): (n_events x n_meausures)-dataframe of
                behavioral data used for llh evaluation

        Returns:
        -------
            np.ndarray: (n_tau_candidates x n_lambda_candidates)-array of
            tau- and lambda-specific log likelihood values
        """
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
        """Method to evaluate the maximum likelihood estimation of the decision
        noise parameter tau based on dataset of one participant with
        brute-force method. No return, ML estimate is recorded to self.

        Args:
        -----
            candidate_agent (str): Current candidate agent
            data (pd.DataFrame): (n_events x n_meausures)-dataframe of
                behavioral data
        """
        # Evaluate llh function over candidate tau values
        llh_function = self.eval_llh_function_tau(
            candidate_agent=candidate_agent, data=data)

        # Identify tau with maximum likelihood
        ml_tau = self.est_params.tau_bf_cand_space[
            np.argmax(llh_function)]

        # Record ml estimate
        if ("agent" in data.columns and
                self.current_cand_agent == data.iloc[0]["agent"]):
            self.tau_est_result_gen_agent = ml_tau
            self.max_llh_current_gen_agent = np.max(llh_function)
        else:
            self.tau_est_result_current_cand_agent = ml_tau
            self.max_llh_current_cand_agent = np.max(llh_function)

    def eval_brute_force_tau_lambda(self, candidate_agent: str,
                                    data: pd.DataFrame):
        """Method to evaluate the maximum likelihood estimation of the
        2-dimensional parameter space theta=(tau, lambda) based on dataset of
        one participant with brute force method.

        tau is the decision noise parameter and lambda the weighting parameter.
    
        No return, ML estimate is
        recorded to self.

        Args:
        -----
            candidate_agent (str): Current candidate agent
            data (pd.DataFrame): (n_events x n_meausures)-dataframe of
                behavioral data
        """

        # Evaluate multivariate llh function over cand tau and lambda spaces
        llh_function = self.eval_llh_function_tau_and_lambda(
            candidate_agent=candidate_agent, data=data)

        # Identify theta=(tau,lambda) with max likelihood
        ml_two_dim_index = np.unravel_index(llh_function.argmax(),
                                            llh_function.shape)
        ml_tau_index = ml_two_dim_index[0]
        ml_lambda_index = ml_two_dim_index[1]

        ml_tau = self.est_params.tau_bf_cand_space[ml_tau_index]
        ml_lambda = self.est_params.lambda_bf_cand_space[ml_lambda_index]

        # Record ML estimates
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
        """Method to start parameter estimation using a method of choice.

        Note:
        -----
            So far, only brute-force implemented

        Args:
        -----
            method (str): If "brute-force"
            candidate_agent (str): Current candidate agent
                data (pd.DataFrame): (n_events x n_meausures)-dataframe of
                behavioral data
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

        self.reset_result_variables()

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

    def evaluate_pep_s(self, val_results_df: pd.DataFrame,
                       data_type: str):
        
        peps_df = pd.DataFrame()

        mll_col_names = [col_name for col_name in val_results_df.columns
                         if "MLL" in col_name]
        bic_col_names = [col_name for col_name in val_results_df.columns
                         if "BIC" in col_name]
        
        analizing_agents = sorted(list(set(
            [mll_col_name[-2:] for mll_col_name in mll_col_names
             ])), key=custom_sort_key)

        col_names_for_mll_array = mll_col_names
        col_names_for_mll_array.append("participant")

        col_names_for_bic_array = bic_col_names
        col_names_for_bic_array.append("participant")

        n_models = len(analizing_agents)

        if data_type == "sim":
            for agent_gen in val_results_df.agent.unique():

                recording_dic = {"agent": agent_gen}
                this_agents_df = val_results_df[val_results_df.agent == agent_gen]

                for tau_gen in this_agents_df.tau_gen.unique():
                    recording_dic["tau_gen"] =tau_gen
            
                    if "A" in agent_gen:
                        this_taus_df = this_agents_df[this_agents_df.tau_gen == tau_gen]
                    else:  # if "C" in agent
                        this_taus_df = this_agents_df

                    for lambda_gen in this_taus_df.lambda_gen.unique():
                        recording_dic["lambda_gen"] = lambda_gen
                        if agent_gen == "A3":
                            this_lambdas_df = this_taus_df[
                                this_taus_df.lambda_gen == lambda_gen]

                        else:  # if "C" in agent or "A1" or "A2"
                            this_lambdas_df = this_taus_df

                        mll_df = this_lambdas_df[
                            col_names_for_mll_array].set_index("participant")
                        mll_df.rename(columns=lambda x: x[-2:], inplace=True)  # Rename columns to hold agent name only
                        mll_array = xr.DataArray(
                            mll_df, dims=("participant", "analizing_agent"))
                        
                        bic_df = this_lambdas_df[
                            col_names_for_bic_array].set_index("participant")
                        bic_df.rename(columns=lambda x: x[-2:], inplace=True)  # Rename columns to hold agent name only
                        bic_array = xr.DataArray(
                            mll_df, dims=("participant", "analizing_agent"))

                        bmc_object = BMCObject(mll=np.array(mll_array),
                                            bic=np.array(bic_array),
                                            n=np.full((1, n_models),
                                                        320),  # dummy n TODO: save during mll estimation
                                                        k=np.array([[0, 0, 0, 1, 1, 2]])
                                            )

                        abm_bmc.abm_bmc(bmc=bmc_object)

                        for i, analizing_agent in enumerate(analizing_agents):
                            recording_dic[
                                f"PEP_{analizing_agent}"] = [bmc_object.phi[i]]

                        this_gen_params_and_models_peps = pd.DataFrame(recording_dic)  # TODO: hier weiter

                        peps_df = pd.concat(
                            [peps_df, this_gen_params_and_models_peps],
                            ignore_index=True)

        else:  # if data_type == "exp"
            recording_dic = {}
            mll_df = val_results_df[
                col_names_for_mll_array].set_index("participant")
            mll_df.rename(columns=lambda x: x[-2:], inplace=True)  # Rename columns to hold agent name only
            mll_array = xr.DataArray(
                mll_df, dims=("participant", "analizing_agent"))
            
            bic_df = val_results_df[
                col_names_for_bic_array].set_index("participant")
            bic_df.rename(columns=lambda x: x[-2:], inplace=True)  # Rename columns to hold agent name only
            bic_array = xr.DataArray(
                mll_df, dims=("participant", "analizing_agent"))

            bmc_object = BMCObject(
                mll=np.array(mll_array),
                bic=np.array(bic_array),
                n=np.full((1, n_models), 320),  # dummy n TODO: save during mll estimation
                k=np.array([[0, 0, 0, 1, 1, 2]])
                )

            abm_bmc.abm_bmc(bmc=bmc_object)

            for i, analizing_agent in enumerate(analizing_agents):
                recording_dic[f"PEP_{analizing_agent}"] = [
                    bmc_object.phi[i]]

            this_gen_params_and_models_peps = pd.DataFrame(recording_dic)

            peps_df = pd.concat(
                [peps_df, this_gen_params_and_models_peps],
                ignore_index=True)

        return peps_df

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

        for i, agent_model in enumerate(self.est_params.agent_candidate_space):
            self.current_cand_agent = agent_model

            n_params = self.determine_n_parameters(agent_model)
            n_valid_choices = data.a.count()

            if data_type == "sim":
                if ("A" in agent_model and
                        agent_model == data.iloc[0]["agent"]):
                    max_llh_data = self.max_llh_current_gen_agent
                else:
                    self.eval_llh_data(candidate_agent=agent_model,
                                       method=est_method,
                                       data=data)
                    max_llh_data = self.max_llh_current_cand_agent

            else:  # if data_type == "exp"
                self.eval_llh_data(candidate_agent=agent_model,
                                   method=est_method,
                                   data=data)
                max_llh_data = self.max_llh_current_cand_agent

            # Record agent specific maximized loglikelihood
            self.mll_results[0, i] = max_llh_data

            # Record agent specific BIC
            agent_specific_bic_s[
                f"BIC_{agent_model}"] = self.eval_bic_giv_theta_hat(
                llh_theta_hat=max_llh_data,
                n_params=n_params,
                n_valid_actions=n_valid_choices)

        return agent_specific_bic_s
