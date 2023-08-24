"""This script contains classes and methods for model validation analyses."""

import time
import pandas as pd
from utilities.simulation_methods import Simulator, SimulationParameters
from utilities.estimation_methods import Estimator
from utilities.config import DirectoryManager, TaskConfigurator
from utilities.agent import BayesianModelComps


class Validator:
    """Class of methods to run model validation routines"""
    data_dic: dict
    estimator: Estimator = Estimator()

    def __init__(self, sim_params: SimulationParameters,
                 task_configs: TaskConfigurator,
                 bayesian_comps: BayesianModelComps,
                 dir_mgr: DirectoryManager):

        self.sim_params: SimulationParameters = sim_params
        self.task_config = task_configs
        self.bayesian_comps = bayesian_comps
        self.simulator: Simulator = Simulator(task_configs=task_configs,
                                              bayesian_comps=bayesian_comps)
        self.dir_mgr: DirectoryManager = dir_mgr

    def init_data_dic(self, validation_part: str):
        """_summary_
        """
        if validation_part == "model_recovery":
            self.data_dic = {
                "agent": [], "participant": [],
                "tau_gen": [], "tau_mle": [],
                "lambda_gen": [], "lambda_mle": []}
        elif validation_part == "model_comparison":
            self.data_dic = {
                "participant": [],
                }

        for agent in self.estimator.est_params.agent_candidate_space:
            self.data_dic[f"BIC_{agent}"] = []

    def record_data_generating_sim_params(self):
        """_summary_
        """
        self.data_dic["agent"].extend(
            [self.sim_params.current_agent_gen
             ] * self.sim_params.n_participants)
        self.data_dic["tau_gen"].extend(
            [self.sim_params.current_tau_gen
             ] * self.sim_params.n_participants)
        self.data_dic["lambda_gen"].extend(
            [self.sim_params.current_lambda_gen
             ] * self.sim_params.n_participants)

    def record_participant_number(self):
        """_summary_
        """
        self.data_dic["participant"].append(self.sim_params.current_part)

    def record_tau_estimate(self, tau_estimate: float):
        """_summary_

        Args:
            tau_estimate (float): _description_
        """
        self.data_dic["tau_mle"].append(tau_estimate)

    def record_lambda_estimate(self, lambda_estimate: float):
        """_summary_

        Args:
            lambda_estimate (float): _description_
        """
        self.data_dic["lambda_mle"].append(lambda_estimate)

    def record_bics(self, bics: dict):
        """_summary_

        Args:
            bics (dict): Dictioniary containing the BIC values for all
            candidate agent models
        """
        for agent in self.estimator.est_params.agent_candidate_space:
            self.data_dic[f"BIC_{agent}"].append(bics[f"BIC_{agent}"])

    def save_results(self, validation_part: str, sub_id: str):
        """Method to save validation results to a .tsv file

        Args:
            sub_id (str): Subject ID
        """
        filename = "not_defined"
        if validation_part == "model_recovery":
            self.dir_mgr.define_model_recov_results_filename(sub_id)
            filename = self.dir_mgr.paths.this_sub_model_recov_result_fn
        elif validation_part == "model_comparison":
            self.dir_mgr.define_model_comp_results_filename(
                sub_id=str(self.sim_params.current_part))
            filename = self.dir_mgr.paths.this_sub_model_comp_results_fn

        mle_df = pd.DataFrame(self.data_dic)
        with open(f"{filename}.tsv", "w", encoding="utf8") as tsv_file:
            tsv_file.write(mle_df.to_csv(sep="\t", na_rep="nan", index=False))

    def estimate_parameter_values(self, data: pd.DataFrame):
        """_summary_

        Args:
            data (pd.DataFrame): Data to be used to estimate parameters
        """
        self.estimator.estimate_parameters(
            data=data,
            method="brute_force",
            candidate_agent=self.sim_params.current_agent_gen,
            task_configs=self.simulator.task_configs,
            bayesian_comps=self.simulator.bayesian_comps)

        mle_tau_est = self.estimator.tau_est_result_gen_agent
        mle_lambda_est = self.estimator.lambda_est_result_gen_agent

        self.record_tau_estimate(mle_tau_est)
        self.record_lambda_estimate(mle_lambda_est)

    def evaluate_model_fitting_performance(self, data: pd.DataFrame):
        """_summary_
        """
        bics = self.estimator.evaluate_bic_s(est_method="brute_force",
                                             data=data,
                                             data_type="sim")
        self.record_bics(bics)

    def run_model_recovery(self, validation_part: str):
        """For each participant, simulate behavioral data, estimate parameter
        values and evaluate model recovery performance.

        Args:
            validation_part (str): "model_recovery" or "model_comparison"
            sub_id (str): _description_
        """

        self.init_data_dic(validation_part=validation_part)
        self.record_data_generating_sim_params()
        self.record_participant_number()

        simulated_data = self.simulator.simulate_beh_data(self.sim_params)

        start = time.time()
        self.estimate_parameter_values(data=simulated_data)
        end = time.time()
        print("time needed for ML parameter estimation with "
              f"{self.sim_params.current_agent_gen} as generating agent: "
              f"{round((end-start), ndigits=2)} sec.")

        start = time.time()
        self.evaluate_model_fitting_performance(data=simulated_data)
        end = time.time()
        print(
            "time needed for evaluatung mordel recovery performance for data",
            f" from {self.sim_params.current_agent_gen} as generating agent: ",
            f"{round((end-start), ndigits=2)} sec.")

        return pd.DataFrame(self.data_dic)

    def run_model_fitting_routine(self, data: pd.DataFrame):
        """For each participant, simulate behavioral data, estimate parameter
        values and evaluate model recovery performance"""

        self.init_data_dic(validation_part="model_comparison")
        self.record_participant_number()

        start = time.time()
        self.estimator.instantiate_sim_obj(
            task_configs=self.simulator.task_configs,
            bayesian_comps=self.simulator.bayesian_comps
        )
        self.evaluate_model_fitting_performance(data=data)
        end = time.time()
        print(
            "time needed for evaluatung mordel fitting performances for ",
            " experimental data from participant ",
            f" {self.sim_params.current_part} ",
            f"{round((end-start), ndigits=2)} sec.")

        self.save_results(validation_part="model_comparison",
                          sub_id=str(self.sim_params.current_part))
