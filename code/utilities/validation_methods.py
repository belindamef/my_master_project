"""This script contains classes and methods for model validation analyses."""

import time
import pandas as pd
import numpy as np
import xarray as xr
from utilities.simulation_methods import Simulator, SimulationParameters
from utilities.estimation_methods import Estimator, EstimationParameters
from utilities.config import TaskConfigurator, humanreadable_time
from utilities.agent import BayesianModelComps


class ValidationParameters:
    """Class to store and manage parameters model validation routines
    """

    n_reps: int
    repetition_numbers: range
    n_participants: int
    participant_numbers: range
    current_rep: int
    current_part: int

    def get_params_from_args(self, args):
        """Method to fetch simulation parameters from command line or bash
        script arguments."""
        self.repetition_numbers = args.repetition
        self.participant_numbers = args.participant
        self.n_participants = len(self.participant_numbers)
        self.n_reps = len(self.repetition_numbers)
        return self

    def define_numbers(self, n_rep: int = 1, n_part: int = 1,):
        """Method to define pass number of repetitions and participants to
        class instance.

        Parameters
        ----------
        n_rep : int
            Number of repetitions. Default value is 1
        n_part : int
            Number of participants. Default value is 1
            """
        self.n_reps = n_rep
        self.repetition_numbers = range(self.n_reps)
        self.n_participants = n_part
        self.participant_numbers = range(self.n_participants)


class Validator:
    """Class of methods to run model validation routines
    
    Attributes:
        data_dict (dict): dictionary that stores validation results"""
    data_dic: dict

    def __init__(self, sim_params: SimulationParameters,
                 val_params: ValidationParameters,
                 task_configs: TaskConfigurator,
                 bayesian_comps: BayesianModelComps,
                 est_params: EstimationParameters):

        self.val_params = val_params
        self.sim_params: SimulationParameters = sim_params
        self.task_config = task_configs
        self.bayesian_comps = bayesian_comps
        self.simulator: Simulator = Simulator(task_configs=task_configs,
                                              bayesian_comps=bayesian_comps)
        self.estimator: Estimator = Estimator(estim_params=est_params)

    def init_data_dic(self, validation_part: str):
        """_summary_
        """
        if validation_part == "model_recovery":
            self.data_dic = {
                "agent": [], "participant": [],
                "tau_gen": [], "tau_mle": [],
                "lambda_gen": [], "lambda_mle": []}
        elif validation_part == "model_estimation":
            self.data_dic = {
                "participant": [],
                }

        for agent in self.estimator.est_params.agent_candidate_space:
            self.data_dic[f"BIC_{agent}"] = []
            self.data_dic[f"PEP_{agent}"] = []
            self.data_dic[f"MLL_{agent}"] = []

    def record_data_generating_sim_params(self):
        """_summary_
        """
        self.data_dic["agent"].extend(
            [self.sim_params.current_agent_gen
             ] * self.val_params.n_participants)
        self.data_dic["tau_gen"].extend(
            [self.sim_params.current_tau_gen
             ] * self.val_params.n_participants)
        self.data_dic["lambda_gen"].extend(
            [self.sim_params.current_lambda_gen
             ] * self.val_params.n_participants)

    def record_participant_number(self):
        """_summary_
        """
        self.data_dic["participant"].append(self.val_params.current_part)

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

    def record_peps(self, peps: dict):
        """_summary_

        Args:
            bics (dict): Dictioniary containing the BIC values for all
            candidate agent models
        """
        for agent in self.estimator.est_params.agent_candidate_space:
            self.data_dic[f"PEP_{agent}"].append(peps[f"PEP_{agent}"])

    def record_mlls(self, mlls: np.ndarray):
        """_summary_

        Args:
            bics (dict): Dictioniary containing the BIC values for all
            candidate agent models
        """
        for i, agent in enumerate(
                self.estimator.est_params.agent_candidate_space):
            self.data_dic[f"MLL_{agent}"].append(mlls[0, i])

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

    def evaluate_model_fitting_performance(self, data: pd.DataFrame,
                                           datatype: str):
        """_summary_
        """
        bics = self.estimator.evaluate_bic_s(est_method="brute_force",
                                             data=data,
                                             data_type=datatype)
        self.record_bics(bics)

        peps = self.estimator.evaluate_pep_s(data=data)

        self.record_peps(peps)

        self.record_mlls(self.estimator.mll)

    def run_model_recovery(self):
        """For each participant, simulate behavioral data, estimate parameter
        values and evaluate model recovery performance.

        Args:
            validation_part (str): "model_recovery" or "model_estimation"
            sub_id (str): _description_
        """

        self.init_data_dic(validation_part="model_recovery")
        self.record_data_generating_sim_params()
        self.record_participant_number()

        simulated_data = self.simulator.simulate_beh_data(self.sim_params)

        print(f"Running ML parameter estimation with data from "
              f"{self.sim_params.current_agent_gen} ...")
        start = time.time()
        self.estimate_parameter_values(data=simulated_data)
        end = time.time()
        print(" ... finished ML parameter estimation "
              f"\n     time needed: {humanreadable_time(end-start)}")

        print("Running model estimation with simulated data from",
              f" {self.sim_params.current_agent_gen} ...")
        start = time.time()
        self.evaluate_model_fitting_performance(data=simulated_data,
                                                datatype="sim")
        end = time.time()
        print(" ... finished model estimationting ",
              f"\n     time needed: {humanreadable_time(end-start)}")
        return pd.DataFrame(self.data_dic)

    def run_model_estimation(self, data: pd.DataFrame) -> pd.DataFrame:
        """For each participant's behavioral data, estimate parameter
        values and evaluate model recovery performance"""

        self.init_data_dic(validation_part="model_estimation")
        self.record_participant_number()

        self.estimator.instantiate_sim_obj(
            task_configs=self.simulator.task_configs,
            bayesian_comps=self.simulator.bayesian_comps
        )
        print("Running model estimation with experimental data from ",
              f" {self.val_params.current_part} ...")
        start = time.time()
        self.evaluate_model_fitting_performance(data=data, datatype="exp")
        end = time.time()
        print("finished model fitting with experimental data from participant",
              f" {self.val_params.current_part}, ",
              f"repitition no. {self.val_params}",
              f"\n     time needed: {humanreadable_time(end-start)}")
        return pd.DataFrame(self.data_dic)
