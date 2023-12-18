"""This script contains classes and methods for model validation analyses."""

import time
import pandas as pd
import numpy as np
from utilities.simulation_methods import Simulator, SimulationParameters
from utilities.estimation_methods import Estimator, EstimationParameters
from utilities.config import humanreadable_time
from utilities.task import TaskConfigurator, GridConfigParameters
from utilities.agent import StochasticMatrices


class ValidationParameters:
    """Class to store and manage parameters for model validation 

    Attributes:
    ----------
        n_reps (int): Number of repetition
        repetition_numbers (range): Repitions numbers, iterable
        n_participants (int): number of participants
        participant_numbers (range): Pariticipant numbers, iterable
        current_rep (int): Current repetition number
        current_part (int): Current participant number
    """

    n_reps: int
    repetition_numbers: range
    n_participants: int
    participant_numbers: range
    current_rep: int
    current_part: int

    def get_params_from_args(self, args):
        """Method to fetch simulation parameters from arguments passed 
        from command line or shell script arguments.

        Args:
        -----
            args (TODO): TODO
            """
        self.repetition_numbers = args.repetition
        self.participant_numbers = args.participant
        self.n_participants = len(self.participant_numbers)
        self.n_reps = len(self.repetition_numbers)
        return self

    def define_numbers(self, n_rep: int = 1, n_part: int = 1,):
        """Method to define number of repetitions and participants to
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
    """Class of methods to run model validation

    Attributes:
    -----------
        data_dict (dict): dictionary to record validation results
        sim_params (SimulationParameters): Data generating
            model and parameter values. e.g. agent model, tau value etc..
        val_params (ValidationParameters): Validation
            parameters, e.g. number of repetitions
        task_configs (TaskConfigurator): Current task configuration
        bayesian_comps (BayesianModelComps): Bayesian model components, e.g.
            likelihood, prior etc.
        est_params (EstimationParameters): Candidate model and parameter
            spaces and current values for mll estimation.
        simulator (Simulator): Object to simulate trial interactions for
            mll estimations
        estimator (Estimator): Object to perform model and parameter
            estimations.

    Args:
    -----
        sim_params (SimulationParameters): Data generating
            model and parameter values. e.g. agent model, tau value etc..
        val_params (ValidationParameters): Validation
            parameters, e.g. number of repetitions
        task_configs (TaskConfigurator): Current task configuration
        bayesian_comps (BayesianModelComps): Bayesian model components, e.g.
            likelihood, prior etc.
        est_params (EstimationParameters): Candidate model and parameter
            spaces and current values for mll estimation.
    """
    data_dic: dict

    def __init__(self, sim_params: SimulationParameters,
                 val_params: ValidationParameters,
                 task_configs: TaskConfigurator,
                 task_params: GridConfigParameters,
                 bayesian_comps: StochasticMatrices,
                 est_params: EstimationParameters):

        self.val_params = val_params
        self.sim_params: SimulationParameters = sim_params
        self.task_config = task_configs
        self.task_params = task_params
        self.bayesian_comps = bayesian_comps

        self.simulator: Simulator = Simulator(task_configs=task_configs,
                                              bayesian_comps=bayesian_comps)
                                              # tODO: add task_design_paramsa s argument
        self.estimator: Estimator = Estimator(estim_params=est_params)

    def init_data_dic(self, validation_type: str):
        """Method to initial recording dictionary for validation results.

        Args:
        -----
            validation_type (str): Type of validation routine. "model_recovery"
                for validation with simulated datasets. "model_estimation" for
                    model validation with experimental datasets.
        """
        if validation_type == "model_recovery":
            self.data_dic = {
                "agent": [], "participant": [],
                "tau_gen": [], "tau_mle": [],
                "lambda_gen": [], "lambda_mle": []}
        elif validation_type == "model_estimation":
            self.data_dic = {
                "participant": [],
                }

        self.data_dic["n_valid_actions"] = []

        for agent in self.estimator.est_params.agent_candidate_space:
            self.data_dic[f"BIC_{agent}"] = []
            self.data_dic[f"MLL_{agent}"] = []

    def record_sim_params(self):
        """Method to record data generating model and parameter values
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
        """Method to record participant number"""
        self.data_dic["participant"].append(self.val_params.current_part)

    def record_tau_estimate(self, tau_estimate: float):
        """Method to record tau estimate

        Args:
        ----
            tau_estimate (float): tau estimate value
        """
        self.data_dic["tau_mle"].append(tau_estimate)

    def record_lambda_estimate(self, lambda_estimate: float):
        """Method to record tau estimate

        Args:
            lambda_estimate (float): lambda estimate value
        """
        self.data_dic["lambda_mle"].append(lambda_estimate)

    def record_bics(self, bics: dict):
        """Method to record analzing agent specific BIC values

        Args:
        ----
            bics (dict of str: float): Dictioniary containing the BIC values
                for all candidate agent models
        """
        for agent in self.estimator.est_params.agent_candidate_space:
            self.data_dic[f"BIC_{agent}"].append(bics[f"BIC_{agent}"])

    def record_mlls(self, mlls: np.ndarray):
        """Method to record MLL values for all candidate agent models.

        Args:
        ----
            bics (dict of str: float): Dictioniary containing the BIC values
                for all candidate agent models
        """
        for i, agent in enumerate(
                self.estimator.est_params.agent_candidate_space):
            self.data_dic[f"MLL_{agent}"].append(mlls[0, i])

    def record_n_valid_actions(self, n_valid_actions: int):
        """Method to record number of valid actions

        Args:
        -----
            n_valid_actions (int): Number of valid actions
        """
        self.data_dic["n_valid_actions"].append(n_valid_actions)

    def estimate_parameter_values(self, data: pd.DataFrame):
        """Method to estimate and record parameter estimates for given dataset

        Args:
        ----
            data (pd.DataFrame): (n_events x n_meausures)-dataframe of
                behavioral data
        """
        self.estimator.estimate_parameters(
            task_params=self.task_params,
            data=data,
            method="brute_force",
            candidate_agent=self.sim_params.current_agent_gen,
            task_configs=self.simulator.task_configs,
            bayesian_comps=self.simulator.bayesian_comps)

        # TODO: simplefy mle recording...

        mle_tau_est = self.estimator.tau_est_result_gen_agent
        mle_lambda_est = self.estimator.lambda_est_result_gen_agent

        self.record_tau_estimate(mle_tau_est)
        self.record_lambda_estimate(mle_lambda_est)

    def evaluate_bics(self, data: pd.DataFrame, datatype: str):
        """Method to let estimator object evaluate BICs (and mll estimates
        as side produkt...) and record values. 

        Args:
        ----
            data (pd.DataFrame): (n_events x n_meausures)-dataframe of
                behavioral data
            datatype (str): "sim" for simulated dataset, "exp" for experimental
                dataset
        """
        # Let estimator run estimations
        bics = self.estimator.evaluate_bic_s(method="brute_force",
                                             data=data,
                                             data_type=datatype)

        # Record results
        self.record_bics(bics)
        self.record_mlls(self.estimator.mll_results)
        n_valid_action_choices = data.a.count()
        self.record_n_valid_actions(n_valid_action_choices)

    def evaluate_peps(self, val_results: pd.DataFrame,
                      data_type: str):
        """Method to evaluate peps

        Args:
            val_results (pd.DataFrame): model recovery results including column
                with mll values.
            datatype (str): "sim" for simulated dataset, "exp" for experimental
                dataset
        """
        return self.estimator.evaluate_pep_s(val_results_df=val_results,
                                             data_type=data_type)

    def run_model_recovery(self) -> pd.DataFrame:
        """Main method to run model recovery routine. For one agent
        participant, this method simulates behavioral data (1), and runs
        parameter (2) and model (3) recovery analyses.

        Returns:
        -------
            pd.DataFrame: Model recovery results
        """

        # Initialize result recording
        self.init_data_dic(validation_type="model_recovery")
        self.record_sim_params()
        self.record_participant_number()

        # ------ (1) Simulate behavioral data ---------------------------------
        simulated_data = self.simulator.simulate_beh_data(self.sim_params)

        # ------ (2) Estimate parameter value(s) ------------------------------
        print(f"Running ML parameter estimation with data from "
              f"{self.sim_params.current_agent_gen} ...")
        start = time.time()
        self.estimate_parameter_values(data=simulated_data)
        end = time.time()
        print(" ... finished ML parameter estimation "
              f"\n     \n ... time:  {humanreadable_time(end-start)}\n")

        # ------(3) Start model recovery --------------------------------------
        print("Running model estimation with simulated data from",
              f" {self.sim_params.current_agent_gen} ...")
        start = time.time()
        self.evaluate_bics(data=simulated_data,
                           datatype="sim")
        end = time.time()
        print(" ... finished model estimationting ",
              f"\n     \n ... time:  {humanreadable_time(end-start)}\n")

        return pd.DataFrame(self.data_dic)

    def run_model_estimation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Method to run model estimation for one participant's
        behavioral data, this method evaluates model validation performances
        (i.e. MLL and BIC) for each candidate model.

        Args:
        -----
            data (pd.DataFrame): (n_events x n_meausures)-dataframe of
                behavioral data

        Returns:
        -------
            pd.DataFrame: Model estimation results"""

        # Initialize result recording
        self.init_data_dic(validation_type="model_estimation")
        self.record_participant_number()

        # Instantiate simulation-obj within estimator-obj for mll estimations
        self.estimator.instantiate_sim_obj(
            task_configs=self.simulator.task_configs,
            bayesian_comps=self.simulator.bayesian_comps,
            task_params=self.simulator.task_params
            )

        # Start estimations
        print("Running model estimation with experimental data ",
              f"from participant {self.val_params.current_part} ...")
        start = time.time()
        self.evaluate_bics(data=data, datatype="exp")
        end = time.time()
        print("finished model validaton with experimental data of participant",
              f" {self.val_params.current_part}, ",
              f"repitition no. {self.val_params}",
              f"\n     \n ... time:  {humanreadable_time(end-start)}\n")

        return pd.DataFrame(self.data_dic)
