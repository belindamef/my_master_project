"""
This script contains classes and methods for model validation analyses.

Author: Belinda Fleischmann
"""

from utilities.simulation_methods import Simulator, SimulationParameters
from utilities.estimation_methods import ParamAndModelRecoverer
from utilities.modelling import AgentInitObject
from utilities.config import DirectoryManager
import numpy as np
import pandas as pd


class Validator:
    data_dic: dict = {
        "agent": [], "participant": [],
        "tau_gen": [], "tau_mle": [],
        "lambda_gen": [], "lambda_mle": []}
    recoverer: ParamAndModelRecoverer = ParamAndModelRecoverer()

    def __init__(self, sim_params: SimulationParameters,
                 simulator: Simulator, dir_mgr: DirectoryManager):
        self.sim_params: SimulationParameters = sim_params
        self.simulator: Simulator = simulator
        self.dir_mgr: DirectoryManager = dir_mgr

    def record_data_generating_sim_params(self):
        # TODO: more elegant solution please...
        self.data_dic["agent"].extend(
            [self.simulator.sim_params.current_agent_attributes.name
             ] * self.simulator.sim_params.n_participants)
        self.data_dic["tau_gen"].extend(
            [self.simulator.sim_params.current_tau_gen
             ] * self.simulator.sim_params.n_participants)
        self.data_dic["lambda_gen"].extend(
            [self.simulator.sim_params.current_lambda_gen
             ] * self.simulator.sim_params.n_participants)

    def record_participant_number(self):
        self.data_dic["participant"].append(self.sim_params.current_part)

    def record_tau_estimate(self, tau_estimate):
        self.data_dic["tau_mle"].append(tau_estimate)

    def record_lambda_estimate(self, lambda_estimate):
        self.data_dic["lambda_mle"].append(lambda_estimate)

    def save_param_est_results(self):
        self.dir_mgr.create_agent_sub_id(self.sim_params)
        self.dir_mgr.define_val_results_filename()

        mle_df = pd.DataFrame(self.data_dic)

        with open(f"{self.dir_mgr.paths.this_sub_val_results_filename}.tsv",
                  "w", encoding="utf8") as tsv_file:
            tsv_file.write(mle_df.to_csv(sep="\t", na_rep=np.NaN, index=False))

    def estimate_parameter_values(self):
        self.recoverer.instantiate_sim_obj(
            exp_data=self.simulator.data,
            task_configs=self.simulator.task_configs,
            bayesian_comps=self.simulator.bayesian_comps
            )

        # Embed simulation params in estimator sim object
        self.recoverer.sim_object.sim_params = self.sim_params

        if (np.isnan(self.sim_params.current_tau_gen)
                and np.isnan(self.sim_params.current_lambda_gen)):
            mle_tau_est = np.nan
            mle_lambda_est = np.nan

        elif np.isnan(self.sim_params.current_lambda_gen):
            mle_tau_est = self.recoverer.estimate_tau(
                method="brute_force")
            mle_lambda_est = np.nan

        else:         # TODO: hier weiter! wie 2-dimensional schätzen??!!
            mle_tau_est, mle_lambda_est = self.recoverer.estimate_tau_lambda(
                    method="brute_force")

        self.record_tau_estimate(mle_tau_est)
        self.record_lambda_estimate(mle_lambda_est)
        self.save_param_est_results()

    def evaluate_model_recovery_performance(self):
        self.recoverer.evaluate_BICs()

    def iterate_participants(self):
        """For each participant, simulate behavioral data, estimate parameter
        values and evaluate model recovery performance"""
        for participant in self.sim_params.participant_numbers:
            self.sim_params.current_part = participant + 1
            self.record_participant_number()

            self.simulator.simulate_beh_data()

            self.estimate_parameter_values()

            self.evaluate_model_recovery_performance()

    def iterate_parameter_space(self):
        for tau_gen in self.simulator.sim_params.tau_space_gen:
            self.simulator.sim_params.current_tau_gen = tau_gen

            for lambda_gen in self.simulator.sim_params.lambda_gen_space:
                self.simulator.sim_params.current_lambda_gen = lambda_gen

                self.record_data_generating_sim_params()

                self.iterate_participants()

    def iterate_reps_and_generating_beh_models(self):
        for repetition in self.sim_params.repetition_numbers:
            self.sim_params.current_rep = repetition + 1

            for agent_model in self.simulator.sim_params.agent_space_gen:
                self.sim_params.current_agent_attributes = AgentInitObject(
                    agent_model)

                self.sim_params.current_agent_model = agent_model

                self.iterate_parameter_space()

    def run_simulation_and_validation_routine(self):
        self.iterate_reps_and_generating_beh_models()
