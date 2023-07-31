"""
This script contains classes and methods for model validation analyses.

Author: Belinda Fleischmann
"""

from utilities.simulation_methods import Simulator, SimulationParameters
from utilities.estimation_methods import ParamAndModelRecoverer
from utilities.config import DirectoryManager
import numpy as np
import pandas as pd


class Validator:
    data_dic: dict
    recoverer: ParamAndModelRecoverer = ParamAndModelRecoverer()

    def __init__(self, sim_params: SimulationParameters,
                 simulator: Simulator, dir_mgr: DirectoryManager):
        self.sim_params: SimulationParameters = sim_params
        self.simulator: Simulator = simulator
        self.dir_mgr: DirectoryManager = dir_mgr

    def init_data_dic(self):
        self.data_dic = {
            "agent": [], "participant": [],
            "tau_gen": [], "tau_mle": [],
            "lambda_gen": [], "lambda_mle": []}

        for agent in self.recoverer.recov_params.agent_candidate_space:
            self.data_dic[f"BIC_{agent}"] = []

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

    def record_bics(self, bics: dict):
        for agent in self.recoverer.recov_params.agent_candidate_space:
            self.data_dic[f"BIC_{agent}"].append(bics[f"BIC_{agent}"])

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

        # Set candidate agent model to generating agent
        self.recoverer.current_cand_agent = self.sim_params.current_agent_gen

        self.recoverer.reset_result_variables_to_nan()  # TODO hübscher

        self.recoverer.estimate_parameters(method="brute_force")

        mle_tau_est = self.recoverer.tau_est_result_gen_agent
        mle_lambda_est = self.recoverer.lambda_est_result_gen_agent

        self.record_tau_estimate(mle_tau_est)
        self.record_lambda_estimate(mle_lambda_est)

    def evaluate_model_recovery_performance(self):
        bics = self.recoverer.evaluate_bic_s(est_method="brute_force")
        self.record_bics(bics)

    def iterate_participants(self):
        """For each participant, simulate behavioral data, estimate parameter
        values and evaluate model recovery performance"""
        for participant in self.sim_params.participant_numbers:
            self.sim_params.current_part = participant + 1

            self.init_data_dic()
            self.record_data_generating_sim_params()
            self.record_participant_number()

            self.simulator.simulate_beh_data()

            self.estimate_parameter_values()

            self.evaluate_model_recovery_performance()

            self.save_param_est_results()
