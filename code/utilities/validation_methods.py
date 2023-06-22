"""
This script contains classes and methods for model validation analyses.

Author: Belinda Fleischmann
"""

from utilities.simulation_methods import Simulator, SimulationParameters
from utilities.estimation_methods import ParameterEstimator
from utilities.modelling import AgentInitObject
from utilities.config import DirectoryManager
import numpy as np
import pandas as pd


class Validator:
    data_dic: dict = {
        "agent": [], "participant": [],
        "tau_gen": [], "tau_mle": [],
        "lambda_gen": [], "lambda_mle": []}

    def __init__(self, sim_params: SimulationParameters,
                 simulator: Simulator, dir_mgr: DirectoryManager):
        self.sim_params = sim_params
        self.simulator = simulator
        self.dir_mgr = dir_mgr

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

    def save_data(self):
        out_fn = self.dir_mgr.define_out_single_val_filename(
            self.sim_params.current_rep,
            self.sim_params.current_agent_model,
            self.sim_params.current_tau_gen,
            self.sim_params.current_lambda_gen,
            self.sim_params.current_part)

        mle_df = pd.DataFrame(self.data_dic)

        with open(f"{out_fn}.tsv", "w",
                    encoding="utf8") as tsv_file:
            tsv_file.write(mle_df.to_csv(
                sep="\t", na_rep=np.NaN, index=False))


    def iterate_participants(self):
        for participant in self.sim_params.participant_numbers:
            self.sim_params.current_part = participant + 1
            self.record_participant_number()

            self.simulator.simulate_beh_data()

            estimator = ParameterEstimator()
            estimator.instantiate_sim_obj(
                exp_data=self.simulator.data,
                task_configs=self.simulator.task_configs,
                bayesian_comps=self.simulator.bayesian_comps
                )

            # Embed simulation params in estimator sim object
            estimator.sim_object.sim_params = self.sim_params

            print("Starting brute-force estimation for tau",
                    f"tau_gen: {self.sim_params.current_tau_gen}")

            mle_tau_estimate = estimator.estimate_tau(method="brute_force")
            self.record_tau_estimate(mle_tau_estimate)

            # Estimate lambda, if appliclabe
            if np.isnan(self.sim_params.current_lambda_gen):
                mle_lambda_estimate = np.nan
            else:
                mle_lambda_estimate = estimator.estimate_lambda(
                    method="brute_force")

            self.record_lambda_estimate(mle_lambda_estimate)

            self.save_data()

    def iterate_data_generating_lambda_space(self):
        for lambda_gen in self.simulator.sim_params.lambda_gen_space:
            self.simulator.sim_params.current_lambda_gen = lambda_gen

            self.record_data_generating_sim_params()

            self.iterate_participants()

    def iterate_data_generating_tau_space(self):
        for tau_gen in self.simulator.sim_params.tau_space_gen:
            self.simulator.sim_params.current_tau_gen = tau_gen
            self.iterate_data_generating_lambda_space()

    def iterate_data_generating_agent_model_space(self):
        for agent_model in self.simulator.sim_params.agent_space_gen:
            self.sim_params.current_agent_attributes = AgentInitObject(
                agent_model)
            self.sim_params.current_agent_model = agent_model
            self.iterate_data_generating_tau_space()

    def iterate_repetitions(self):
        for repetition in self.sim_params.repetitions:
            self.sim_params.current_rep = repetition + 1
            self.iterate_data_generating_agent_model_space()

    def start_simulation_and_estimation_routine(self):
        self.iterate_repetitions()
