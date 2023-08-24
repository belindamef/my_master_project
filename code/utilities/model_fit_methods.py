"""This script contains classes and methods for model validation given
participant data.
"""
import time
import pandas as pd
from utilities.simulation_methods import Simulator, SimulationParameters
from utilities.estimation_methods import Estimator
from utilities.config import DirectoryManager


class ModelFitter:
    """Class to perform model fitting"""
    data_dic: dict
    estimator: Estimator = Estimator()

    def __init__(self, sim_params: SimulationParameters,
                 simulator: Simulator, dir_mgr: DirectoryManager):
        self.sim_params: SimulationParameters = sim_params
        self.simulator: Simulator = simulator
        self.dir_mgr: DirectoryManager = dir_mgr

    def init_data_dic(self):
        """_summary_
        """
        self.data_dic = {
            "participant": [],
            }

        for agent in self.estimator.est_params.agent_candidate_space:
            self.data_dic[f"BIC_{agent}"] = []

    def record_participant_number(self):
        """_summary_
        """
        self.data_dic["participant"].append(self.sim_params.current_part)

    def record_bics(self, bics: dict):
        """_summary_

        Args:
            bics (dict): _description_
        """
        for agent in self.estimator.est_params.agent_candidate_space:
            self.data_dic[f"BIC_{agent}"].append(bics[f"BIC_{agent}"])

    def save_results(self):
        """_summary_
        """
        self.dir_mgr.define_model_comp_results_filename(
            sub_id=str(self.sim_params.current_part))

        mle_df = pd.DataFrame(self.data_dic)

        with open(f"{self.dir_mgr.paths.this_sub_model_comp_results_fn}.tsv",
                  "w", encoding="utf8") as tsv_file:
            tsv_file.write(mle_df.to_csv(sep="\t", na_rep="nan", index=False))

    def evaluate_model_fitting_performance(self, data: pd.DataFrame):
        """_summary_

        Args:
            data (pd.DataFrame): _description_
        """
        bics = self.estimator.evaluate_bic_s(
            data=data, est_method="brute_force", data_type="exp")
        self.record_bics(bics)

    def run_model_comp_routine(self, data: pd.DataFrame):
        """For each participant, simulate behavioral data, estimate parameter
        values and evaluate model recovery performance"""

        self.init_data_dic()
        self.record_participant_number()

        start = time.time()
        self.estimator.instantiate_sim_obj(
            exp_data=data,  # TODO: redundant
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

        self.save_results()
