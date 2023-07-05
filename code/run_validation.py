"""
This script evaluates and visualizes beh_model recovery simulations.

Author: Belinda Fleischmann
"""

import time
from utilities.config import DirectoryManager, TaskConfigurator, get_arguments
from utilities.simulation_methods import Simulator, SimulationParameters
from utilities.modelling import BayesianModelComps
from utilities.validation_methods import Validator
import numpy as np


def define_simulation_parameters() -> SimulationParameters:
    sim_parameters = SimulationParameters()

    if arguments.parallel_computing:
        sim_parameters.get_params_from_args(arguments)
    else:
        sim_parameters.define_n_reps_and_participants_manually()
        sim_parameters.define_params_manually()
    return sim_parameters


def adjust_total_trial_numbers(task_configuration_object: object):
    task_configuration_object.params.n_blocks = TEST_N_BLOCKS
    task_configuration_object.params.n_rounds = TEST_N_ROUNDS
    task_configuration_object.params.n_trials = TEST_N_TRIALS


def main():
    dir_mgr = DirectoryManager()
    dir_mgr.create_val_out_dir(out_dir_label=OUT_DIR_LABEL, version=VERSION_NO)

    task_config = TaskConfigurator(dir_mgr.paths).get_config(TASK_CONFIG_LABEL)
    bayesian_comps = BayesianModelComps(task_config.params).get_comps()

    sim_params = define_simulation_parameters()
    simulator = Simulator(task_config, bayesian_comps, sim_params)

    if IS_QUICK_TEST:
        adjust_total_trial_numbers(task_config)

    validator = Validator(sim_params, simulator, dir_mgr)
    validator.run_simulation_and_estimation_routine()


if __name__ == "__main__":
    start = time.time()
    arguments = get_arguments()

    TASK_CONFIG_LABEL = "exp_msc"
    OUT_DIR_LABEL = "delete"
    VERSION_NO = "1"

    IS_QUICK_TEST = True
    TEST_N_BLOCKS = 1
    TEST_N_ROUNDS = 10
    TEST_N_TRIALS = 12

    main()
    end = time.time()
    print(f"Total time for beh_model validation: "
          f"{round((end-start), ndigits=2)} sec.")
