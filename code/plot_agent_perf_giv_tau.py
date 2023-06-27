import os
import glob
import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as pyplot
from utilities import very_plotter
from utilities.config import DirectoryManager
import time
import argparse
from utilities.config import DirectoryManager, TaskConfigurator
from utilities.simulation_methods import Simulator, SimulationParameters
from utilities.modelling import BayesianModelComps
from utilities.validation_methods import Validator
from utilities.modelling import AgentInitObject
import numpy as np


def get_arguments():
    """Get arguments from environment, if script is executed from command line
    or with a bash jobwrapper."""
    parser = argparse.ArgumentParser(description='Run model validation.')
    parser.add_argument('--parallel_computing', action="store_true")
    parser.add_argument('--repetition', type=int, nargs='+')
    parser.add_argument('--agent_model', type=str, nargs='+')
    parser.add_argument('--tau_value', type=float, nargs='+')
    parser.add_argument('--lambda_value', type=float, nargs='+')
    parser.add_argument('--participant', type=int, nargs='+')
    args = parser.parse_args()
    return args


def define_simulation_parameters() -> SimulationParameters:
    sim_parameters = SimulationParameters()

    if arguments.parallel_computing:
        sim_parameters.get_params_from_args(arguments)

    else:  # Define parameters for local tests, i.e. not parallel computing
        # note to myself: this overwrites default class attributes
        sim_parameters.agent_space_gen = ["A3"]
        sim_parameters.tau_space_gen = np.linspace(0.1, 2., 5)
        sim_parameters.tau_gen_space_if_fixed = [0.1]
        sim_parameters.n_participants = 10
        sim_parameters.lambda_gen_space = np.linspace(0.1, 0.9, 5)
    return sim_parameters


def adjust_total_trial_numbers(task_configuration_object: object):
    task_configuration_object.params.n_blocks = TEST_N_BLOCKS
    task_configuration_object.params.n_rounds = TEST_N_ROUNDS
    task_configuration_object.params.n_trials = TEST_N_TRIALS


def main():

    # Create all objects needed to cofigure, perform and record simulation
    dir_mgr = DirectoryManager()
    dir_mgr.create_raw_beh_data_out_dir(out_dir_label=OUT_DIR_LABEL,
                                        data_type="sim")
    task_configs = TaskConfigurator(dir_mgr.paths).get_config(
        config_label="exp_msc")
    bayesian_comps = BayesianModelComps(task_configs.params).get_comps()
    simulator = Simulator(task_configs=task_configs,
                          bayesian_comps=bayesian_comps)
    sim_params = SimulationParameters()
    simulator.sim_params = sim_params

    for agent_model in agent_model_space:
        sim_params.agent_attr = AgentInitObject(
            agent_model).def_attributes()
        
        if agent_model in ["A1", "A2", "A3"]:
            sim_params.tau_space_gen = np.linspace(0.1, 0.9, 9)

            sim_params.current_lambda_gen = 0.5

            for tau_gen in sim_params.tau_space_gen:
                sim_params.current_tau_gen = tau_gen
                for participant in range(n_participants):
                    sim_params.this_part = participant
                    dir_mgr.prepare_beh_output(sim_params)
                    simulator.simulate_beh_data()
                    dir_mgr.save_data_to_tsv(simulator.data, current_tau_gen=tau_gen)


if __name__ == "__main__":

    arguments = get_arguments()

    TASK_CONFIG_LABEL = "exp_msc"
    OUT_DIR_LABEL = "agent_perf_giv_tau"
    VERSION_NO = "1"

    IS_QUICK_TEST = True
    TEST_N_BLOCKS = 1
    TEST_N_ROUNDS = 10
    TEST_N_TRIALS = 12

    main()
