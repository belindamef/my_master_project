#!/usr/bin/env python3
"""
This script starts the simulation of agent-task-interactions for given
experimental parameter and agent model and saves simulated behavioral data.

Author: Belinda Fleischmann
"""

import time
import argparse
from utilities.config import DirectoryManager, TaskConfigurator
from utilities.simulation_methods import Simulator, SimulationParameters
from utilities.modelling import AgentInitObject, BayesianModelComps


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

    return sim_parameters


def adjust_total_trial_numbers(task_configuration_object: object):
    task_configuration_object.params.n_blocks = TEST_N_BLOCKS
    task_configuration_object.params.n_rounds = TEST_N_ROUNDS
    task_configuration_object.params.n_trials = TEST_N_TRIALS


def main():
    dir_mgr = DirectoryManager()
    dir_mgr.create_raw_beh_data_out_dir(data_type="sim",
                                        out_dir_label=OUT_DIR_LABEL)

    task_config = TaskConfigurator(dir_mgr.paths).get_config(TASK_CONFIG_LABEL)
    bayesian_comps = BayesianModelComps(task_config.params).get_comps()

    if IS_QUICK_TEST:
        adjust_total_trial_numbers(task_config)

    sim_params = define_simulation_parameters()
    simulator = Simulator(task_config, bayesian_comps, sim_params)

    for repetition in sim_params.repetitions:
        sim_params.current_rep = repetition

        for agent_model in sim_params.agent_space_gen:
            sim_params.current_agent_attributes = AgentInitObject(
                agent_model).def_attributes()
            # if not sim_params.current_agent_attributes.is_deterministic:
            #     n_participants = 50  # TODO: nochmal checken, ob brauchen!

            for tau_gen in sim_params.tau_space_gen:
                sim_params.current_tau_gen = tau_gen

                for lambda_gen in sim_params.lambda_gen_space:
                    sim_params.current_lambda_gen = lambda_gen

                    for participant in sim_params.participant_numbers:
                        sim_params.current_part = participant

                        dir_mgr.prepare_beh_output(sim_params)

                        simulator.simulate_beh_data()
    
                        dir_mgr.save_data_to_tsv(simulator.data)


if __name__ == "__main__":
    start = time.time()
    arguments = get_arguments()

    TASK_CONFIG_LABEL = "exp_msc"
    OUT_DIR_LABEL = "exp_msc"

    IS_QUICK_TEST = False
    TEST_N_BLOCKS = 1
    TEST_N_ROUNDS = 2
    TEST_N_TRIALS = 2

    main()

    end = time.time()
    print(f"Total time for simulation: {round((end-start), ndigits=2)} sec.")
