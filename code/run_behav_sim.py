#!/usr/bin/env python3
"""
This script starts the simulation of agent-task-interactions for given
experimental parameter and agent model and saves simulated behavioral data.

Author: Belinda Fleischmann
"""

import time
import numpy as np
from utilities.config import DirectoryManager, TaskConfigurator, TaskDesignParameters, get_arguments
from utilities.config import DataHandler
from utilities.simulation_methods import Simulator, SimulationParameters
from utilities.agent import AgentAttributes, HiddenMarkovModel
from utilities.validation_methods import ValidationParameters


def define_simulation_parameters() -> SimulationParameters:
    """_summary_

    Returns:
        SimulationParameters: _description_
    """
    sim_parameters = SimulationParameters()

    if arguments.parallel_computing:
        sim_parameters.get_params_from_args(arguments)
    else:
        sim_parameters.define_params_manually(
            agent_gen_space=AGENT_GEN_SPACE,
            tau_gen_space=TAU_GEN_SPACE,
            lambda_gen_space=LAMBDA_GEN_SPACE
        )

    return sim_parameters


def define_validation_parameters() -> ValidationParameters:
    """_summary_

    Returns:
        ValidationParameters: _description_
    """
    val_params = ValidationParameters()
    if arguments.parallel_computing:
        val_params.get_params_from_args(arguments)
    else:
        val_params.define_numbers(
            n_rep=N_REPS, n_part=N_PARTS
        )
    return val_params


def adjust_total_trial_numbers(task_configuration_object: TaskConfigurator):
    """_summary_

    Args:
        task_configuration_object (TaskConfigurator): _description_
    """
    task_configuration_object.params.n_blocks = TEST_N_BLOCKS
    task_configuration_object.params.n_rounds = TEST_N_ROUNDS
    task_configuration_object.params.n_trials = TEST_N_TRIALS


def main(task_params: TaskDesignParameters):
    """Main function"""
    dir_mgr = DirectoryManager()
    dir_mgr.define_raw_beh_data_out_path(data_type="sim",
                                         exp_label=OUT_DIR_LABEL,
                                         make_dir=True)

    task_config = TaskConfigurator(
        path=dir_mgr.paths,
        params=task_params
        ).get_config(EXP_LABEL)
    hmm_comps = HiddenMarkovModel(
        task_config.params).compute_or_load_components()

    if IS_QUICK_TEST:
        adjust_total_trial_numbers(task_config)

    sim_params = define_simulation_parameters()
    val_params = define_validation_parameters()
    simulator = Simulator(task_configs=task_config,
                          bayesian_comps=hmm_comps,
                          task_params=task_params)

    for repetition in val_params.repetition_numbers:
        val_params.current_rep = repetition

        for agent_model in sim_params.agent_space_gen:
            sim_params.current_agent_gen_init_obj = AgentAttributes(
                agent_model)
            sim_params.current_agent_gen = agent_model

            for tau_gen in sim_params.tau_space_gen:
                sim_params.current_tau_gen = tau_gen

                for lambda_gen in sim_params.lambda_gen_space:
                    sim_params.current_lambda_gen = lambda_gen

                    for participant in val_params.participant_numbers:
                        val_params.current_part = participant

                        sub_id = sim_params.create_agent_sub_id(participant,
                                                                repetition)
                        dir_mgr.define_sim_beh_output_paths(sub_id)

                        simulated_data = simulator.simulate_beh_data(
                            sim_params)

                        DataHandler(dir_mgr.paths, EXP_LABEL).save_data_to_tsv(
                            simulated_data,
                            dir_mgr.paths.this_sub_beh_out_filename)


if __name__ == "__main__":
    start = time.time()
    arguments = get_arguments()

    EXP_LABEL = "test_ahmm"
    OUT_DIR_LABEL = "test_ahmm_11_30"

    # Define task configuration parameters
    task_params = TaskDesignParameters(
        dim=2,
        n_hides=2,
        n_nodes=4
    )

    # Define simulation parameters
    AGENT_GEN_SPACE = ["A1"]
    TAU_GEN_SPACE = [0.01]
    LAMBDA_GEN_SPACE = [np.nan]

    # Define repetition_parameters
    N_REPS = 1
    N_PARTS = 1

    IS_QUICK_TEST = True
    TEST_N_BLOCKS = 1
    TEST_N_ROUNDS = 1  # NOTE: only 1 round possible because prior_c (!?)
    TEST_N_TRIALS = 2

    main(task_params=task_params)

    end = time.time()
    print(f"Total time for simulation: {round((end-start), ndigits=2)} sec.")
