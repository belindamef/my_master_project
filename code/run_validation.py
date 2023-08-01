"""
This script evaluates and visualizes beh_model recovery analyses.

Author: Belinda Fleischmann
"""

import time
from utilities.config import DirectoryManager, TaskConfigurator, get_arguments
from utilities.simulation_methods import Simulator, SimulationParameters
from utilities.modelling import BayesianModelComps, AgentInitObject
from utilities.validation_methods import Validator
import numpy as np


def define_simulation_parameters() -> SimulationParameters:
    sim_parameters = SimulationParameters()

    if arguments.parallel_computing:
        sim_parameters.get_params_from_args(arguments)
    else:
        sim_parameters.define_n_reps_and_participants_manually()
        sim_parameters.define_params_manually(
            agent_gen_space=AGENT_GEN_SPACE
        )
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

    if IS_QUICK_TEST:
        adjust_total_trial_numbers(task_config)

    sim_params = define_simulation_parameters()

    simulator = Simulator(task_config, bayesian_comps, sim_params)
    validator = Validator(sim_params, simulator, dir_mgr)

    for repetition in sim_params.repetition_numbers:
        sim_params.current_rep = repetition + 1

        for gen_agent in sim_params.agent_space_gen:
            sim_params.current_agent_gen_attributes = AgentInitObject(gen_agent)

            sim_params.current_agent_gen = gen_agent

            # Define simulation parameters if local computing
            if not arguments.parallel_computing:
                if "A" in gen_agent:
                    sim_params.tau_space_gen = TAU_GEN_SPACE

                elif "C" in gen_agent:
                    sim_params.tau_space_gen = [np.nan]

            for tau_gen in sim_params.tau_space_gen:
                sim_params.current_tau_gen = tau_gen

                # Define simulation and recovery parameters if local computing
                if not arguments.parallel_computing:
                    if "C" in gen_agent:
                        sim_params.lambda_gen_space = [np.nan]
                        validator.recoverer.recov_params.def_params_manually(
                            agent_candidate_space=AGENT_CAND_SPACE,
                            tau_bf_cand_space=[np.nan],
                            lambda_bf_cand_space=[np.nan]
                            )

                    elif gen_agent in ["A1", "A2"]:
                        sim_params.lambda_gen_space = [np.nan]
                        validator.recoverer.recov_params.def_params_manually(
                            agent_candidate_space=AGENT_CAND_SPACE,
                            tau_bf_cand_space=TAU_CAND_SPACE,
                            lambda_bf_cand_space=[np.nan]
                            )

                    elif gen_agent == "A3":
                        sim_params.lambda_gen_space = LAMBDA_GEN_SPACE
                        validator.recoverer.recov_params.def_params_manually(
                            agent_candidate_space=AGENT_CAND_SPACE,
                            tau_bf_cand_space=TAU_CAND_SPACE,
                            lambda_bf_cand_space=LAMBDA_GEN_SPACE
                            )

                for lambda_gen in sim_params.lambda_gen_space:
                    sim_params.current_lambda_gen = lambda_gen

                    validator.iterate_participants()
                    validator.evaluate_model_recovery_performance()


if __name__ == "__main__":
    start = time.time()
    arguments = get_arguments()

    TASK_CONFIG_LABEL = "exp_msc"
    OUT_DIR_LABEL = "exp_msc_tests"
    VERSION_NO = "5"

    # Define Simulation parameters
    AGENT_GEN_SPACE = ["C1", "C2", "C3", "A1", "A2"]
    TAU_GEN_SPACE = np.linspace(0.01, 0.5, 5).tolist()
    LAMBDA_GEN_SPACE = np.linspace(0, 1, 5).tolist()

    AGENT_CAND_SPACE = ["C1", "C2", "C3", "A1"]
    TAU_CAND_SPACE = np.linspace(0.01, 0.5, 5).tolist()
    LAMBDA_CAND_SPACE = np.linspace(0, 1, 5).tolist()

    IS_QUICK_TEST = False
    TEST_N_BLOCKS = 1
    TEST_N_ROUNDS = 1
    TEST_N_TRIALS = 12

    main()
    end = time.time()
    print(f"Total time for beh_model validation: "
          f"{round((end-start), ndigits=2)} sec.")
