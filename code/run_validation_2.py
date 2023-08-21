"""
This script evaluates and visualizes beh_model recovery analyses.

Author: Belinda Fleischmann
"""

import time
import os
import numpy as np
from utilities.config import DirectoryManager, TaskConfigurator, get_arguments
from utilities.simulation_methods import Simulator, SimulationParameters
from utilities.modelling import BayesianModelComps, AgentInitObj
from utilities.validation_methods import Validator


def define_simulation_parameters() -> SimulationParameters:
    """Function to confifgure simulation parameters, either (a) from arugments
    passed from bash script or command line, or defined under if name == main
    idiom if no arguments are given """
    sim_parameters = SimulationParameters()

    if arguments.parallel_computing:
        sim_parameters.get_params_from_args(arguments)
    else:
        sim_parameters.define_n_reps_and_participants_manually(
            n_rep=N_REPS, n_part=N_PARTS
        )
        sim_parameters.define_params_manually(
            agent_gen_space=AGENT_GEN_SPACE
        )
    return sim_parameters


def adjust_total_trial_numbers(task_configuration_object: TaskConfigurator):
    """Function to adjust total number of trials as hard coded under
    if_name==main idiom; for quick tests."""
    task_configuration_object.params.n_blocks = TEST_N_BLOCKS
    task_configuration_object.params.n_rounds = TEST_N_ROUNDS
    task_configuration_object.params.n_trials = TEST_N_TRIALS


def define_tau_gen_space(sim_params: SimulationParameters):
    """Function to set the data generating tau parameter space depending on
    the data generating behavioral model."""
    if "A" in sim_params.current_agent_gen:
        sim_params.tau_space_gen = TAU_GEN_SPACE

    elif "C" in sim_params.current_agent_gen:
        sim_params.tau_space_gen = [np.nan]


def define_lambda_gen_space(sim_params: SimulationParameters):
    """Function to set the data generating lambda parameter space depending on
    the data generating behavioral model."""
    if "C" in sim_params.current_agent_gen:
        sim_params.lambda_gen_space = [np.nan]
    elif sim_params.current_agent_gen in ["A1", "A2"]:
        sim_params.lambda_gen_space = [np.nan]
    elif sim_params.current_agent_gen == "A3":
        sim_params.lambda_gen_space = LAMBDA_GEN_SPACE


def define_model_recovery_parameters(valdidator_object: Validator):
    """Function to set candidate model and parameter spaces for model recovery
    as hardcoded under the if_name==main idiom."""
    valdidator_object.estimator.est_params.def_params_manually(
        agent_candidate_space=AGENT_CAND_SPACE,
        tau_bf_cand_space=TAU_CAND_SPACE,
        lambda_bf_cand_space=LAMBDA_CAND_SPACE
        )


def check_output_existence(out_filename: str) -> bool:
    """Function to check if .tsv version of a given output filename, reflecting
    certain parameter values and number of participant and repetition, already
    exists in the output directory.

    Returns
    -------
    outfile_exists : bool
        True, if file exists, False, else
    """
    outfile_exists = os.path.exists(f"{out_filename}.tsv")
    print(f"Skipping recovery Routine for {out_filename}, "
          "output file already exists"
          )
    return outfile_exists


def main():
    """Main function that runs model validation routine."""
    dir_mgr = DirectoryManager()
    dir_mgr.define_val_results_path(dir_label=OUT_DIR_LABEL, version=VERSION_NO,
                                make_dir=True)
    task_config = TaskConfigurator(dir_mgr.paths).get_config(TASK_CONFIG_LABEL)
    bayesian_comps = BayesianModelComps(task_config.params).get_comps()

    if IS_QUICK_TEST:
        adjust_total_trial_numbers(task_config)

    sim_params = define_simulation_parameters()
    simulator = Simulator(task_config, bayesian_comps, sim_params)
    validator = Validator(sim_params, simulator, dir_mgr)
    define_model_recovery_parameters(validator)

    for repetition in sim_params.repetition_numbers:
        sim_params.current_rep = repetition + 1

        for gen_agent in sim_params.agent_space_gen:
            sim_params.current_agent_gen_init_obj = AgentInitObj(gen_agent)
            sim_params.current_agent_gen = gen_agent
            if not arguments.parallel_computing:
                define_tau_gen_space(sim_params)

            for tau_gen in sim_params.tau_space_gen:
                sim_params.current_tau_gen = tau_gen
                if not arguments.parallel_computing:
                    define_lambda_gen_space(sim_params)

                for lambda_gen in sim_params.lambda_gen_space:
                    sim_params.current_lambda_gen = lambda_gen

                    for participant in sim_params.participant_numbers:
                        sim_params.current_part = participant + 1
                        dir_mgr.create_agent_sub_id(sim_params)
                        dir_mgr.define_val_results_filename()
                        outfile_thisparams_exists = check_output_existence(
                            dir_mgr.paths.this_sub_val_result_fn)
                        if not outfile_thisparams_exists:
                            validator.run_param_model_recovery_routine()


if __name__ == "__main__":
    arguments = get_arguments()

    TASK_CONFIG_LABEL = "exp_msc"
    OUT_DIR_LABEL = f"{TASK_CONFIG_LABEL}_test_parallel"
    VERSION_NO = 2

    # Define Simulation parameters
    N_REPS = 1
    N_PARTS = 1
    AGENT_GEN_SPACE = ["C1", "C2", "C3", "A1", "A2", "A3"]
    TAU_GEN_SPACE = np.linspace(0.01, 0.5, 5).tolist()
    LAMBDA_GEN_SPACE = np.linspace(0, 1, 5).tolist()

    AGENT_CAND_SPACE = ["C1", "C2", "C3", "A1", "A2", "A3"]
    TAU_CAND_SPACE = np.linspace(0.01, 0.3, 6).tolist()
    LAMBDA_CAND_SPACE = np.linspace(0.25, 0.75, 6).tolist()

    IS_QUICK_TEST = False
    TEST_N_BLOCKS = 1
    TEST_N_ROUNDS = 1
    TEST_N_TRIALS = 12

    start = time.time()
    main()
    end = time.time()

    print(f"Total time for beh_model validation: "
          f"{round((end-start), ndigits=2)} sec.")
    # out_file.close()
