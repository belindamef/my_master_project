#!/usr/bin/env python3
"""This script evaluates beh_model recovery analyses."""

import time
import os
import numpy as np
from utilities.config import DirectoryManager, DataHandler
from utilities.config import TaskConfigurator, get_arguments
from utilities.simulation_methods import Simulator, SimulationParameters
from utilities.agent import BayesianModelComps
from utilities.model_fit_methods import ModelFitter


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


def define_model_recovery_parameters(model_fitting_object: ModelFitter):
    """Function to set candidate model and parameter spaces for model recovery
    as hardcoded under the if_name==main idiom."""
    model_fitting_object.estimator.est_params.def_params_manually(
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
    if outfile_exists:
        print(f"Skipping model fitting Routine for {out_filename}, "
              "output file already exists")
    return outfile_exists


def main():
    """Main function that runs model validation routine."""
    dir_mgr = DirectoryManager()
    dir_mgr.define_raw_beh_data_out_path(data_type="exp",
                                         out_dir_label=EXP_LABEL,
                                         make_dir=False)
    dir_mgr.define_model_comp_results_path(dir_label=EXP_LABEL,
                                           version=VERSION,
                                           make_dir=True)
    data_loader = DataHandler(dir_mgr.paths, EXP_LABEL)
    exp_ev_all_subs_df = data_loader.load_exp_events()

    task_config = TaskConfigurator(dir_mgr.paths).get_config(EXP_LABEL)
    bayesian_comps = BayesianModelComps(task_config.params).get_comps()

    if IS_QUICK_TEST:
        adjust_total_trial_numbers(task_config)

    sim_params = define_simulation_parameters()
    simulator = Simulator(task_config, bayesian_comps)
    model_fitter = ModelFitter(sim_params, simulator, dir_mgr)
    define_model_recovery_parameters(model_fitter)

    if arguments.parallel_computing:
        participant_list = arguments.participant
    else:
        participant_list = exp_ev_all_subs_df.sub_id.unique().tolist()

    for repetition in sim_params.repetition_numbers:
        sim_params.current_rep = repetition + 1

        for participant in participant_list:
            sim_params.current_part = participant

            this_participants_data = exp_ev_all_subs_df[
                exp_ev_all_subs_df.sub_id == participant]
            this_participants_data = this_participants_data.rename(
                columns={"action": "a", "round": "round_"})

            dir_mgr.define_model_comp_results_filename(sub_id=participant)

            outfile_thisparams_exists = check_output_existence(
                dir_mgr.paths.this_sub_model_comp_results_fn)

            if not outfile_thisparams_exists:
                model_fitter.run_model_comp_routine(
                    data=this_participants_data)


if __name__ == "__main__":
    arguments = get_arguments()

    EXP_LABEL = "exp_msc"
    VERSION = "test_1"

    # Define Simulation parameters
    N_REPS = 1
    N_PARTS = 1
    AGENT_GEN_SPACE = ["C1", "C2", "C3", "A1", "A2", "A3"]
    TAU_GEN_SPACE = np.linspace(0.01, 0.5, 5).tolist()
    LAMBDA_GEN_SPACE = np.linspace(0, 1, 5).tolist()

    AGENT_CAND_SPACE = ["C1", "C2", "C3", "A1", "A2", "A3"]
    TAU_CAND_SPACE = np.linspace(0.01, 0.3, 3).tolist()
    LAMBDA_CAND_SPACE = np.linspace(0.25, 0.75, 3).tolist()

    IS_QUICK_TEST = True
    TEST_N_BLOCKS = 1
    TEST_N_ROUNDS = 1
    TEST_N_TRIALS = 12

    start = time.time()
    main()
    end = time.time()

    print(f"Total time for beh_model validation: "
          f"{round((end-start), ndigits=2)} sec.")
    # out_file.close()
