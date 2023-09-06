#!/usr/bin/env python3
"""This script evaluates and visualizes beh_model recovery analyses."""

import time
import os
import numpy as np
from utilities.config import DirectoryManager, TaskConfigurator, get_arguments
from utilities.config import DataHandler
from utilities.simulation_methods import SimulationParameters
from utilities.agent import AgentAttributes, BayesianModelComps
from utilities.validation_methods import Validator, ValidationParameters
from utilities.estimation_methods import EstimationParameters


def define_simulation_parameters() -> SimulationParameters:
    """Function to confifgure simulation parameters, either (a) from arugments
    passed from bash script or command line, or defined under if name == main
    idiom if no arguments are given """
    sim_parameters = SimulationParameters()

    if arguments.parallel_computing:
        sim_parameters.get_params_from_args(arguments)
    else:

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


def adjust_tau_gen_space(sim_params: SimulationParameters):
    """Function to set the data generating tau parameter space depending on
    the data generating behavioral model."""
    if "A" in sim_params.current_agent_gen:
        sim_params.tau_space_gen = TAU_GEN_SPACE

    elif "C" in sim_params.current_agent_gen:
        sim_params.tau_space_gen = [np.nan]


def adjust_lambda_gen_space(sim_params: SimulationParameters):
    """Function to set the data generating lambda parameter space depending on
    the data generating behavioral model."""
    if "C" in sim_params.current_agent_gen:
        sim_params.lambda_gen_space = [np.nan]
    elif sim_params.current_agent_gen in ["A1", "A2"]:
        sim_params.lambda_gen_space = [np.nan]
    elif sim_params.current_agent_gen == "A3":
        sim_params.lambda_gen_space = LAMBDA_GEN_SPACE


def define_estimation_parameters() -> EstimationParameters:
    """_summary_

    Returns:
        EstimationParameters: _description_
    """
    estim_params = EstimationParameters()
    if arguments.parallel_computing:
        estim_params.get_params_from_args(arguments)
        estim_params.agent_candidate_space = AGENT_CAND_SPACE
    else:
        estim_params.def_params_manually(
            agent_candidate_space=AGENT_CAND_SPACE,
            tau_bf_cand_space=TAU_CAND_SPACE,
            lambda_bf_cand_space=LAMBDA_CAND_SPACE
            )
    return estim_params


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
        print(f"Skipping recovery Routine for {out_filename}, "
              "output file already exists")
    return outfile_exists


def prepare_path_variables(dir_mgr_obj: DirectoryManager):
    """_summary_

    Args:
        dir_mgr_obj (DirectoryManager): _description_
    """
    dir_mgr_obj.define_model_recov_results_path(exp_label=EXP_LABEL,
                                                version=VERSION,
                                                make_dir=True)
    dir_mgr_obj.define_raw_beh_data_out_path(data_type="exp",
                                             exp_label=EXP_LABEL,
                                             make_dir=False)
    dir_mgr_obj.define_model_est_results_path(exp_label=EXP_LABEL,
                                              version=VERSION,
                                              make_dir=True)


def run_model_recovery_routine(sim_params: SimulationParameters,
                               val_params: ValidationParameters,
                               validator: Validator,
                               dir_mgr: DirectoryManager,
                               data_handler: DataHandler):
    """_summary_

    Args:
        sim_params (SimulationParameters): _description_
        validator (Validator): _description_
        dir_mgr (DirectoryManager): _description_
        data_handler (DataHandler): _description_
    """
    for repetition in val_params.repetition_numbers:
        val_params.current_rep = repetition + 1

        for gen_agent in sim_params.agent_space_gen:
            sim_params.current_agent_gen_init_obj = AgentAttributes(gen_agent)
            sim_params.current_agent_gen = gen_agent
            if not arguments.parallel_computing:
                adjust_tau_gen_space(sim_params)

            for tau_gen in sim_params.tau_space_gen:
                sim_params.current_tau_gen = tau_gen
                if not arguments.parallel_computing:
                    adjust_lambda_gen_space(sim_params)

                for lambda_gen in sim_params.lambda_gen_space:
                    sim_params.current_lambda_gen = lambda_gen

                    for participant in val_params.participant_numbers:
                        val_params.current_part = participant
                        sub_id = sim_params.create_agent_sub_id(participant,
                                                                repetition)
                        dir_mgr.define_model_recov_results_filename(sub_id)
                        outfile_thisparams_exists = check_output_existence(
                            dir_mgr.paths.this_sub_model_recov_result_fn)
                        if not outfile_thisparams_exists:
                            recovery_results = validator.run_model_recovery()
                            data_handler.save_data_to_tsv(
                                recovery_results,
                                dir_mgr.paths.this_sub_model_recov_result_fn
                                )


def run_model_comparison_routine(val_params: ValidationParameters,
                                 dir_mgr: DirectoryManager,
                                 validator: Validator,
                                 data_handler: DataHandler):
    """_summary_

    Args:
        val_params (ValidationParameters): _description_
        dir_mgr (DirectoryManager): _description_
        validator (Validator): _description_
        data_handler (DataHandler): _description_
    """
    exp_ev_all_subs_df = data_handler.load_exp_events()
    if arguments.parallel_computing:
        participant_list = arguments.participant
    else:
        participant_list = exp_ev_all_subs_df.sub_id.unique().tolist()

    for repetition in val_params.repetition_numbers:
        val_params.current_rep = repetition + 1

        for participant in participant_list:
            val_params.current_part = participant

            this_participants_data = exp_ev_all_subs_df[
                exp_ev_all_subs_df.sub_id == participant]
            this_participants_data = this_participants_data.rename(
                columns={"action": "a", "round": "round_"})

            dir_mgr.define_model_est_results_filename(sub_id=participant)

            outfile_thisparams_exists = check_output_existence(
                dir_mgr.paths.this_sub_model_est_results_fn)

            if not outfile_thisparams_exists:
                estimation_results = validator.run_model_estimation(
                    data=this_participants_data)
                data_handler.save_data_to_tsv(
                    estimation_results,
                    dir_mgr.paths.this_sub_model_est_results_fn
                )


def main():
    """Main function that runs model validation routine."""
    dir_mgr = DirectoryManager()
    prepare_path_variables(dir_mgr)
    data_handler = DataHandler(dir_mgr.paths, EXP_LABEL)

    task_config = TaskConfigurator(dir_mgr.paths).get_config(EXP_LABEL)
    bayesian_comps = BayesianModelComps(task_config.params).get_comps()

    if IS_QUICK_TEST:
        adjust_total_trial_numbers(task_config)

    sim_params = define_simulation_parameters()
    val_params = define_validation_parameters()
    est_params = define_estimation_parameters()

    validator = Validator(sim_params=sim_params,
                          val_params=val_params,
                          task_configs=task_config,
                          bayesian_comps=bayesian_comps,
                          est_params=est_params)

    if RUN_RECOVERY:
        run_model_recovery_routine(sim_params=sim_params,
                                   val_params=val_params,
                                   validator=validator,
                                   dir_mgr=dir_mgr,
                                   data_handler=data_handler)

    if RUN_ESTIMATION_EXP:
        run_model_comparison_routine(val_params=val_params,
                                     dir_mgr=dir_mgr,
                                     validator=validator,
                                     data_handler=data_handler)


if __name__ == "__main__":
    arguments = get_arguments()

    EXP_LABEL = "exp_msc"
    if arguments.parallel_computing:
        VERSION = arguments.version
    else:
        VERSION = "debug_from_script_0905"

    # Define repetition_parameters
    N_REPS = 1
    N_PARTS = 1

    # Define Simulation parameters, and generating parameter sapce
    AGENT_GEN_SPACE = ["C1", "C2", "C3", "A1", "A2", "A3"]
    TAU_GEN_SPACE = np.linspace(0.01, 0.5, 2).tolist()
    LAMBDA_GEN_SPACE = np.linspace(0, 1, 2).tolist()

    # Define parameter estimation candidate space
    AGENT_CAND_SPACE = ["C1", "C2", "C3", "A1", "A2", "A3"]
    TAU_CAND_SPACE = np.linspace(0.01, 0.3, 2).tolist()
    LAMBDA_CAND_SPACE = np.linspace(0.25, 0.75, 2).tolist()

    # Configure quick test
    IS_QUICK_TEST = True
    TEST_N_BLOCKS = 1
    TEST_N_ROUNDS = 1
    TEST_N_TRIALS = 3

    RUN_RECOVERY = True
    RUN_ESTIMATION_EXP = False

    start = time.time()
    main()
    end = time.time()

    print(f"Total time for beh_model validation: "
          f"{round((end-start), ndigits=2)} sec.")
