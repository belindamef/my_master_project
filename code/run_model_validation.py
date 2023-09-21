#!/usr/bin/env python3
"""This script evaluates model validation performances.

This includes
    (1) model recovery: data simulation, and paramer and model validation with
                        simulated data.
    (2) model estimation: i.e. model validation with experimental data
    (3) pep evalation: grp-lvl evaluation of PEP for each analzing model and
                       each dataset

Set respective boolean variables (RUN_RECOVERY or RUN_ESTIMATION_EXP or
EVAL_PEP) as True or False.

Script is robust in that it will skip routines if respective outputs already
exists on disk.
"""

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
    passed from sellscript or command line, or defined under if name == main
    idiom
    """
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
    """Function to adjust the data generating tau parameter space depending on
    the data generating behavioral model. [nan] for control agents. """
    if "A" in sim_params.current_agent_gen:
        sim_params.tau_space_gen = TAU_GEN_SPACE

    elif "C" in sim_params.current_agent_gen:
        sim_params.tau_space_gen = [np.nan]


def adjust_lambda_gen_space(sim_params: SimulationParameters):
    """Function to adjust the data generating lambda parameter space depending
    on the data generating agent model. i.e. [nan] for all agents beside "A3".
    """
    if "C" in sim_params.current_agent_gen:
        sim_params.lambda_gen_space = [np.nan]
    elif sim_params.current_agent_gen in ["A1", "A2"]:
        sim_params.lambda_gen_space = [np.nan]
    elif sim_params.current_agent_gen == "A3":
        sim_params.lambda_gen_space = LAMBDA_GEN_SPACE


def define_estimation_parameters() -> EstimationParameters:
    """Function to define agent model and parameter candidate spaces for brute
    force estimations. If this script was called from command line or
    shellscript, values are set to passed arguments during script call, or set
    to hardcoded values otherwise.

    Returns:
    --------
        EstimationParameters: Dataclass object storing estimation parameter
            values
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
    """Function to define validation parameter (number of repetions and 
    participants). If this script was called from command line or
    shellscript, values are set to passed arguments during script call, or set
    to hardcoded values otherwise.

    Returns:
    -------
        ValidationParameters: Dataclass object storing validation parameter
            values
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
        print(f"Skipping validation routine for {out_filename}, "
              "output file already exists \n")
    return outfile_exists


def run_model_recovery_routine(sim_params: SimulationParameters,
                               val_params: ValidationParameters,
                               validator: Validator,
                               dir_mgr: DirectoryManager,
                               data_handler: DataHandler):
    """Function to run model and parameter recovery routine. For each
    generating agent model, parameter values (if applicable) and participant,
    model validation performance is evaluated <val_params.n_repetition> times.

    Args:
    -----
        sim_params (SimulationParameters): Data generating model and parameter
            spaces and current values.
        val_params (ValidationParameters): Validaton parameters
            (i.e., n_repetions, n_participants)
        validator (Validator): Object to run model validation analyses.
        dir_mgr (DirectoryManager): Object to store path variables and manage
            directories.
        data_handler (DataHandler): Object to load and save data to disk.

    Output:
    ------
        Saves one .tsv file with model validation results for each subject,
        i.e. unique combination of generating model, parameter values
        repetition and participant number to disk.

    Note:
    -----
        Function is robust, in that it will skip routine for subjects if
            respective output already exists.
    """
    # Prepare ouput path
    dir_mgr.define_model_recov_results_path(exp_label=EXP_LABEL,
                                            version=VERSION,
                                            make_dir=True)

    # Start recovery routine
    for repetition in val_params.repetition_numbers:
        val_params.current_rep = repetition + 1

        for gen_agent in sim_params.agent_space_gen:
            sim_params.current_agent_gen_init_obj = AgentAttributes(gen_agent)
            sim_params.current_agent_gen = gen_agent

            # Adjust generating tau space depending on generating agent
            if not arguments.parallel_computing:
                adjust_tau_gen_space(sim_params)

            for tau_gen in sim_params.tau_space_gen:
                sim_params.current_tau_gen = tau_gen

                # Adjust generating lambda space depending generating agent
                if not arguments.parallel_computing:
                    adjust_lambda_gen_space(sim_params)

                for lambda_gen in sim_params.lambda_gen_space:
                    sim_params.current_lambda_gen = lambda_gen

                    for participant in val_params.participant_numbers:
                        val_params.current_part = participant

                        # Prepare subject-specific output path
                        sub_id = sim_params.create_agent_sub_id(participant,
                                                                repetition)
                        dir_mgr.define_sub_lvl_model_recov_results_fn(sub_id)

                        # Check if recovery results for this subject exists
                        outfile_thisparams_exists = check_output_existence(
                            dir_mgr.paths.this_sub_model_recov_result_fn)

                        # Run routine, only if results don't exist on disk
                        if not outfile_thisparams_exists:

                            # Evaluate model recovery performance
                            recovery_results = validator.run_model_recovery()

                            # Save results to disk
                            data_handler.save_data_to_tsv(
                                recovery_results,
                                dir_mgr.paths.this_sub_model_recov_result_fn
                                )


def run_model_est_routine(val_params: ValidationParameters,
                          dir_mgr: DirectoryManager,
                          validator: Validator,
                          data_handler: DataHandler):
    """Function to run model estimation routine. For each
    experimental dataset, model estimation performance is evaluated
    <val_params.n_repetition> times.

    Args:
    -----agent model and parameter candidate spaces for brute
    -force estimations
        val_params (ValidationParameters): Validaton parameters
            (i.e., n_repetions, n_participants)
        validator (Validator): Object to run model validation analyses.
        dir_mgr (DirectoryManager): Object to store path variables and manage
            directories.
        data_handler (DataHandler): Object to load and save data to disk.

    Output:
    ------
        Saves one .tsv file with model validation results for each repetition
        to disk.

    Note:
    -----
        Function is robust in that it will skip routine for subjects if
        respective output already exists.
    """
    # Prepare paths
    dir_mgr.define_model_est_results_path(exp_label=EXP_LABEL,
                                          version=VERSION,
                                          make_dir=True)

    # Load data of all subjects
    exp_ev_all_subs_df = data_handler.load_proc_exp_events()

    # Define participant iterable
    if arguments.parallel_computing:
        participant_list = arguments.participant
    else:
        participant_list = exp_ev_all_subs_df.sub_id.unique().tolist()

    # Start routine
    for repetition in val_params.repetition_numbers:
        val_params.current_rep = repetition + 1

        for participant in participant_list:
            val_params.current_part = participant

            # Prepare this subject's dataframe
            this_participants_data = exp_ev_all_subs_df[
                exp_ev_all_subs_df.sub_id == participant]
            this_participants_data = this_participants_data.rename(
                columns={"action": "a", "round": "round_"})

            # Prepare subject-specific output path
            dir_mgr.define_model_est_results_filename(sub_id=participant)

            # Check if recovery results for this subject exists
            outfile_thisparams_exists = check_output_existence(
                dir_mgr.paths.this_sub_model_est_results_fn)

            # Run routine, only if results don't exist on disk
            if not outfile_thisparams_exists:

                # Evaluate model estimation performances
                estimation_results = validator.run_model_estimation(
                    data=this_participants_data)

                # Save results to disk
                data_handler.save_data_to_tsv(
                    data=estimation_results,
                    filename=dir_mgr.paths.this_sub_model_est_results_fn
                )


def evaluate_peps(dir_mgr: DirectoryManager,
                  validator: Validator,
                  data_handler: DataHandler):
    """Evaluate group level PEPs given participant and model specific mll
    values, i.d. n_participants x n_models

    Args:
        dir_mgr (DirectoryManager): _description_
        validator (Validator): _description_
        data_handler (DataHandler): _description_
    """
    # Prepare output path
    dir_mgr.define_grp_lvl_model_validation_results_fn_s()

    # ----Evaluate peps of model recovery -------------------------------------
    # Load validation results
    all_val_results_df = data_handler.load_data_in_one_folder(
        folder_path=dir_mgr.paths.this_model_recov_sub_lvl_results
        )
    # Evaluate PEPs
    peps = validator.evaluate_peps(val_results=all_val_results_df,
                                   data_type="sim")
    # Save PEP results to disk
    data_handler.save_data_to_tsv(
        data=peps,
        filename=dir_mgr.paths.grp_lvl_model_recovery_results_fn)

    # ----Evaluate peps of model estimation -----------------------------------
    # Load validation results
    all_val_results_df = data_handler.load_data_in_one_folder(
        folder_path=dir_mgr.paths.this_model_est_sub_lvl_results)
    # Evaluate PEPs
    peps = validator.evaluate_peps(val_results=all_val_results_df,
                                   data_type="exp")
    # Save PEP results to disk
    data_handler.save_data_to_tsv(
        data=peps,
        filename=dir_mgr.paths.grp_lvl_model_recovery_results_fn)


def main():
    """Main function to prepare and run model validation routines."""

    # Prepare path variables and init data handler object
    dir_mgr = DirectoryManager()
    data_handler = DataHandler(paths=dir_mgr.paths, exp_label=EXP_LABEL)

    # Load task configurations and bayesian model components from disk
    task_config = TaskConfigurator(path=dir_mgr.paths).get_config(
        config_label=EXP_LABEL)
    bayesian_comps = BayesianModelComps(
        task_design_params=task_config.params).get_comps()

    # For debigging, adjust total number of trials
    if IS_TEST:
        adjust_total_trial_numbers(task_configuration_object=task_config)

    # Prepare paramter spaces and validation object
    sim_params = define_simulation_parameters()
    val_params = define_validation_parameters()
    est_params = define_estimation_parameters()

    validator = Validator(sim_params=sim_params,
                          val_params=val_params,
                          task_configs=task_config,
                          bayesian_comps=bayesian_comps,
                          est_params=est_params)

    # Run selected validation routines
    if RUN_RECOVERY:
        run_model_recovery_routine(sim_params=sim_params,
                                   val_params=val_params,
                                   validator=validator,
                                   dir_mgr=dir_mgr,
                                   data_handler=data_handler)

    if RUN_ESTIMATION_EXP:
        run_model_est_routine(val_params=val_params,
                              dir_mgr=dir_mgr,
                              validator=validator,
                              data_handler=data_handler)

    if EVAL_PEP:
        evaluate_peps(dir_mgr=dir_mgr,
                      validator=validator,
                      data_handler=data_handler)


if __name__ == "__main__":

    # ============================
    #        Settings
    # ============================
    # Select validation routines to run
    RUN_RECOVERY = False  # If True, run model and parameter recovery
    RUN_ESTIMATION_EXP = True  # If True run model validuation with exp. data
    EVAL_PEP = False  # If True, evaluate peps (group level)

    # Define task configuration and version labels
    EXP_LABEL = "exp_msc"  # "exp_msc" for master project task configuration
    VERSION = "res-3_0918"  # Will be added to pip3 output directory path name
    VERSION = "debug_0921"  # Will be added to output directory path name

    # Define number of repetitions and participants (only applied for recovery)
    N_REPS = 1
    N_PARTS = 1

    # Define data generating model and parameter spaces for simulation
    AGENT_GEN_SPACE = ["C1", "C2", "C3", "A1", "A2", "A3"]
    TAU_GEN_SPACE = np.linspace(0.01, 0.5, 2).tolist()
    LAMBDA_GEN_SPACE = np.linspace(0, 1, 2).tolist()

    # Define model and parameter spaces for brute-force estimations
    AGENT_CAND_SPACE = ["C1", "C2", "C3", "A1", "A2", "A3"]
    TAU_CAND_SPACE = np.linspace(0.01, 0.3, 3).tolist()
    LAMBDA_CAND_SPACE = np.linspace(0.25, 0.75, 3).tolist()

    # Fetch arguments passed when called from terminal or shellscript
    arguments = get_arguments()
    # Overwrite settings with arguments from script call
    if arguments.parallel_computing:
        VERSION = arguments.version  # TODO: Add remaining args from env

    # For debugging, adjust total number of trials
    IS_TEST = True  # If True, use test no. trials. Esle use exp's default no.
    TEST_N_BLOCKS = 1
    TEST_N_ROUNDS = 1
    TEST_N_TRIALS = 3

    # ============================
    #        End of settings
    # ============================

    start = time.time()
    main()
    end = time.time()
    print(f"Total time for this validation: "
          f"{round((end-start), ndigits=2)} sec.")
