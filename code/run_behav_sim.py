#!/usr/bin/env python3
"""
This script starts the simulation of agent-task-interactions for given
experimental parameter and agent model and saves simulated behavioral data.

Author: Belinda Fleischmann
"""

import time
import logging
import numpy as np
from pympler import asizeof
from utilities.config import DataHandler, DirectoryManager, get_arguments
from utilities.simulation_methods import Simulator, GenModelNParameterSpaces
from utilities.task import TaskSetsNCardinalities, TaskStatesConfigurator
from utilities.task import TaskNGridParameters
from utilities.agent import AgentAttributes, StochasticMatrices
from utilities.very_plotter_new import VeryPlotter


def main(exp_label: str,
         task_params: TaskNGridParameters,
         sim_params: GenModelNParameterSpaces,
         dir_mgr: DirectoryManager):
    """Main function"""

    # Load or create task configuration-specific state spaces
    task_state_values = TaskStatesConfigurator(
        path=dir_mgr.paths,
        task_params=task_params
    ).get_task_state_values(exp_label)

    logging.info("------------------------------------------")
    logging.info("Loading/Computing Task State and observation sets")
    logging.info("------------------------------------------")
    # Create model task object to store and transfer state spaces
    task_sets_n_cardinalities = TaskSetsNCardinalities(
        task_params=task_params,
        ).compute_or_load_sets()

    logging.info("------------------------------------------")
    logging.info("Loading/Computing Stochastic Matrices")
    logging.info("------------------------------------------")
    # Load or create Stochastic Matrices for Hidden Markov Model
    stoch_matrices = StochasticMatrices(
        task_states_n_cardins=task_sets_n_cardinalities,
        task_params=task_params,
    ).compute_or_load_components()

    logging.info(" ----- SUMMARY SHAPES AND SIZES ----------------------")
    logging.info("                 Value/Shape           Size")
    logging.info(" Cardinality  n: %s",
                 task_sets_n_cardinalities.n)
    logging.info("                                       %s",
                 asizeof.asizeof(task_sets_n_cardinalities.n))
    logging.info(" Cardinality  m: %s",
                 task_sets_n_cardinalities.m,)
    logging.info("                                       %s",
                 asizeof.asizeof(task_sets_n_cardinalities.m))
    logging.info("          set S: %s",
                 task_sets_n_cardinalities.S.shape)
    logging.info("                                       %s",
                 asizeof.asizeof(task_sets_n_cardinalities.S))
    logging.info("          set O: %s",
                 task_sets_n_cardinalities.O_.shape)
    logging.info("                                       %s",
                 asizeof.asizeof(task_sets_n_cardinalities.O_))
    logging.info("           beta: %s",
                 stoch_matrices.beta_0.shape)
    logging.info("                                       %s",
                 asizeof.asizeof(stoch_matrices.beta_0))
    logging.info("            Phi: (5, %s, %s)",
                 stoch_matrices.Phi[0].shape[0],
                 stoch_matrices.Phi[0].shape[1])
    logging.info("                                       %s",
                 asizeof.asizeof(stoch_matrices.Phi))
    logging.info("          Omega: (2, %s, %s)",
                 stoch_matrices.Omega['drill'].shape[0],
                 stoch_matrices.Omega['drill'].shape[1])
    logging.info("                                       %s \n",
                 asizeof.asizeof(stoch_matrices.Omega))

    simulator = Simulator(
        state_values=task_state_values,
        agent_stoch_matrices=stoch_matrices,
        task_params=task_params,
        task_sets_n_cardinalities=task_sets_n_cardinalities
        )

    for repetition in sim_params.repetition_numbers:
        sim_params.current_rep = repetition

        for agent_name in sim_params.agent_space_gen:
            sim_params.current_agent_gen_init_obj = AgentAttributes(
                agent_model_name=agent_name)
            sim_params.current_agent_gen = agent_name

            for tau_gen in sim_params.tau_space_gen:
                sim_params.current_tau_gen = tau_gen

                for lambda_gen in sim_params.lambda_gen_space:
                    sim_params.current_lambda_gen = lambda_gen

                    for participant in sim_params.participant_numbers:
                        sim_params.current_part = participant

                        sub_id = sim_params.create_agent_sub_id(
                            current_part=participant,
                            current_rep=repetition
                            )
                        dir_mgr.define_sim_beh_output_paths(sub_id=sub_id)

                        logging.info("------------------------------------------")
                        logging.info("Start behavioral data simulation")
                        logging.info("------------------------------------------")
                        simulated_data = simulator.simulate_beh_data(
                            sim_params=sim_params
                            )

                        logging.info("-------End of simulation------------- \n")
                        DataHandler(
                            paths=dir_mgr.paths,
                            exp_label=exp_label
                        ).save_data_to_tsv(
                            data=simulated_data,
                            filename=dir_mgr.paths.this_sub_beh_out_fn
                                )

                        VeryPlotter(
                            paths=dir_mgr.paths
                        ).plot_heat_maps_of_belief_states(
                            task_params=task_params,
                            beh_data=simulated_data
                            )


if __name__ == "__main__":

    # Define experiment / simulation label
    DIM = 5
    HIDES = 6
    EXP_LABEL = f"test_dim-{DIM}_h-{HIDES}_02_12"
    comment = "Make set O sparse"

    # Prepare output directory
    dir_manager = DirectoryManager()
    dir_manager.define_raw_beh_data_out_path(
        data_type="sim",
        exp_label=EXP_LABEL,
        make_dir=True
        )

    # Prepare logging
    log_fn = f"{dir_manager.paths.this_sim_rawdata}/out.log"
    logging.basicConfig(
        filename=log_fn,
        filemode='w',
        level=logging.INFO,
        # level=logging.DEBUG,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'  # Format without milliseconds
    )
    # Create a FileHandler to flush log messages to the file in real-time
    file_handler = logging.FileHandler(log_fn)
    logging.getLogger().addHandler(file_handler)

    logging.info("SIMULATION LABEL: '%s' \n", EXP_LABEL)
    logging.info("Comment: %s \n", comment)
    start = time.time()

    # Get arguements from environment (in case is started from bash script)
    arguments = get_arguments()

    # Define task configuration parameters
    task_parameters = TaskNGridParameters(
        dim=DIM,
        n_hides=HIDES,
        n_blocks=1,
        n_rounds=1,
        n_trials=12
        )

    logging.info("------------------------------------------")
    logging.info("Task Parameters:")
    logging.info("------------------------------------------")
    for attr_name, attr_value in vars(task_parameters).items():
        logging.info("%s: %s", attr_name, attr_value)
    logging.info("------------------------------------------\n")

    # Define data generating model and parameter spaces
    sim_parameters = GenModelNParameterSpaces()

    if arguments.parallel_computing:
        sim_parameters.get_params_from_args(arguments)
    else:
        sim_parameters.define_params_manually(
            agent_gen_space=["A1"],
            tau_gen_space=[0.01],
            lambda_gen_space=[np.nan]
            )
        sim_parameters.define_numbers(
            n_rep=1,
            n_part=1
            )

    logging.info("------------------------------------------")
    logging.info("Simulation Parameters:")
    logging.info("------------------------------------------")
    for attr_name, attr_value in vars(sim_parameters).items():
        logging.info("%s: %s", attr_name, attr_value)
    logging.info("------------------------------------------\n")

    try:
        main(
            exp_label=EXP_LABEL,
            task_params=task_parameters,
            sim_params=sim_parameters,
            dir_mgr=dir_manager
            )
    except Exception as e:
        logging.error("An error occurred: %s", e)

    end = time.time()
    logging.info("Total time for simulation: %.2f sec.",
                 round((end-start), ndigits=2))
