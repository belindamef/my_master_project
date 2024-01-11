#!/usr/bin/env python3
"""
This script starts the simulation of agent-task-interactions for given
experimental parameter and agent model and saves simulated behavioral data.

Author: Belinda Fleischmann
"""

import time
import numpy as np
from utilities.config import DataHandler, DirectoryManager, get_arguments
from utilities.simulation_methods import Simulator, GenModelNParameterSpaces
from utilities.task import Task, TaskStatesConfigurator, TaskNGridParameters
from utilities.agent import AgentAttributes, StochasticMatrices
from utilities.very_plotter_new import VeryPlotter


def main(task_params: TaskNGridParameters,
         sim_params: GenModelNParameterSpaces):
    """Main function"""

    dir_mgr = DirectoryManager()
    dir_mgr.define_raw_beh_data_out_path(
        data_type="sim",
        exp_label=EXP_LABEL,
        make_dir=True
        )

    # Load or create task configuration-specific state spaces
    task_state_values = TaskStatesConfigurator(
        path=dir_mgr.paths,
        task_params=task_params
    ).get_task_state_values(EXP_LABEL)

    # Create model task object to store and transfer state spaces
    task_model = Task(
        state_values=task_state_values,
        task_params=task_params
        )

    # Load or create Stochastic Matrices for Hidden Markov Model
    stoch_matrices = StochasticMatrices(
        task_model=task_model,
        task_params=task_params,
    ).compute_or_load_components()

    simulator = Simulator(
        state_values=task_state_values,
        agent_stoch_matrices=stoch_matrices,
        task_params=task_params
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

                        simulated_data = simulator.simulate_beh_data(
                            sim_params=sim_params
                            )

                        DataHandler(
                            paths=dir_mgr.paths,
                            exp_label=EXP_LABEL
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
    start = time.time()
    arguments = get_arguments()

    EXP_LABEL = "test_ahmm_01_09"

    # Define task configuration parameters
    task_params = TaskNGridParameters(
        dim=2,
        n_hides=2,
        n_blocks=1,
        n_rounds=1,
        n_trials=12
        )

    # Define data generating model and parameter spaces
    sim_params = GenModelNParameterSpaces()

    if arguments.parallel_computing:
        sim_params.get_params_from_args(arguments)
    else:
        sim_params.define_params_manually(
            agent_gen_space=["A1"],
            tau_gen_space=[0.01],
            lambda_gen_space=[np.nan]
            )
        sim_params.define_numbers(
            n_rep=1,
            n_part=1
            )

    # Start simulation
    main(
        task_params=task_params,
        sim_params=sim_params
        )

    end = time.time()
    print(f"Total time for simulation: {round((end-start), ndigits=2)} sec.")
