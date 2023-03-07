#!/usr/bin/env python3
"""
This script starts the simulation of agent-task-interactions for given
experimental parameter and agent models.

Task configurations will be either loaded from existing configuration files
or sampled at random if not existing for given experimental parameters.
This script calls the method ./utilities/agent_task_interaction.py which
iterates over the here given set of agents and performs the interactions
for each agent and task configuration.

Author: Belinda Fleischmann
"""

import os
from utilities.agent_task_interaction import start_agent_task_interaction
from utilities.model_comp import ModelComps
from utilities.create_task_config import TaskConfigurator
from utilities.config import Paths
from utilities.config import TaskDesignParameters
from utilities.config import TaskConfigurator




def main():
    # Get directory paths and create non-existing output directories
    paths = Paths()
    if not os.path.exists(paths.sim_data):
        os.makedirs(paths.sim_data)

    # Prepare task configuration
    task_design_params = TaskDesignParameters()
    task_configurator = TaskConfigurator(paths, task_design_params)
    task_configurator.prepare_task_config()
    task_configs = task_configurator.return_task_configuration()

    # Define sim output dir path and create if not existent
    sim_name = str(input("Enter label for this simulation run: "))
    paths.sim_output = os.path.join(paths.sim_data, sim_name)
    if not os.path.exists(paths.sim_output):
        os.makedirs(paths.sim_output)
    else:
        print("A simualation output directory with this name already exists. \n"
              "Program will be terminated.")
        exit()  # TODO: why sys.exit() better?")

    # Define agent model space
    agent_models = ["C1", "C2", "C3", "A1", "A2", "A3"]


    # Create model components (or load file if existent)
    model_comp = ModelComps(working_dir=working_dir, dim=dim, n_hides=n_hides)


    # Start simulation
    start_agent_task_interaction(
        working_dir=working_dir,
        output_dir=raw_sim_data_dir,
        data_dir=data_dir,
        n_blocks=n_blocks,
        n_rounds=n_rounds,
        n_trials=n_trials,
        dim=dim,
        n_hides=n_hides,
        agent_models=agent_models,
        task_configs=task_configs,
        model_comps=model_comp,
        mode='simulation'
    )


if __name__ == "__main__":
    main()
