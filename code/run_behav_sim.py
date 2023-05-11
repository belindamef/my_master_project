#!/usr/bin/env python3
"""
This script starts the simulation of agent-task-interactions for given
experimental parameter and agent models.

Task configurations will be either loaded from existing configuration files
or sampled at random if not existing for given experimental parameters.
This script calls the method ./utilities/simulation_methods.py which
iterates over the here given set of agents and performs the interactions
for each agent and task configuration.

Author: Belinda Fleischmann
"""

import time
from utilities.config import DirectoryManager, TaskConfigurator
from utilities.simulation_methods import Simulator, SimulationParameters
from utilities.modelling import AgentInitObject, BayesianModelComps


def main():

    n_participants = 1
    agent_model_space = ["C1", "C2", "C3", "A1", "A2", "A3"]

    # Create all objects needed to cofigure, perform and record simulation
    dir_mgr = DirectoryManager()
    dir_mgr.create_beh_data_out_dir()
    task_configs = TaskConfigurator(dir_mgr.paths).get_config(
        config_label="exp_msc")
    bayesian_comps = BayesianModelComps(task_configs.params).get_comps()
    simulator = Simulator(task_configs=task_configs,
                          bayesian_comps=bayesian_comps)
    sim_params = SimulationParameters()
    simulator.sim_params = sim_params

    for agent_model in agent_model_space:
        sim_params.agent_attr = AgentInitObject(
            agent_model).def_attributes()
        if not sim_params.agent_attr.is_deterministic:
            n_participants = 50

        for participant in range(n_participants):
            sim_params.this_part = participant
            dir_mgr.prepare_beh_output(sim_params)
            simulator.simulate_beh_data()
            dir_mgr.save_data_to_tsv(simulator.data)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total time for simulation: {round((end-start), ndigits=2)} sec.")
