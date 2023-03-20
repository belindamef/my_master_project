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
from utilities.simulation_methods import Simulator
from utilities.modelling import AgentInitObject


def main():
    simulator = Simulator(mode="behavior_sim")
    simulator.dir_mgr.create_data_out_dir()

    for agent_model in simulator.agent_model_space:
        agent_attr = AgentInitObject(agent_model).def_attributes()
        if agent_attr.is_deterministic:
            simulator.n_repetitions = 1

        simulator.simulate(agent_attr)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total time for simulation: {round((end-start), ndigits=2)} sec.")
