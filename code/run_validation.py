"""
This script evaluates and visualizes beh_model recovery simulations for.

Author: Belinda Fleischmann
"""

import time
from utilities.simulation_methods import Simulator
from utilities.modelling import AgentInitObject


def main():
    simulator = Simulator(mode="validation")
    simulator.dir_mgr.create_data_out_dir()
    simulator.taus = [0, 0.4, 0.8, 1.2, 1.6, 2.0]

    for agent_model in simulator.agent_model_space:
        agent_attr = AgentInitObject(agent_model).def_attributes()

        for tau in simulator.taus:
            simulator.tau = tau
            simulator.simulate(agent_attr)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total time for beh_model validation: "
          f"{round((end-start), ndigits=2)} sec.")
