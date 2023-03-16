"""
This script evaluates and visualizes model recovery simulations for.

Author: Belinda Fleischmann
"""

from dataclasses import dataclass, is_dataclass
import time
import run_simulation
from utilities.simulation_methods import DefaultSimulationParameters, Recorder



@dataclass
class ValidationParameter:
    simulation_parameters = DefaultSimulationParameters(n_repetitions=50)
    pi = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

if __name__ == "__main__":
    validation_parameters = ValidationParameter()

    start = time.time()
    run_simulation.main(mode="validation",
                        validation_parameters=validation_parameters)
    end = time.time()
    print(f"Total time for model validation: "
          f"{round((end-start), ndigits=2)} sec.")
