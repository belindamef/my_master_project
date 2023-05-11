"""
This script evaluates and visualizes beh_model recovery simulations.

Author: Belinda Fleischmann
"""

import os.path
import time
import pickle
import argparse
from utilities.config import DirectoryManager, TaskConfigurator
from utilities.simulation_methods import Simulator, SimulationParameters
from utilities.modelling import AgentInitObject, BayesianModelComps
from utilities.estimation_methods import ParameterEstimator
import numpy as np
import pandas as pd


def save_llh(llh_array, sim_out_path):
    """Save agent, tau and participant specific sum log likelihood.
    Parameters
    ----------
    llh_array: DataArray"""
    filename = os.path.join(sim_out_path, "llh")
    with open(f'{filename}.pkl', 'wb') as handle:
        pickle.dump(llh_array, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_arguments():
    """Get arguments from environment, if script is executed from command line
    or with a bash jobwrapper."""
    parser = argparse.ArgumentParser(description='Run model validation.')
    parser.add_argument('--parallel_computing', action="store_true")
    parser.add_argument('--repetition', type=int, nargs='+')
    parser.add_argument('--agent_model', type=str, nargs='+')
    parser.add_argument('--tau_value', type=float, nargs='+')
    parser.add_argument('--participant', type=int, nargs='+')
    args = parser.parse_args()
    return args


def define_simulation_parameters() -> SimulationParameters:
    sim_parameters = SimulationParameters()
    if arguments.parallel_computing:
        sim_parameters.get_params_from_args(arguments)
    return sim_parameters


def main():
    dir_mgr = DirectoryManager()
    dir_mgr.create_val_out_dir(out_dir_label=sim_params.out_dir_label)
    task_configs = TaskConfigurator(dir_mgr.paths).get_config(
        config_label=sim_params.task_config_label)
    bayesian_comps = BayesianModelComps(task_configs.params).get_comps()
    simulator = Simulator(task_configs=task_configs,
                          bayesian_comps=bayesian_comps)

    simulator.sim_params = sim_params  # link sim_params to simulation instance

    task_configs.params.n_blocks = 1  # TODO: for testing only

    for repetition in sim_params.repetition_numbers:

        sim_params.current_rep = repetition + 1

        mle_recorder = {
            "agent": [], "tau_gen": [], "participant": [], "mle": []}

        for agent_model in sim_params.agent_space_gen:
            sim_params.current_agent_attributes = AgentInitObject(
                agent_model).def_attributes()
            mle_recorder["agent"].extend(
                [agent_model for tau in sim_params.tau_space_gen])

            for tau_gen in sim_params.tau_space_gen:
                sim_params.current_tau_gen = tau_gen
                mle_recorder["tau_gen"].extend(
                    [tau_gen for partic in sim_params.participant_numbers]
                    )

                for participant in sim_params.participant_numbers:
                    sim_params.current_part = participant + 1
                    mle_recorder["participant"].append(participant + 1)

                    simulator.simulate_beh_data()

                    estimator = ParameterEstimator(
                        exp_data=simulator.data,
                        task_configs=task_configs,
                        bayesian_comps=bayesian_comps
                        )
                    estimator.sim_object.sim_params = sim_params
                    mle_estimate = estimator.estimate_tau(method="brute_force")

                    mle_recorder["mle"].append(mle_estimate)
                    # mle_recorder["mle"].append(estimator.brute_force_est())

        mle_group_av_df = pd.DataFrame(mle_recorder)
        out_fn = dir_mgr.define_out_single_val_filename(
                repetition,
                agent_model,
                tau_gen,
                participant)

        with open(f"{out_fn}.tsv", "w", encoding="utf8") as tsv_file:
            tsv_file.write(mle_group_av_df.to_csv(sep="\t", na_rep=np.NaN,
                                                  index=False))


if __name__ == "__main__":
    start = time.time()
    arguments = get_arguments()
    sim_params = define_simulation_parameters()
    sim_params.task_config_label = "exp_msc"
    sim_params.out_dir_label = "tests"
    main()
    end = time.time()
    print(f"Total time for beh_model validation: "
          f"{round((end-start), ndigits=2)} sec.")
