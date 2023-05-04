"""
This script evaluates and visualizes beh_model recovery simulations.

Author: Belinda Fleischmann
"""

import os.path
import time
import pickle
import argparse
from utilities.config import DirectoryManager, TaskConfigurator
from utilities.simulation_methods import Simulator, CurrentParameters, SimulationParameters
from utilities.modelling import AgentInitObject, BayesianModelComps
from utilities.estimation_methods import ParameterEstimator
import numpy as np
import pandas as pd
# import xarray as xr


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
    parser.add_argument('--repetition', type=int, default=range(10), nargs='+')
    parser.add_argument('--agent_model', type=str, nargs='+',
                        default=["C1", "C2", "C3", "A1", "A2", "A3"])
    parser.add_argument('--tau_value', type=float, nargs='+',
                        default=np.linspace(0.25, 2, 10))
    parser.add_argument('--participant', type=int, nargs='+',
                        default=range(1))
    args = parser.parse_args()
    return args


def sort_sim_params(args) -> SimulationParameters:
    sim_params = SimulationParameters()
    sim_params.n_repetitions = args.repetition
    sim_params.agents_space_gen = args.agent_model
    sim_params.tau_space_gen = args.tau_value
    return sim_params


def main(args):
    dir_mgr = DirectoryManager()
    dir_mgr.create_val_out_dir(out_dir_label="tests")
    task_configs = TaskConfigurator(dir_mgr.paths).get_config(
        config_label="exp_msc")
    bayesian_comps = BayesianModelComps(task_configs.task_params).get_comps()
    simulator = Simulator(mode="validation",
                          task_configs=task_configs,
                          bayesian_comps=bayesian_comps)

    current_params = CurrentParameters()
    current_params.this_rep = args.repetition
    current_params.agent_attr = AgentInitObject(
        args.agent_model).def_attributes()
    current_params.this_tau_gen = args.tau_value
    current_params.this_part = args.participant
    simulator.current_params = current_params

    simulator.tau_gen = args.tau_value
    n_participants = 1  # TODO: find better way to automatize loop
    n_repetitions = 1
    simulator.task_configs.task_params.n_blocks = 1

    for repetition in range(n_repetitions):

        simulator.this_rep = repetition + 1

        mle_recorder = {"agent": [], "tau_gen": [], "participant": [],
                        "mle": []}
        for agent_model in [args.agent_model]:
            mle_recorder["agent"].append(agent_model)
            simulator.agent_attr = AgentInitObject(
                agent_model).def_attributes()

            for tau_gen in [simulator.tau_gen]:
                mle_recorder["tau_gen"].append(tau_gen)

                for this_part in range(n_participants):
                    mle_recorder["participant"].append(this_part + 1)
                    simulator.this_part = this_part + 1
                    simulator.tau_gen = tau_gen
                    simulator.simulate_beh()

                    estimator = ParameterEstimator(simulator)

                    mle_recorder["mle"].append(estimator.brute_force_est())

        mle_group_av_df = pd.DataFrame(mle_recorder)
        out_fn = dir_mgr.define_out_single_val_filename(
                args.repetition, args.agent_model, args.tau_value,
                args.participant)

        with open(f"{out_fn}.tsv", "w", encoding="utf8") as tsv_file:
            tsv_file.write(mle_group_av_df.to_csv(sep="\t", na_rep=np.NaN,
                                                  index=False))

            # Save estimate to pickle
            # with open(f'{out_fn}.pkl', 'wb') as file:
            #     pickle.dump(ml_estimate_this_part, file)

            # Save ML estimate to tsv
            # with open(
            #     f"{out_fn}.tsv", "w", encoding="utf8") as tsv_file:
            #     tsv_file.write(str(ml_estimate_this_part))

            # np.savetxt(
            # "../data/test2.txt", x, fmt="%2.3f", delimiter=",")

    # save_llh(llh, simulator.dir_mgr.paths.this_sim_out_dir)


if __name__ == "__main__":
    start = time.time()
    arguments = get_arguments()
    print(f"Starting validation "
          f"repetition {arguments.repetition} "
          f"for agent {arguments.agent_model}, "
          f"with generating tau: {arguments.tau_value}, "
          f"participant {arguments.participant} ")
    main(args=arguments)
    end = time.time()
    print(f"Total time for beh_model validation: "
          f"{round((end-start), ndigits=2)} sec.")

    # llh = xr.DataArray(
    #     np.full((n_agents, n_taus, n_partic, n_rep), np.nan),
    #     dims=("agent", "tau", "participant", "repetition"),
    #     coords={"agent": simulator.agent_model_space,
    #             "tau": simulator.taus,
    #             "participant": np.array(range(n_partic)) + 1,
    #             "repetition": np.array(range(n_rep)) + 1
    #             }
    # )
