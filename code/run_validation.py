"""
This script evaluates and visualizes beh_model recovery simulations.

Author: Belinda Fleischmann
"""

import os.path
import time
import pickle
import argparse
from utilities.simulation_methods import Simulator
from utilities.modelling import AgentInitObject
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


# TODO: create a validation object class to store all the objects needed for
#  simulation and estimation, e.g. bayesian components, and after simulation
#  the data set etc.


def get_arguments():
    """Get arguments from environment, if script is executed from command line
    or with a bash jobwrapper."""
    parser = argparse.ArgumentParser(description='Run model validation.')
    # type checks that the input type is correct and converts it to int, otherwise it is a string
    # requires makes the subject argument mandatory, otherwise it can be script and is set to None by default
    parser.add_argument('--repetition', type=int, required=True)  #, help='The subject number to process')
    parser.add_argument('--agent_model', type=str, required=True)
    parser.add_argument('--tau_value', type=float, required=True)
    parser.add_argument('--participant', type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    simulator = Simulator(mode="validation")
    simulator.dir_mgr.create_val_out_dir(out_dir_label="tests")  # TODO: sort out where what output
    
    simulator.n_repetitions = 1  # TODO: will be one for one rep, but how to have multiple repetitions?
    simulator.agent_model_space = [args.agent_model]
    simulator.tau_gen = args.tau_value
    simulator.taus_analize = np.linspace(0.25, 2, 10)
    simulator.n_participants = len(args.participant)
    simulator.task_configs.params.n_blocks = 1  # TODO: change to 3

    # FOR QUICK_TESTS
    # simulator.agent_model_space = ["C2", "C3", "A1", "A2", "A3"]
    # simulator.agent_model_space = ["C1", "A1"]
    # simulator.agent_model_space = ["A1"]
    # simulator.n_participants = 1
    # simulator.taus = np.linspace(0.25, 2.5, 2)

    # llh = xr.DataArray(
    #     np.full((n_agents, n_taus, n_partic, n_rep), np.nan),
    #     dims=("agent", "tau", "participant", "repetition"),
    #     coords={"agent": simulator.agent_model_space,
    #             "tau": simulator.taus,
    #             "participant": np.array(range(n_partic)) + 1,
    #             "repetition": np.array(range(n_rep)) + 1
    #             }
    # )

    for repetition in range(simulator.n_repetitions):

        simulator.this_rep = repetition + 1

        mle_group_av_recorder = {"tau_gen": []}
        for agent_i, agent_model in enumerate(simulator.agent_model_space):
            mle_group_av_recorder[f'{agent_model}_mle_group_mean'] = []
            mle_group_av_recorder[f'{agent_model}_mle_group_sd'] = []
            simulator.agent_attr = AgentInitObject(
                agent_model).def_attributes()

            # mle_group_av = {}  # Model- and tau-spec group-av ml estimtes

            for tau in [simulator.tau_gen]:

                # ml_estimates_all_part = np.full(
                #     simulator.n_participants, np.nan)
                if agent_i == 0:
                    mle_group_av_recorder["tau_gen"].append(tau)

                for participant in range(simulator.n_participants):
                    simulator.this_part = participant + 1
                    simulator.tau_gen = tau
                    simulator.simulate_beh()

                    # llh.loc[dict(agent=agent_model, tau=tau,
                    #              participant=participant + 1,
                    #              repetition=repetition + 1)
                    # ] = np.nansum(simulator.data["log_p_a_giv_h"])

                    estimator = ParameterEstimator(simulator)

                    ml_estimate_this_part = estimator.brute_force_est()

                    out_fn = simulator.dir_mgr.define_out_single_val_filename(
                        repetition, agent_model, tau, args.participant
                    )
                    
                    # Save estimate to pickle
                    # with open(f'{out_fn}.pkl', 'wb') as file:
                    #     pickle.dump(ml_estimate_this_part, file)
                    
                    # Save ML estimate to tsv
                    with open(
                        f"{out_fn}.tsv", "w", encoding="utf8") as tsv_file:
                        tsv_file.write(str(ml_estimate_this_part))

                    # TODO: or better as .txt file?
                    # np.savetxt(
                    # "../data/test2.txt", x, fmt="%2.3f", delimiter=",")

                    # ml_estimates_all_part[
                    # participant] = ml_estimate_this_part

                # mle_group_av[tau] = ml_estimates_all_part.mean(axis=0)
                # mle_group_av_recorder[f'{agent_model}_mle_group_mean'].append(
                #    ml_estimates_all_part.mean(axis=0))
                # mle_group_av_recorder[f'{agent_model}_mle_group_sd'].append(
                #    ml_estimates_all_part.std(axis=0))

        # mle_group_av_df = pd.DataFrame(mle_group_av_recorder)
        # with open(f"{simulator.dir_mgr.paths.this_sim_out_dir}/"
        #           f"param_recovery.tsv", "w", encoding="utf8") as \
        #         tsv_file:
        #     tsv_file.write(mle_group_av_df.to_csv(sep="\t", na_rep=np.NaN,
        #                                           index=False))

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
