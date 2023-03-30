"""
This script evaluates and visualizes beh_model recovery simulations for.

Author: Belinda Fleischmann
"""
import os.path
import sys
import time

import pandas as pd

from utilities.simulation_methods import Simulator
from utilities.modelling import AgentInitObject
from utilities.estimation_methods import ParameterEstimator
import numpy as np
#import xarray as xr
import pickle
import argparse
import click




def save_llh(llh_array, sim_out_path):
    """Save agent, tau and participant specific sum log likelihood

    Parameters
    ----------
    llh_array: DataArray"""
    filename = os.path.join(sim_out_path, "llh")
    with open(f'{filename}.pkl', 'wb') as handle:
        pickle.dump(llh_array, handle, protocol=pickle.HIGHEST_PROTOCOL)


# TODO: create a validation object class to store all the objects needed for
#  simulation and estimation, e.g. bayesian components, and after simulation
#  the data set etc.

@click.command()
@click.option('--subject', type=int, required=True, help='The subject number to process.')
@click.option('--tau', type=int, required=True, help='This is my tau.')
def main(subject, tau, out_dir_label="test"):
    simulator = Simulator(mode="validation")
    simulator.dir_mgr.create_data_out_dir(out_dir_label)
    #simulator.taus = np.arange(0.5, 2.5, 0.1)
    simulator.taus = np.linspace(0.25,2,10)
    simulator.n_participants =  10
    simulator.agent_model_space = ["C2", "C3", "A1", "A2", "A3"]
    simulator.task_configs.params.n_blocks = 1
    #simulator.agent_model_space = ["A1"]

    # FOR QUICK_TESTS
    # simulator.agent_model_space = ["C1", "A1"]
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
        # TODO: >1 repetitions only needed with minimize(?)
        simulator.this_rep = repetition + 1

        mle_group_av_recorder = {"tau_gen": []}
        for agent_i, agent_model in enumerate(simulator.agent_model_space):
            mle_group_av_recorder[f'{agent_model}_mle_group_mean'] = []
            mle_group_av_recorder[f'{agent_model}_mle_group_sd'] = []
            simulator.agent_attr = AgentInitObject(agent_model).def_attributes()

            mle_group_av = {}  # Model- and tau-spec group-av ml estimtes

            for tau_i, tau in enumerate(simulator.taus):

                ml_estimates_all_part = np.full(simulator.n_participants, np.nan)
                if agent_i == 0:
                    mle_group_av_recorder["tau_gen"].append(tau)

                for participant in range(simulator.n_participants):
                    simulator.this_part = participant + 1
                    simulator.tau = tau
                    simulator.simulate_beh()

                    # llh.loc[dict(agent=agent_model, tau=tau,
                    #              participant=participant + 1,
                    #              repetition=repetition + 1)
                    # ] = np.nansum(simulator.data["log_p_a_giv_h"])

                    estimator = ParameterEstimator(simulator)

                    ml_estimate_this_part = estimator.brute_force_est()

                    ml_estimates_all_part[participant] = ml_estimate_this_part

                mle_group_av[tau] = ml_estimates_all_part.mean(axis=0)
                mle_group_av_recorder[f'{agent_model}_mle_group_mean'].append(
                    ml_estimates_all_part.mean(axis=0))
                mle_group_av_recorder[f'{agent_model}_mle_group_sd'].append(
                    ml_estimates_all_part.std(axis=0))



        mle_group_av_df = pd.DataFrame(mle_group_av_recorder)
        with open(f"{simulator.dir_mgr.paths.this_sim_out_dir}/"
                  f"param_recovery.tsv", "w", encoding="utf8") as \
                tsv_file:
            tsv_file.write(mle_group_av_df.to_csv(sep="\t", na_rep=np.NaN,
                                                  index=False))


    #save_llh(llh, simulator.dir_mgr.paths.this_sim_out_dir)

def get_arguments():
    #parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Run my analysis.')
    # type checks that the input type is correct and converts it to int, otherwise it is a string
    # requires makes the subject argument mandatory, otherwise it can be script and is set to None by default
    parser.add_argument('--subject', type=int, required=True, help='The subject number to process')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    start = time.time()
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
    end = time.time()
    print(f"Total time for beh_model validation: "
          f"{round((end-start), ndigits=2)} sec.")
