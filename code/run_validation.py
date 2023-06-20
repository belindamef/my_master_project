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
from utilities.estimation_methods import ParameterEstimator, EstimationParams
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
    parser.add_argument('--lambda_value', type=float, nargs='+')
    parser.add_argument('--participant', type=int, nargs='+')
    args = parser.parse_args()
    return args


def define_simulation_parameters() -> SimulationParameters:
    sim_parameters = SimulationParameters()
    if arguments.parallel_computing:
        sim_parameters.get_params_from_args(arguments)
    else:  # Define parameters for local tests, i.e. not parallel computing
        sim_parameters.agent_space_gen = ["A3"]
        sim_parameters.tau_gen_space = np.linspace(0.1, 2., 5)
        sim_parameters.tau_gen_space_if_fixed = [0.1]
        sim_parameters.lambda_gen_space = np.linspace(0.1, 0.9, 5)
        sim_parameters.n_participants = 10
    return sim_parameters


def define_estimation_parameters() -> EstimationParams:
    estimation_params = EstimationParams()
    return estimation_params


def define_lambda_gen_space(agent_model: str, tau_gen: float):
    if (agent_model == "A3"):  # and
        # tau_gen in sim_params.tau_gen_space_if_fixed):
        sim_params.lambda_gen_space = np.linspace(0.1, 0.9, 5)
    else:
        sim_params.lambda_gen_space = [0.5]


def main():
    dir_mgr = DirectoryManager()
    dir_mgr.create_val_out_dir(out_dir_label=OUT_DIR_LABEL, version=VERSION_NO)
    task_configs = TaskConfigurator(dir_mgr.paths).get_config(
        config_label=sim_params.task_config_label)
    bayesian_comps = BayesianModelComps(task_configs.params).get_comps()
    simulator = Simulator(task_configs=task_configs,
                          bayesian_comps=bayesian_comps)

    simulator.sim_params = sim_params  # link sim_params to simulator instance

    task_configs.params.n_blocks = 1  # TODO: for testing only
    # task_configs.params.n_rounds = 2
    # task_configs.params.n_trials = 2

    for repetition in sim_params.repetition_numbers:
        sim_params.current_rep = repetition + 1

        for agent_model in sim_params.agent_space_gen:
            sim_params.current_agent_attributes = AgentInitObject(
                agent_model).def_attributes()

            for tau_gen in sim_params.tau_gen_space:
                sim_params.current_tau_gen = tau_gen

                # define_lambda_gen_space(agent_model, tau_gen)

                for lambda_gen in sim_params.lambda_gen_space:
                    mle_recorder = {
                        "agent": [], "participant": [],
                        "tau_gen": [], "tau_mle": [],
                        "lambda_gen": [], "lambda_mle": []}

                    sim_params.current_lambda_gen = lambda_gen
                    mle_recorder["lambda_gen"].extend(
                        [lambda_gen for part in sim_params.participant_numbers]
                    )

                    mle_recorder["agent"].extend(
                        [agent_model for part in sim_params.participant_numbers
                         ])
                    mle_recorder["tau_gen"].extend(
                        [tau_gen for part in sim_params.participant_numbers]
                        )

                    for participant in sim_params.participant_numbers:
                        sim_params.current_part = participant + 1
                        mle_recorder["participant"].append(participant + 1)

                        simulator.simulate_beh_data()

                        estimator = ParameterEstimator()

                        estimator.instantiate_sim_obj(
                            exp_data=simulator.data,
                            task_configs=task_configs,
                            bayesian_comps=bayesian_comps
                            )

                        # Embed simulation params in estimator sim object
                        estimator.sim_object.sim_params = sim_params
                        # TODO: leave this or change, ask Dirk
                        estimator.est_params.current_lambda_analyze = \
                            sim_params.current_lambda_gen

                        # Estimate tau
                        print("Starting brute-force estimation for tau",
                              f"tau_gen: {tau_gen}")
                        mle_tau_estimate = estimator.estimate_tau(
                            method="brute_force")
                        mle_recorder["tau_mle"].append(mle_tau_estimate)

                        # Estimate lambda, if appliclabe
                        if np.isnan(lambda_gen):
                            mle_recorder["lambda_mle"].append(np.nan)
                        else:
                            mle_lambda_estimate = estimator.estimate_lambda(
                                method="brute_force"
                            )
                            mle_recorder["lambda_mle"].append(
                                mle_lambda_estimate)

                        out_fn = dir_mgr.define_out_single_val_filename(
                                repetition,
                                agent_model,
                                tau_gen,
                                lambda_gen,
                                participant)

                        mle_df = pd.DataFrame(mle_recorder)

                        with open(f"{out_fn}.tsv", "w",
                                  encoding="utf8") as tsv_file:
                            tsv_file.write(mle_df.to_csv(
                                sep="\t", na_rep=np.NaN, index=False))


if __name__ == "__main__":
    start = time.time()
    arguments = get_arguments()
    sim_params = define_simulation_parameters()
    sim_params.task_config_label = "exp_msc"
    OUT_DIR_LABEL = "tests"
    VERSION_NO = 2
    est_params = define_estimation_parameters()
    main()
    end = time.time()
    print(f"Total time for beh_model validation: "
          f"{round((end-start), ndigits=2)} sec.")
