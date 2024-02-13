#!/usr/bin/env python3
"""
This script computes and saves the subset O2 of observations of
the node colors

Author: Belinda Fleischmann
"""
import os
import time
import pickle
from utilities.config import DataHandler, Paths, humanreadable_time
from utilities.task import TaskSetsNCardinalities, TaskNGridParameters


def main(task_params: TaskNGridParameters):
    """_summary_

    Args:
        task_params (TaskNGridParameters): _description_
    """

    task_sets_n_cardins = TaskSetsNCardinalities(
        task_params=task_params
    )

    data_handler = DataHandler(paths=Paths())

    # ------ Set of observations O2-------------------------------------------
    set_O2_path = data_handler.create_matrix_fn(
        matrix_name="set_O2",
        n_nodes=task_params.n_nodes,
        n_hides=task_params.n_hides)

    if os.path.exists(f"{set_O2_path}.pkl"):
        # Load matrices from hd for this task grid configuration
        print(
            f"{set_O2_path}.pkl already exists",
            )

        with open(f"{set_O2_path}.pkl", "rb") as file:
            O2 = pickle.load(file)
        debug = "here"

    else:
        # Compute for this task grid configuration and save to hd
        print("Computing set O2 for given task config ...")
        # print("Computing set O2 for given task config ("
        #       f"{self.params.n_nodes} nodes and "
        #       f"{self.params.n_hides} hiding spots) ...")
        start = time.time()
        task_sets_n_cardins.compute_O2()
        end = time.time()
        print("Time needed to compute set O2:",
              humanreadable_time(end-start)
              )
        # print(f" ... finished computing S. \n ... time:  "
        #       f"{humanreadable_time(end-start)}\n")
        start = time.time()
        data_handler.save_arrays(
            n_nodes=task_params.n_nodes,
            n_hides=task_params.n_hides,
            set_O2=task_sets_n_cardins.O2
            )
        end = time.time()
        print("Time needed to save set O2 to disk: \n",
              humanreadable_time(end-start))
        # print(f" ... finisehd writing O to disk. \n ... time:  "
        #       f"{humanreadable_time(end-start)}\n"
        #       )
        # print("                 Value/Shape           Size")
        # print("          set O2: %s",
        #                 task_sets_n_cardins.O2.shape)
        # print("                                       %s \n",
        #                 asizeof.asizeof(task_sets_n_cardins.O2))


if __name__ == "__main__":

    # Define experiment / simulation label
    DIM = 2
    HIDES = 1

    # Define task configuration parameters
    task_parameters = TaskNGridParameters(
        dim=DIM,
        n_hides=HIDES,
        n_blocks=1,
        n_rounds=1,
        n_trials=12
        )

    main(task_params=task_parameters)
