""" testing methods to create observation sets"""
import os
import more_itertools
import csv
import numpy as np
import time
  from config import humanreadable_time
from matplotlib import cm, colorbar, pyplot
from config import DirectoryManager
from very_plotter_new import VeryPlotter
from matplotlib.colors import ListedColormap


def compute_set_S(n_nodes, n_hides) -> np.ndarray:
    """Method to compute the set of states"""

    hide_combos = sorted(
        more_itertools.distinct_combinations(
            range(1, n_nodes + 1), r=2
            )
    )

    tr_possibility_factor = n_hides / n_nodes
    cardinality_S = int(
        n_nodes * n_nodes * len(hide_combos) * tr_possibility_factor)

    S = np.full((cardinality_S,
                 1 + 1 + n_hides),
                np.nan)

    i = 0
    for possible_position in range(n_nodes):
        possible_position += 1
        for possible_tr_loc in range(n_nodes):
            possible_tr_loc += 1
            for hiding_spot_combo in hide_combos:
                if possible_tr_loc in hiding_spot_combo:
                    S[i, 0] = possible_position
                    S[i, 1] = possible_tr_loc
                    S[i, 2:4] = hiding_spot_combo
                    i += 1

    return S


def compute_node_color_combos(n_nodes, n_hides) -> list:
    """Method to compute the node color components of observation set, i.e.
    the last 4 entries of each set element's vector"""

    obs_values = [0] * n_nodes
    obs_values.extend([1] * (n_nodes - n_hides))
    obs_values.extend([2] * n_hides)

    node_color_combos = sorted(
        more_itertools.distinct_permutations(obs_values, r=n_nodes))

    return node_color_combos


def compute_set_O(n_nodes, n_hides) -> np.ndarray:
    """Method to compute complete set of Observations"""

    node_color_combos = compute_node_color_combos(n_nodes=n_nodes,
                                                  n_hides=n_hides)

    O = np.full((2 * len(node_color_combos), n_nodes + 1), np.nan)
    i = 0

    for treasure_found in [0, 1]:
        
        for color_combo in node_color_combos:

            # TODO: write observations
            O[i, 0] = treasure_found

            O[i, 1:] = color_combo

            i += 1
    # TODO: linearer index

    return O


def compute_Omega(S, O, A) -> np.ndarray:
    """Method to compute """

    n = len(S)
    m = len(O)
    p = len(A)

    Omega = np.full((p, n, m), 0)

    node_colors = {"black": 0,
                   "grey": 1,
                   "blue": 2}

    for i_a, a in enumerate(A):
        for i_s, s in enumerate(S):
            for i_o, o in enumerate(O):
                # Extract state components
                current_pos = int(s[0])  # NOTE: set S[0] := {1, ..., n}
                tr_location = int(s[1])
                hiding_spots = s[2:]

                # Extract observation components
                tr_flag = o[0]

                # TODO: alle observations, wo node colors mit hiding spot
                # siehe Handwritten NOTE 24.11.

                #  locations keinen sinn machen
                # -------After DRILL actions: ---------------------------------
                if a == 0:

                    # CONDITION:                        CORRESP MODEL VARIABLE:
                    # ---------------------------------------------------------
                    # if new position...                                s[1]
                    # ...IS NOT treasure location                       s[2]
                    # ...IS NOT hiding spot,                            s[3:]
                    # all observation, for which...
                    # ...tr_flag == 0,                                  o[1]
                    # ...and new node color == grey,                    o[2:]
                    #  = 1
                    if (
                            current_pos != tr_location
                            and current_pos not in hiding_spots
                            and tr_flag == 0
                            and o[current_pos] == node_colors["grey"]):
                        Omega[i_a, i_s, i_o] = 1

                    # CONDITION:                        CORRESP MODEL VARIABLE:
                    # ---------------------------------------------------------
                    # if new position...                                s[1]
                    # ...IS NOT treasure location                       s[2]
                    # ...IS hiding spot,                                s[3:]
                    # all observation, for which...
                    # ...tr_flag == 0,                                  o[1]
                    # ...and new node color == blue,                    o[2:]
                    #  = 1
                    if (
                            current_pos != tr_location
                            and current_pos in hiding_spots
                            and tr_flag == 0
                            and o[current_pos] == node_colors["blue"]):
                        Omega[i_a, i_s, i_o] = 1

                    # All other observaton probabilites remain 0 as initiated.

                # -------After STEP actions: -----------------------------
                else:  # if a != 0
                    # CONDITION:                        CORRESP MODEL VARIABLE:
                    # ---------------------------------------------------------
                    # if new position...                                s[1]
                    # ...IS NOT treasure location                       s[2]
                    # ...IS NOT hiding spot,                            s[3:]
                    # all observation, for which...
                    # ...tr_flag == 0,                                  o[1]
                    # ...and new node color in ["black", "grey"],       o[2:]
                    #  = 1
                    if (
                            current_pos != tr_location
                            and current_pos not in hiding_spots
                            and tr_flag == 0
                            and o[current_pos] in [node_colors["black"],
                                                   node_colors["grey"]]):
                        Omega[i_a, i_s, i_o] = 1

                    # CONDITION:                        CORRESP MODEL VARIABLE:
                    # ---------------------------------------------------------
                    # if new position...                                s[1]
                    # ...IS NOT treasure location                       s[2]
                    # ...IS hiding spot,                                s[3:]
                    # all observation, for which...
                    # ...tr_flag == 0,                                  o[1]
                    # ...and new node color in ["black", "blue"],       o[2:]
                    #  = 1
                    if (
                            current_pos != tr_location
                            and current_pos in hiding_spots
                            and tr_flag == 0
                            and o[current_pos] in [node_colors["black"],
                                                   node_colors["blue"]]):
                        Omega[i_a, i_s, i_o] = 1

                    # CONDITION:                        CORRESP MODEL VARIABLE:
                    # ---------------------------------------------------------
                    # if new position...                                s[1]
                    # ...IS treasure location                           s[2]
                    # ...IS hiding spot,                                s[3:]
                    # all observation, for which...
                    # ...tr_flag == 1,                                  o[1]
                    # ...and new node color in ["black", "blue"],       o[2:]
                    #  = 1
                    if (
                            current_pos == tr_location
                            and current_pos in hiding_spots
                            and tr_flag == 1
                            and o[current_pos] in [node_colors["black"],
                                                   node_colors["blue"]]):
                        Omega[i_a, i_s, i_o] = 1

    return Omega


def save_arrays(n_nodes, n_hides, **arrays):

    for key, array in arrays.items():

        # Define the output file name
        out_fn = ("/home/belindame_f/treasure-hunt/code/utilities/"
                  + f"{key}-{n_nodes}-nodes_{n_hides}-hides.csv")

        # Write the vectors to the TSV file
        with open(out_fn, 'w', newline='', encoding="utf8") as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(array)


def plot_color_map(n_nodes, n_hides, **arrays):

    dir_mgr = DirectoryManager()

    for key, array in arrays.items():
        fig_fn = f"{key}-{n_nodes}-nodes_{n_hides}-hides"

        # Preapre figure
        plotter = VeryPlotter(paths=dir_mgr.paths)
        plt = pyplot

        rc_params = plotter.define_run_commands()
        plt = pyplot
        plt.rcParams.update(rc_params)
        fig, ax = plt.subplots(figsize=(11, 5))

        # Create a custom discrete colormap
        cmap = ListedColormap(['darkgrey', 'darkcyan'])
        image = ax.matshow(array, cmap=cmap)

        # Add colorbar
        cbar = plt.colorbar(image, ticks=[0, 1], shrink=0.4)

        # Save or display the plot
        plotter.save_figure(fig=fig, figure_filename=fig_fn)


def compute_beta_1_0(n_nodes, set_S):

    n_S = len(set_S)
    beta_1_0 = np.full((n_S, 1), 1 / n_S)

    return beta_1_0


if __name__ == "__main__":
    DIM = 2
    N_HIDES = 2
    N_NODES = DIM**2

    dim = np.sqrt(N_NODES)

    print("\n Computing sets O, S and A")
    start = time.time()
    O = compute_set_O(n_nodes=N_NODES, n_hides=N_HIDES)
    S = compute_set_S(n_nodes=N_NODES, n_hides=N_HIDES)
    # A = [0, -dim, +1, dim, -1]
    A = [0, 1]  # 0: drill, 1: step
    end = time.time()
    print(f"\n ... finished computung sets O, S and A, \n ... time:  "
          f"{humanreadable_time(end-start)}"
          )

    print("\n Computing Omega...")
    start = time.time()
    Omega = compute_Omega(S=S, O=O, A=A)
    end = time.time()
    print(f"\n ... finished computung Omega, \n ... time:  "
          f"{humanreadable_time(end-start)}"
          )

    print("\n Writing files to disk...")
    start = time.time()
    save_arrays(n_nodes=N_NODES, n_hides=N_HIDES,
                set_S=S, set_O=O,
                Omega_drill=Omega[0],
                Omega_step=Omega[1],
                # Omega_3=Omega[2],
                # Omega_4=Omega[3],
                # Omega_5=Omega[4]
                )
    end = time.time()
    print(f"\n ...finished writing arrays to disk, \n ... time:  "
          f"{humanreadable_time(end-start)}"
          )

    plot_color_map(n_nodes=N_NODES, n_hides=N_HIDES,
                   Omega_drill=Omega[0],
                   Omega_step=Omega[1])
