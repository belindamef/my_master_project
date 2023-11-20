""" testing methods to create observation sets"""

from encodings import utf_8
from itertools import product
import more_itertools
import csv
import numpy as np


def print_combos():
    """_summary_"""
    # Define the possible values and their maximum counts
    values = [0, 1, 2]
    max_counts = [4, 2, 2]

    # Generate all possible permutations with repetitions
    all_permutations = product(*[range(count + 1) for count in max_counts])

    # Filter out permutations that exceed the maximum counts
    valid_permutations = [perm for perm in all_permutations if sum(perm) == 4]

    # Print the valid permutations
    for perm in valid_permutations:
        vector = []

        for val, count in zip(values, perm):
            vector.extend([val] * count)

        vector.extend([0] * (4 - len(vector)))

        print(vector)


def compute_obs_perms(vector_size, n_hides):
    """Method to evaluate permutations of observations"""

    obs_values = [0] * vector_size
    obs_values.extend([1] * (vector_size - n_hides))
    obs_values.extend([2] * n_hides)

    eg = [0,0,0,0,1,1,2,2]

    node_color_set_perms = sorted(
        more_itertools.distinct_permutations(obs_values, r=vector_size))
    print(node_color_set_perms)


    # TODO: how to compute number of permutation * 4 * 2

    O = np.full((4 * 2 * len(node_color_set_perms), 6), np.nan)
    i = 0

    for position in range(vector_size):
        for treasure_found in [0, 1]:
            
            for color in node_color_set_perms:

                # TODO: write observations
                O[i, 0] = position
                O[i, 1] = treasure_found

                O[i, 2:6] = color

                i += 1
    
    # TODO: linearer index

    # Define the output file name

    output_file = f'/home/belindame_f/treasure-hunt/code/utilities/obs_{vector_size}-nodes_{n_hides}-hides.csv'

    # Write the vectors to the TSV file
    with open(output_file, 'w', newline='', encoding="utf8") as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(O)

    print(f"Permutations written to {output_file}")


if __name__ == "__main__":
    N_NODES = 4
    N_HIDES = 2

    compute_obs_perms(vector_size=N_NODES, n_hides=N_HIDES)
