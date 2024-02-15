#!/usr/bin/env python3
"""
This script computes and saves the subset O2 of observations of
the node colors

Author: Belinda Fleischmann
"""
import os
import time
import pickle
import more_itertools
import numpy as np
import scipy.sparse as sp
from pympler import asizeof
from bitarray import bitarray
from utilities.config import DataHandler, Paths, humanreadable_time
from utilities.task import TaskSetsNCardinalities, TaskNGridParameters


def heaps_algorithm_partial(iterable, n_nodes):
    """
    Generate permutations of a subset of an iterable using Heap's Algorithm.
    """
    def generate(k, array):
        if k == 1:
            yield list(array[:n_nodes])
        else:
            for i in range(k):
                yield from generate(k - 1, array)
                j = 0 if k % 2 == 1 else i
                array[j], array[k-1] = array[k-1], array[j]

    yield from generate(len(iterable), list(iterable))


def heaps_algorithm_distinct(iterable, n_nodes):
    """
    Generate distinct permutations of a subset of an iterable using Heap's Algorithm.
    """
    def generate(k, array):
        if k == 1:
            yield array[:n_nodes]
        else:
            for i in range(k):
                yield from generate(k - 1, array)
                j = 0 if k % 2 == 1 else i
                array[j], array[k-1] = array[k-1], array[j]
                if tuple(array[:n_nodes]) not in seen:
                    seen.add(tuple(array[:n_nodes]))
                    yield array[:n_nodes]

    seen = set()  # Initialize set to store seen permutations
    yield from generate(len(iterable), list(iterable))


def distinct_permutations(iterable, n_nodes):
    """
    Generate distinct permutations of a subset of an iterable.
    """
    def generate(permutation):
        if len(permutation) == n_nodes:
            yield permutation
        else:
            seen = set()  # Set to keep track of permutations already generated
            for i, element in enumerate(iterable):
                if element not in seen:
                    seen.add(element)
                    yield from generate(permutation + [element])

    yield from generate([])


def next_permutation(arr, permutations):
    # Find the largest index k such that arr[k] < arr[k + 1]
    k = len(arr) - 2
    while k >= 0 and arr[k] >= arr[k + 1]:
        k -= 1
    if k == -1:
        return False  # If no such index exists, this is the last permutation

    # Find the largest index l greater than k such that arr[k] < arr[l]
    l = len(arr) - 1
    while arr[k] >= arr[l]:
        l -= 1

    # Swap arr[k] and arr[l]
    arr[k], arr[l] = arr[l], arr[k]

    # Reverse the sequence from arr[k + 1] up to and including the final element
    arr[k + 1:] = reversed(arr[k + 1:])

    permutations.append(arr[:])  # Store the current permutation
    return True


# Algorithm from more_itertools
def _partial(A, r):
    # Split A into the first r items and the last r items
    head, tail = A[:r], A[r:]
    right_head_indexes = range(r - 1, -1, -1)
    left_tail_indexes = range(len(tail))

    while True:
        # Yield the permutation we have
        yield tuple(head)

        # Starting from the right, find the first index of the head with
        # value smaller than the maximum value of the tail - call it i.
        pivot = tail[-1]
        for i in right_head_indexes:
            if head[i] < pivot:
                break
            pivot = head[i]
        else:
            return

        # Starting from the left, find the first value of the tail
        # with a value greater than head[i] and swap.
        for j in left_tail_indexes:
            if tail[j] > head[i]:
                head[i], tail[j] = tail[j], head[i]
                break
        # If we didn't find one, start from the right and find the first
        # index of the head with a value greater than head[i] and swap.
        else:
            for j in right_head_indexes:
                if head[j] > head[i]:
                    head[i], head[j] = head[j], head[i]
                    break

        # Reverse head[i + 1:] and swap it with tail[:r - (i + 1)]
        tail += head[: i - r : -1]  # head[i + 1:][::-1]
        i += 1
        head[i:], tail[:] = tail[: r - i], tail[r - i :]


def test_loops(task_params: TaskNGridParameters, inner_only=False):

    task_sets_n_cardins = TaskSetsNCardinalities(
        task_params=task_params
    )

    def test_double_loop():
        start_loop = time.time()
        for color in [0, 1, 2]:
            start = time.time()
            for i in range(task_sets_n_cardins.n_O2):

                pass

            end = time.time()
            print("Finished color: ", color, " in ", humanreadable_time(end-start))

        end_loop = time.time()
        print("finished loop in ", humanreadable_time(end_loop-start_loop))

    def test_innerloop_only():
        start_loop = time.time()
        for i in range(task_sets_n_cardins.n_O2):
            pass
        end_loop = time.time()
        print("finished loop in ", humanreadable_time(end_loop-start_loop))

    if inner_only:
        test_innerloop_only()
    else:
        test_double_loop()

def compute_O2_indices(n_nodes, n_hides, n_O2):
    """Method to compute index lists for entries in the set of observation
    """

    colors = [0, 1, 2]

    # Create list of values for the urn model
    urn_values = [0] * n_nodes
    urn_values.extend([1] * (n_nodes - n_hides))
    urn_values.extend([2] * n_hides)

    my_permutations_generator = more_itertools.distinct_permutations(
        iterable=urn_values, r=n_nodes
    )

    # Initialize empty dict of dicts of lists to store indices
    o2_node_specfic_indices = {
        0: {},
        1: {},
        2: {}
    }
    for color in colors:
        for node in range(n_nodes):
            o2_node_specfic_indices[color][node] = []

    # Get Indices
    for j in range(n_O2):

        this_permutation = next(my_permutations_generator)

        for node, entry in enumerate(this_permutation):
            for color in colors:
                if entry == color:
                    o2_node_specfic_indices[color][node].append(j)

    return o2_node_specfic_indices


def test_indices(test_node, n_nodes, n_hides, indices, color):
    """Function to test the indices. Only with small dimensions!

    Args:
        test_nodes (int): node number to be tested
        n_nodes (int): _description_
        n_hides (int): _description_
    """
    # Create list of permutations to test indices
    urn_values = [0] * n_nodes  # TODO: hier weiter, Permutationen berechnen
    urn_values.extend([1] * (n_nodes - n_hides))
    urn_values.extend([2] * n_hides)
    O2 = np.array(sorted(
        more_itertools.distinct_permutations(
            iterable=urn_values,  # All values in the Urn model
            r=n_nodes             # Number of items to be sampled
            )
        ))

    print(f"Permutations, in which node {test_node + 1} is {color}: \n",
          O2[indices[test_node],])


def Omega_a_j_generator(task_states_n_cardins: TaskSetsNCardinalities,
                        a, o):

    Omega_a_j = sp.csr_matrix(
        (task_states_n_cardins.n, 1),
        dtype=np.int8)

    node_colors = {"black": 0,
                   "grey": 1,
                   "blue": 2}

    # Initiate row and columns index lists to construct sparse matrices
    rows = []
    cols = []

    for i_s, s in enumerate(task_states_n_cardins.S):

        # TODO: alle observations noch drin,
        #  wo node colors mit hiding spot
        #  locations keinen sinn machen
        # siehe Handwritten NOTE 24.11.

        cur_pos_i = s[0] - 1

        # -------After DRILL actions: -----------------------------
        if a == 0:

            # CONDITION:                    CORRESP MODEL VARIABLE:
            # ---------------------------------------------------------
            # if new position...                              s[1]
            # ...IS NOT treasure location                     s[2]
            # ...IS NOT hiding spot,                          s[3:]
            # all observation, for which...
            # ...tr_flag == 0,                                o[1]
            # ...and new node color == grey,                  o[2:]
            #  = 1
            if (
                    s[0] != s[1]
                    and s[0] not in s[2:]
                    and o[0] == 0
                    and (o[s[0]] == node_colors["grey"])
                            ):
                rows.append(i_s)
                cols.append(0)

            # CONDITION:                    CORRESP MODEL VARIABLE:
            # ---------------------------------------------------------
            # if new position...                              s[1]
            # ...IS NOT treasure location                     s[2]
            # ...IS hiding spot,                              s[3:]
            # all observation, for which...
            # ...tr_flag == 0,                                o[1]
            # ...and new node color == blue,                  o[2:]
            #  = 1
            if (
                    s[0] != s[1]  # not necessary because O sliced
                    and s[0] in s[2:]
                    and o[0] == 0
                    and o[s[0]] == node_colors["blue"]
                    ):
                rows.append(i_s)
                cols.append(0)
            # All other observaton probabs remain 0 as initiated.

        # -------After STEP actions: -----------------------------
        else:  # if a == "step"

            # CONDITION:                    CORRESP MODEL VARIABLE:
            # ---------------------------------------------------------
            # if new position...                              s[1]
            # ...IS NOT treasure location                     s[2]
            # ...IS NOT hiding spot,                          s[3:]
            # all observation, for which...
            # ...tr_flag == 0,                                o[1]
            # ...and new node color in ["black", "grey"],     o[2:]
            #  = 1
            if (
                    s[0] != s[1]
                    and s[0] not in s[2:]
                    and o[0] == 0
                    and (o[s[0]] in [node_colors["black"],
                                     node_colors["grey"]])
                        ):
                rows.append(i_s)
                cols.append(0)

            # CONDITION:                    CORRESP MODEL VARIABLE:
            # ---------------------------------------------------------
            # if new position...                              s[1]
            # ...IS NOT treasure location                     s[2]
            # ...IS hiding spot,                              s[3:]
            # all observation, for which...
            # ...tr_flag == 0,                                o[1]
            # ...and new node color in ["black", "blue"],     o[2:]
            #  = 1
            if (
                    s[0] != s[1]  # not necessary because O sliced
                    and s[0] in s[2:]
                    and o[0] == 0
                    and o[s[0]] == node_colors["blue"]
                        ):
                rows.append(i_s)
                cols.append(0)

            # CONDITION:                    CORRESP MODEL VARIABLE:
            # ---------------------------------------------------------
            # if new position...                              s[1]
            # ...IS treasure location                         s[2]
            # ...IS hiding spot,                              s[3:]
            # all observation, for which...
            # ...tr_flag == 1,                                o[1]
            # ...and new node color in ["black", "blue"],     o[2:]
            #  = 1
            if (
                    s[0] == s[1]  # not necessary because O sliced
                    and s[0] in s[2:]
                    and o[0] == 1
                    and o[s[0]] in [node_colors["black"], node_colors["blue"]]
                        ):
                rows.append(i_s)
                cols.append(0)

    # TODO: find out, why if shape not specified, sparse Omega matrix
                    # turns out as a 45 x 63 matrix ??
    Omega_a_j = sp.csc_matrix(
        ([1]*len(rows),  # data
            (rows, cols)),  # indices
        shape=(task_states_n_cardins.n, 1),  # shape
        dtype=np.int8
        )

    return Omega_a_j  # TODO über yield speichern ?


def test_O_indices_computations(task_params: TaskNGridParameters):
    """_summary_

    Args:
        task_params (TaskNGridParameters): _description_
    """

    task_sets_n_cardins = TaskSetsNCardinalities(
        task_params=task_params
    )

    data_handler = DataHandler(paths=Paths())

    # ------ Set of observations O2-------------------------------------------
    O2_indices_path = data_handler.create_matrix_fn(
        matrix_name="O2_indices",
        n_nodes=task_params.n_nodes,
        n_hides=task_params.n_hides)

    if os.path.exists(f"{O2_indices_path}.pkl"):
        # Load matrices from hd for this task grid configuration
        print(
            f"{O2_indices_path}.pkl already exists",
            )

        with open(f"{O2_indices_path}.pkl", "rb") as file:
            O2_indices = pickle.load(file)
        debug = "here"

    else:
        # Compute for this task grid configuration and save to hd
        print("Computing set O2 for given task config ...")
        # print("Computing set O2 for given task config ("
        #       f"{self.params.n_nodes} nodes and "
        #       f"{self.params.n_hides} hiding spots) ...")
        start = time.time()
        # task_sets_n_cardins.compute_O2()

        # Example usage:
        indices = compute_O2_indices(
            n_nodes=task_params.n_nodes,
            n_hides=task_params.n_hides,
            n_O2=task_sets_n_cardins.n_O2)

        # Test indices:
        test_indices(test_node=2,
                     n_nodes=task_params.n_nodes,
                     n_hides=task_params.n_hides,
                     indices=indices[0],
                     color="black")

        test_indices(test_node=3,
                     n_nodes=task_params.n_nodes,
                     n_hides=task_params.n_hides,
                     indices=indices[1],
                     color="gray")

        test_indices(test_node=1,
                     n_nodes=task_params.n_nodes,
                     n_hides=task_params.n_hides,
                     indices=indices[2],
                     color="blue")

        end = time.time()
        print("Time needed:",
              humanreadable_time(end-start)
              )


def test_Omega_on_the_fly(task_params: TaskNGridParameters):
    """_summary_

    Args:
        task_params (TaskNGridParameters): _description_
    """

    task_sets_n_cardins = TaskSetsNCardinalities(
        task_params=task_params
    )
    start = time.time()
    Omega_a_j_generator(task_states_n_cardins=task_sets_n_cardins,
                        a=0,
                        o=[0]*(task_params.n_nodes + 1))
    end = time.time()
    print("Time needed to compute one Omega_a_j: ",
            humanreadable_time(end-start))


def test_bitarrays():

    def print_size(num_elements):
        # Determine the number of bits required to represent each element
        # Assuming each element is a boolean value (0 or 1)
        bits_per_element = 1

        # Calculate the total number of bits
        total_bits = num_elements * bits_per_element

        # Convert bits to bytes, kilobytes, megabytes, etc.
        bytes_size = total_bits // 8
        kilobytes_size = bytes_size / 1024
        megabytes_size = kilobytes_size / 1024
        gigabytes_size = megabytes_size / 1024

        print(f"\nTheoretical size of the bitarray of size {num_elements}")
        print(f"Bits: {total_bits}")
        print(f"Bytes: {bytes_size}")
        print(f"Kilobytes: {kilobytes_size}")
        print(f"Megabytes: {megabytes_size}")
        print(f"Gigabytes: {gigabytes_size}")

    def compare_sparse_and_bitarray():
        a = bitarray([0, 0, 0, 0, 0])
        b = np.array([0, 0, 0, 0, 0], dtype=np.int8)

        sparse_array = sp.csr_matrix((48, 126), dtype=np.int8)
        bit_array = bitarray(48 * 126)
        print(asizeof.asizeof(sparse_array))
        print(asizeof.asizeof(bit_array))

    n_O = 375411529402    # size of set O for 5x5 grid with 6 hiding spots
    n_O2 = 187705764701   # size of set O2 for 5x5 grid with 6 hiding spots

    # mba = bitarray(n_O)  # MemoryError
    print_size(n_O)
    print_size(n_O2)

    O2_length_bit_array = bitarray(n_O2)
    print(f"\nActual size of bitarray of size {n_O2}:",
          "\nGigabytes: ",
          asizeof.asizeof(O2_length_bit_array)/(1024 * 1024 * 1024),
          )


if __name__ == "__main__":

    # Define experiment / simulation label
    DIM = 5
    HIDES = 6

    # Define task configuration parameters
    task_parameters = TaskNGridParameters(
        dim=DIM,
        n_hides=HIDES,
        n_blocks=1,
        n_rounds=1,
        n_trials=12
        )

    #test_O_indices_computations(task_params=task_parameters)

    test_Omega_on_the_fly(task_params=task_parameters)  # --> 10 sec

    test_loops(task_params=task_parameters, inner_only=True)

    test_bitarrays()
