import numpy as np


def node_to_rowcol(nodes, dim):
    """
    This function transforms positions node notation to a row and column
    notation in a gridworld with dimension dim

    Input
        nodes : (nodes.size)-array with positions in nodes notation
        dim   : dimensionality of gridworld

    Output
        rowcols : (node.size, 2)-array with position in row n columns notation
    """
    rowcols = np.full((nodes.size, 2), np.nan)

    if nodes.size == 1:
        rowcols[:, 0] = nodes // dim
        rowcols[:, 1] = nodes % dim
    else:
        for index, node in enumerate(nodes):
            rowcols[index, 0] = node // dim
            rowcols[index, 1] = node % dim
    return rowcols
