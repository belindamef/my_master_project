import numpy as np


def rowcol_to_xy(rowcol_pos, dim, gridsize):
    """
    Convert node positions in row-column notation
    to node position in coordinate-notation along x&y-axis [cm].

    Parameters
    ----------
    rowcol_pos : array
        array with row columns notation,
        e.g. (1, 1) for top left corner

    dim : int
        dimensionality of the gridworld

    gridsize : int
        size of the gridworld given in cm

    Returns
    -------
    pos_xy : array
        array with x-y-coordinates,
        e.g. (-5, -5) for top left corner

    Examples
    --------
    For the row-column notation of the top left corner position
    (i.e. row: 0, column :0) in a 6x6-dimenional gridworld
    with a size of 12 cm ``rowcol_to_xy`` converts as follows:

    >>> position = np.array([[0., 0.]])
    >>> dimension = 6
    >>> grid_size = 12
    >>> rowcol_to_xy(rowcol_pos=position, dim=dimension, gridsize=grid_size)
    array([-5.,  5.])
    """
    cubesize = gridsize / dim
    # all possible cube positions along x&y-axis [cm], (last entry not needed!)
    stim_pos_all = (np.linspace(0, gridsize, dim + 1) - gridsize / 2) + cubesize / 2
    # Initialize empty array for
    xy_pos = np.empty(2)
    col = int(rowcol_pos[:, 1]) + 1  # add 1 to get (1,1) notation for upper left corner
    row = int(rowcol_pos[:, 0]) + 1
    # enter x-coordinate in cm given col pos., i.e. 2nd entry of pos_start
    xy_pos[0] = stim_pos_all[col - 1]
    # enter y-coordinate in cm given row pos., i.e. 1st of pos_start
    xy_pos[1] = stim_pos_all[-(row + 1)]
    return xy_pos
