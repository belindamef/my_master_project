import numpy as np
import copy as cp
import os
import pandas as pd


class TaskConfigurator:
    """A Class to create task configurations given a set of task parameters.
    Sampled task configuration npy files are written to output_dir

    ...

    Parameters
    ----------
    task_config_dir : str
        Output directory to which task configuration files are written to
    n_blocks : int
        Number of blocks (i.e. number games with distinct task configurations)
    n_rounds : int
        Number of rounds within one block
    dim : int
        Dimensionality of the gridworld
    n_hides : int
        Number of hiding spots

    Attributes
    ----------
    task_config_dir : str
        Output directory to which task configuration files are written to
    n_blocks : int
        Number of blocks (i.e. number games with distinct task configurations)
    n_rounds : int
        Number of rounds within one block
    dim : int
        Dimensionality of the gridworld
    n_nodes : int
        Number of nodes in the gridworld
    n_hides : int
        Number of hiding spots
    s_1 : array_like
        (n_blocks)x(n_rounds)-dimensional array with values for starting
        positions
    s_3_tr_loc: array_like
        (n_blocks)x(n_rounds)-dimensional array with values for treasure
        locations
    hides_loc: array_like
        (n_blocks)x(n_hides)-dimensional array with values for hiding
        spot locations

    Methods
    -------
    sample_task_configs()
        Samples all task configuration parameters.
    return_task_configuration()
        Returns dictionary with task parameters.
    """

    def __init__(self, task_config_dir, n_blocks, n_rounds, dim, n_hides):
        self.task_config_dir = task_config_dir
        self.n_blocks = n_blocks
        self.n_rounds = n_rounds
        self.n_nodes = dim ** 2
        self.n_hides = n_hides

        # Initialize task states
        self.s_1 = np.full((n_blocks, n_rounds), 0)
        self.s_3_tr_loc = np.full((n_blocks, n_rounds), 0)
        self.hides_loc = np.full((n_blocks, self.n_hides), 0)

    def sample_task_configs(self):
        """Sample all task configurations"""
        # Sample hiding spots from a discrete uniform distribution
        # over all nodes (without replacement)
        for block in range(self.n_blocks):
            self.hides_loc[block] = np.random.choice(self.n_nodes,
                                                     self.n_hides,
                                                     replace=False)

        # Sample the starting position from a discrete uniform
        # distribution over all nodes
        for block in range(self.n_blocks):
            for round_ in range(self.n_rounds):
                self.s_1[block, round_] = np.random.choice(self.n_nodes, 1)

        # Sample the tr location from a discrete uniform distribution over the
        # hiding spots
        for block in range(self.n_blocks):
            for round_ in range(self.n_rounds):
                # Set treasure to equal start position
                self.s_3_tr_loc[block, round_] = cp.deepcopy(self.s_1[block,
                                                                      round_])
                # Sample tr location until it's not the starting position s_0
                while self.s_3_tr_loc[block, round_] == self.s_1[block,
                                                                 round_]:
                    self.s_3_tr_loc[block, round_] = np.random.choice(
                        self.hides_loc[block], 1)

    def return_task_configuration(self):
        """Return task configuration for given task parameters. Task
        configurations will be loaded from task_config_dir if existing for
        given  task parameters or sampled and saved to task_config_dir if
        not existing for given task parameters """
        s_1_fn = os.path.join(self.task_config_dir, 's_1.npy')
        s_3_fn = os.path.join(self.task_config_dir, 's_3.npy')
        hides_fn = os.path.join(self.task_config_dir, 'hides.npy')

        # Create task configuration fn and directories or load if existent
        if os.path.exists(self.task_config_dir):
            self.s_1 = np.load(s_1_fn)
            self.s_3_tr_loc = np.load(s_3_fn)
            self.hides_loc = np.load(hides_fn)
        else:
            os.makedirs(self.task_config_dir)

            # Sample and save task configuration(s)
            self.sample_task_configs()
            np.save(s_1_fn, self.s_1)
            np.save(s_3_fn, self.s_3_tr_loc)
            np.save(hides_fn, self.hides_loc)
            df_fn = os.path.join(self.task_config_dir, 'config_params.tsv')
            config_all_block_df = pd.DataFrame()
            for block_ in range(self.n_blocks):

                config_this_block_df = pd.DataFrame(index=range(0,
                                                                self.n_rounds))
                config_this_block_df['block'] = block_
                config_this_block_df['round'] = range(1, self.n_rounds + 1)
                config_this_block_df['hides'] = np.full(self.n_rounds, np.nan)
                config_this_block_df['hides'] = config_this_block_df[
                    'hides'].astype('object')
                for round_ in range(self.n_rounds):
                    config_this_block_df.at[round_, 'hides'] = \
                        self.hides_loc[block_]
                config_this_block_df['s1'] = self.s_1[block_]
                config_this_block_df['s3'] = self.s_3_tr_loc[block_]

                config_all_block_df = config_all_block_df.append(
                    config_this_block_df, ignore_index=True)
            with open(df_fn, 'w') as tsv_file:
                tsv_file.write(config_all_block_df.to_csv(sep='\t',
                                                          index=False))

        # Save task states to dict
        task_configurations = {'s_1': self.s_1,
                               's_3_tr_loc': self.s_3_tr_loc,
                               'hides_loc': self.hides_loc}

        return task_configurations
