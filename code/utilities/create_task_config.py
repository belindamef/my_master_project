import numpy as np
import copy as cp
import os
import pandas as pd


class TaskConfigurator:

    def __init__(self, task_params):
        """
        This function is the instantiation operation for a task configuration object

        input:
                object with task parameters
        output:
                none
        """
        # Fetch simulation mode
        self.task_config_type = task_params.task_config_type

        # Fetch task parameter
        self.blocks = task_params.blocks
        self.rounds = task_params.rounds
        self.n_nodes = task_params.n_nodes
        self.n_hides = task_params.n_hides

        # Initialize task states
        self.s_1 = np.full((self.blocks, self.rounds), 0)  # current position
        self.s_3_tr_loc = np.full((self.blocks, self.rounds), 0)  # hidden state, treasure location of current round

        self.hides_loc = np.full((self.blocks, self.n_hides), 0)

        s_1_fn = os.path.join(task_params.config_file_path, 's_1.npy')
        s_3_fn = os.path.join(task_params.config_file_path, 's_3.npy')
        hides_fn = os.path.join(task_params.config_file_path, 'hides.npy')

        # Create task configuration fn and directories or load if existent
        if os.path.exists(task_params.config_file_path):
            self.s_1 = np.load(s_1_fn)
            self.s_3_tr_loc = np.load(s_3_fn)
            self.hides_loc = np.load(hides_fn)
        else:
            os.makedirs(task_params.config_file_path)

            # Sample and save task configuration(s)
            self.sample_task_configs()
            np.save(s_1_fn, self.s_1)
            np.save(s_3_fn, self.s_3_tr_loc)
            np.save(hides_fn, self.hides_loc)
            df_fn = os.path.join(task_params.config_file_path, 'config_params.tsv')
            config_all_block_df = pd.DataFrame()
            for block_ in range(self.blocks):

                config_this_block_df = pd.DataFrame(index=range(0, self.rounds))
                config_this_block_df['block'] = block_
                config_this_block_df['round'] = range(1, self.rounds + 1)
                config_this_block_df['hides'] = np.full(self.rounds, np.nan)
                config_this_block_df['hides'] = config_this_block_df['hides'].astype('object')
                for round_ in range(self.rounds):
                    config_this_block_df.at[round_, 'hides'] = self.hides_loc[block_]
                config_this_block_df['s1'] = self.s_1[block_]
                config_this_block_df['s3'] = self.s_3_tr_loc[block_]

                config_all_block_df = config_all_block_df.append(config_this_block_df, ignore_index=True)
            with open(df_fn, 'w') as tsv_file:
                tsv_file.write(config_all_block_df.to_csv(sep='\t', index=False))

    def sample_hiding_spots(self):
        """Sample <self.n_hides> hiding spots from a discrete uniform distribution
        over all nodes of the gridworld (without replacement) and set state s_4 component values accordingly"""

        for block in range(self.blocks):
            self.hides_loc[block] = np.random.choice(self.n_nodes, self.n_hides, replace=False)

    def sample_start_pos(self):
        """Sample the starting position from a discrete uniform distribution
        over all nodes of the grid world"""

        for block in range(self.blocks):
            for round_ in range(self.rounds):
                self.s_1[block, round_] = np.random.choice(self.n_nodes, 1)

    def sample_treasure_location(self):
        """Sample the treasure location from a discrete uniform distribution
        over the hiding spots and repeat until it's not start position"""

        for block in range(self.blocks):
            for round_ in range(self.rounds):
                # Set treasure to equal start position
                self.s_3_tr_loc[block, round_] = cp.deepcopy(self.s_1[block, round_])

                # Sample treasure location until its not the starting position s_0
                while self.s_3_tr_loc[block, round_] == self.s_1[block, round_]:
                    self.s_3_tr_loc[block, round_] = np.random.choice(self.hides_loc[block], 1)

    def sample_task_configs(self):
        """Sample all task configurations"""
        self.sample_hiding_spots()
        self.sample_start_pos()
        self.sample_treasure_location()