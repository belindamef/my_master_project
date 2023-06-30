import pandas as pd
import numpy as np

np.set_printoptions(linewidth=500)


class Data:
    """
    A class used extract and clean treasure hunt task tabular data from one
    subject contained in a CSV file.
    """

    def __init__(self, dataset, sub_id, fn, dim):
        self.dataset = dataset
        self.events_df = pd.read_csv(fn, sep='\t')
        self.events_df_drop13 = None
        self.events_block = {}
        self.subject = sub_id
        # self.events_df['sub_id'] = sub_id
        self.events_df.insert(loc=0, column='sub_id', value=sub_id)
        self.dim = dim

        if self.dataset == 'sim':
            self.agent = self.events_df.iloc[0]['agent']

    def drop_practice_trials(self):
        """Drop practice trials from data set"""
        index_exp_start = self.events_df.where(
            self.events_df['block_type'] == 'experiment').first_valid_index()
        # Drop practice trials if existent
        if self.events_df.iloc[0]['block_type'] == 'practice':
            n_prac_blocks = self.events_df.iloc[
                self.events_df.where(
                    self.events_df[
                        'block_type'] == 'practice').last_valid_index()][
                'block']
            # Drop practice block
            self.events_df.drop(
                self.events_df.index[0:index_exp_start], axis=0, inplace=True)
            # Reset block count to start with 1
            self.events_df.loc[:, 'block'] -= n_prac_blocks

    def add_continuous_trial_count(self):
        """Add column with continuous trial counts overall and for each
        block """
        self.events_df = self.events_df[self.events_df.trial != 13]
        n_trials = len(self.events_df)
        n_blocks = self.events_df['block'].iloc[-1]
        n_rounds = self.events_df['round_'].iloc[-1]
        n_trials_perround = self.events_df['trial'].iloc[-1]
        n_trials_perblock = n_rounds * n_trials_perround
        trial_col_index = self.events_df.columns.get_loc("trial")
        self.events_df.insert(
            trial_col_index + 1, 'trial_contin', range(1, n_trials + 1))
        self.events_df.insert(trial_col_index + 1,
                              'trial_cont_overallb',
                              n_blocks * list(range(1, n_trials_perblock + 1)))

    def extract_block_wise_dataframes(self):
        """Group by blocks and extract one separate dataframe for each block
        with continuous trial counts """
        for block_, block_df in self.events_df.groupby('block'):
            events_block_drop13 = block_df[block_df.trial != 13]
            n_trials = len(events_block_drop13)
            trial_col_index = events_block_drop13.columns.get_loc("trial")
            events_block_drop13.insert(
                trial_col_index + 1, 'trial_cont', range(1, n_trials + 1))
            self.events_block[block_] = events_block_drop13

    def map_action_types(self):
        """Rename action with verbose expressions and group into action
        types """
        self.events_df['action_v'] = self.events_df['a'].replace(
            [0, -self.dim, 1, self.dim, -1, 999],
            ['drill', 'up', 'right', 'down', 'left', 'esc'])
        self.events_df['action_type'] = self.events_df[
            'action_v'].map({'drill': 'drill',
                             'up': 'step',
                             'down': 'step',
                             'right': 'step',
                             'left': 'step',
                             'esc': 'esc'})
        self.events_df['action_type_num'] = self.events_df[
            'action_type'].map({'drill': 1, 'step': 0})

    @staticmethod
    def str_to_array(values_string):
        """Transform the values in the column 'hide_nodes_s6 to lists"""
        if type(values_string) == str:
            return np.array(
                list(map(float, values_string.strip('][').split(' '))))
        else:
            return values_string

    def prep_data(self):
        """ Prepare data to be analysis ready"""
        if self.dataset == "behavioral":
            self.drop_practice_trials()

        self.add_continuous_trial_count()
        self.map_action_types()
        self.extract_block_wise_dataframes()

        # -------Convert from string to lists and/or arrays------------
        self.events_df['s2'] = self.events_df[
            's2'].apply(self.str_to_array)
        self.events_df['s4'] = self.events_df['s4'].apply(
            self.str_to_array)
