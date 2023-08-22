"""Modul containing classes and methods to compute descriptive statistics."""
import numpy as np
import pandas as pd


class Demographics:
    """Methods to compute group-level demographics"""

    def __init__(self, part_fn):
        self.participants_df = pd.read_csv(part_fn, sep='\t')
        self.demographics_df = pd.DataFrame(index=range(1))

    def compute_demographics(self):
        """Compute group level demographics stats"""
        self.demographics_df['mean_age'] = np.mean(self.participants_df['age'])
        self.demographics_df['std_age'] = np.std(self.participants_df['age'])
        self.demographics_df['min_age'] = np.min(self.participants_df['age'])
        self.demographics_df['max_age'] = np.max(self.participants_df['age'])
        sex_counts = self.participants_df['sex'].value_counts()
        self.demographics_df['n_female'] = sex_counts['female']
        self.demographics_df['n_male'] = sex_counts['male']
        handedness_counts = self.participants_df['handedness'].value_counts()
        self.demographics_df['n_right'] = handedness_counts['right']
        self.demographics_df['n_left'] = handedness_counts['left']


class Counts:
    """Methods to compute participant-level one-dimensional (i.e.
    unconditioned) summary_stats from a treasure hunt data set (experimental
    or simulated).

    Input: data object with attribute .df todo: write proper docstring
    """

    def __init__(self, events_df):
        self.events_df = events_df  # Input dataframe
        self.n_blocks = max(self.events_df["block"])
        self.counts_df = pd.DataFrame(index=range(1))  # Output dataframe

    def check_interrupt(self):
        """
        Check whether task was interrupted, add column with
        self.stats['interrupt'] = 0, for not interrupted
        self.stats['interrupt'] =1 for interrupted and
        change 999 to np.nan  #TODO: Why again not in Data class ??
        """
        if 'esc' in self.events_df['action_v'].unique():
            # Add variable 1 = was interrupted, 0 = not interrupted
            self.counts_df['interrupted'] = 1
        else:
            self.counts_df['interrupted'] = 0

    def eval_duration(self):
        """
        Calculate duration of entire experiment for given participant
        and add column self.stats['duration']
        """
        if 'ons' in self.events_df:
            self.counts_df['duration'] = (
                    self.events_df['ons'][
                        self.events_df['ons'].last_valid_index()]
                    - self.events_df['ons'][
                        self.events_df['ons'].first_valid_index()])
        else:
            self.counts_df['duration'] = np.nan

    def count_trials(self):
        """Count number of non-nan trials and started blocks and rounds and
        add columns """
        # Count non-na action_vs (excluding 999, i.e. 'escape')
        self.counts_df['n_val_actions'] = self.events_df[
            'action_v'].mask(
            self.events_df['action_v'] == 'esc', np.nan).count()
        # Init blocks count, number will increase only if non-nan action exist
        self.counts_df['n_val_blocks'] = 0
        self.counts_df['n_val_rounds'] = 0

        # Check if any valid action_vs in all trials
        if self.counts_df.iloc[0]['n_val_actions'] > 0:

            # Evaluate No of started blocks (excluding 999, i.e. 'escape')
            self.counts_df['n_val_blocks'] = self.events_df['block'][
                self.events_df['action_v'].mask(
                    self.events_df[
                        'action_v'] == 'esc', np.nan).last_valid_index()]

            # Loop through blocks
            for block, block_df in self.events_df.groupby(['block']):
                # Check for valid actions in this block
                if block_df['action_v'].mask(
                        block_df['action_v'] == 'esc', np.nan).count() > 0:
                    # Evaluate No of started blocks (excl. 999, i.e. 'escape')
                    self.counts_df['n_val_rounds'] += self.events_df['round_'][
                        block_df['action_v'].mask(
                            self.events_df[
                                'action_v'] == 'esc',
                            np.nan).last_valid_index()]

    def count_treasures(self):
        """Count number of treasures found over all trials"""
        # ------Count total treasures-----------------
        self.counts_df['n_tr'] = self.events_df['r'].sum()

        # ------Count treasures blockwise-----------------
        count_tr_blockwise = self.events_df.groupby(['block'])['r'].sum()
        cols = []

        for block, block_df in self.events_df.groupby(['block']):
            cols.append(f'n_tr_b{block}')
        n_tr_b_df = pd.DataFrame([count_tr_blockwise.values], columns=cols)
        self.counts_df = pd.concat([self.counts_df, n_tr_b_df], axis=1)
        self.counts_df["mean_tr_over_blocks"] = np.nanmean(n_tr_b_df)
        self.counts_df["std_tr_over_blocks"] = np.nanstd(n_tr_b_df)

        # ------Count treasures roundwise ------(regardless of block)
        count_tr_roundwise = self.events_df.groupby(['round_'])['r'].sum()
        cols = []
        for round_, round_df in self.events_df.groupby(['round_']):
            cols.append(f'n_tr_r{round_}')
        n_tr_r_df = pd.DataFrame([count_tr_roundwise.values], columns=cols)
        self.counts_df = pd.concat([self.counts_df, n_tr_r_df], axis=1)

    def count_action(self):
        """
        Count number different action over all trials
        """

        # ------Count total-----------------------------------
        action_type_counts = self.events_df['action_type'].value_counts()
        # Check if 'drill' action_v existent in df, else assign 0
        if 'drill' in self.events_df['action_type'].values:
            self.counts_df['n_drills'] = action_type_counts['drill']
            self.counts_df['p_drills'] = (
                    self.counts_df.iloc[0]['n_drills']
                    / self.counts_df.iloc[0]['n_val_actions'])
            self.counts_df['mean_drills'] = np.nanmean(
                self.events_df['action_type_num'])
            self.counts_df['std_drills'] = np.nanstd(
                self.events_df['action_type_num'])

        else:
            self.counts_df['n_drills'] = 0
            self.counts_df['p_drills'] = 0
        # Check if 'step' action_v existent in df, else assign 0
        if 'step' in self.events_df['action_type'].values:
            self.counts_df['n_steps'] = action_type_counts['step']
            self.counts_df['p_steps'] = (
                    self.counts_df.iloc[0]['n_steps']
                    / self.counts_df.iloc[0]['n_val_actions'])
        else:
            self.counts_df['n_steps'] = 0
            self.counts_df['p_steps'] = 0

        # ------Count blockwise and block-roundwise-------------

        n_drills_b_df = pd.DataFrame()
        p_drills_b_df = pd.DataFrame()
        n_steps_b_df = pd.DataFrame()
        n_drills_br_df = pd.DataFrame()

        for block, block_df in self.events_df.groupby(['block']):
            action_v_counts_per_b = block_df['action_type'].value_counts()

            # Count drills:
            if 'drill' in block_df['action_type'].values:
                n_drills_b_df[f'n_drills_b{block}'] = [
                    action_v_counts_per_b['drill']]
                p_drills_b_df[f'mean_drills_b{block}'] = [
                    np.nanmean(block_df['action_type_num'])]

                # Count drills roundwise
                for round_, round_df in block_df.groupby(['round_']):
                    action_v_counts_per_br = round_df[
                        'action_type'].value_counts()
                    if 'drill' in round_df['action_type'].values:
                        n_drills_br_df[
                            f'n_drills_b{block}_r{round_}'] = [
                            action_v_counts_per_br['drill']]
                    else:
                        n_drills_br_df[f'n_drills_b{block}_r{round_}'] = [0]
            else:
                n_drills_b_df[f'n_drills_b{block}'] = [0]
                p_drills_b_df[f'mean_drills_b{block}'] = [0]

            # Count steps:
            if 'step' in block_df['action_type'].values:
                n_steps_b_df[f'n_steps_b{block}'] = [
                    action_v_counts_per_b['step']]

                # Count steps roundwise
                for round_, round_df in block_df.groupby(['round_']):
                    action_v_counts_per_br = round_df[
                        'action_type'].value_counts()
                    if 'step' in round_df['action_type'].values:
                        n_drills_br_df[
                            f'n_drills_b{block}_r{round_:02d}'] = [
                            action_v_counts_per_br['step']]
                    else:
                        n_drills_br_df[
                            f'n_drills_b{block}_r{round_:02d}'] = [0]
            else:
                n_steps_b_df[f'n_steps_b{block}'] = [0]

        # ------Count roundwise-------- (over blocks)

        n_drills_r_df = pd.DataFrame()
        n_steps_r_df = pd.DataFrame()
        p_drills_r_df = pd.DataFrame()
        p_steps_r_df = pd.DataFrame()

        for round_, round_df in self.events_df.groupby(['round_']):
            action_v_counts_per_r = round_df['action_type'].value_counts()

            # Count drills:
            if 'drill' in round_df['action_type'].values:
                n_drills_r_df[f'n_drills_r{round_:02d}'] = [
                    action_v_counts_per_r['drill']]
                p_drills_r_df[f'p_drills_r{round_:02d}'] = (
                        [action_v_counts_per_r['drill']]
                        / round_df['action_type'].count())
                # drills_over_blocks_cols = [
                #     col for col in n_drills_br_df.columns
                #     if f'_r{round_:02d}' in col]
                # mean_this_round_over_blocks = np.mean(n_drills_br_df[
                # drills_over_blocks_cols].values)
            else:
                n_drills_r_df[f'n_drills_r{round_:02d}'] = [0]
                p_drills_r_df[f'p_drills_r{round_:02d}'] = [0]

            # Count steps:
            if 'step' in round_df['action_type'].values:
                n_steps_r_df[f'n_steps_r{round_:02d}'] = [
                    action_v_counts_per_r['step']]
                p_steps_r_df[f'p_steps_r{round_:02d}'] = (
                        [action_v_counts_per_r['step']]
                        / round_df['action_type'].count())
            else:
                n_steps_r_df[f'n_steps_r{round_:02d}'] = [0]
                p_steps_r_df[f'p_steps_r{round_:02d}'] = [0]

        # Append to counts dataframe
        self.counts_df = pd.concat([self.counts_df, n_drills_b_df], axis=1)
        self.counts_df = pd.concat([self.counts_df, p_drills_b_df], axis=1)
        self.counts_df = pd.concat([self.counts_df, n_steps_b_df], axis=1)
        self.counts_df = pd.concat([self.counts_df, n_drills_r_df], axis=1)
        self.counts_df = pd.concat([self.counts_df, n_steps_r_df], axis=1)

    def count_all(self):
        """Compute all participant level one-dimensional measures"""
        self.check_interrupt()
        self.eval_duration()
        self.count_treasures()
        self.count_trials()
        self.count_action()


class ConditionalFrequencies:
    """Evaluate conditional frequencies"""

    def __init__(self, events_df):
        self.events_df = events_df  # Input dataframe
        self.cond_frequ_df = pd.DataFrame(index=range(1))  # Output dataframe

    def tr_giv_n_hides(self):
        """
        Absolute and relative frequency (Probability) of finding a treasure
        given the number of unveiled hiding spots

        self.p_tr_disc_giv_hides : rel. frequ. of finding a treasure as a
        function of the number of unveiled hiding spots
        """
        groupby_n_hides = self.events_df.groupby('n_blue')

        # Absolute frequency
        n_tr_giv_n_hides = groupby_n_hides['r'].sum()
        cols = []
        for number in n_tr_giv_n_hides.index:
            cols.append(f'n_tr_giv_hides_{int(number)}')
        n_tr_giv_n_hides_df = pd.DataFrame(
            [n_tr_giv_n_hides.values], columns=cols)
        self.cond_frequ_df = pd.concat(
            [self.cond_frequ_df, n_tr_giv_n_hides_df], axis=1)

        # Relative frequency
        p_tr_giv_n_hides = groupby_n_hides['r'].mean()
        # out: Pandas series with probs grouped by number of hides
        cols = []
        for number in p_tr_giv_n_hides.index:
            cols.append(f'p_tr_giv_hides_{int(number)}')
        p_tr_giv_n_hides_df = pd.DataFrame(
            [p_tr_giv_n_hides.values], columns=cols)
        self.cond_frequ_df = pd.concat(
            [self.cond_frequ_df, p_tr_giv_n_hides_df], axis=1)

    def p_unv_giv_action_v(self):
        """
        Probability (i.e. relative frequency) of unveiling a hiding spot
        when action_v to drill was chosen

        self.p_unv_giv_drill : rel frequ. of unveiling a hiding spot given that
                                    the action to drill was chosen
        """
        groupby_action_type = self.events_df.groupby('action_type')
        p_unv_giv_drill = groupby_action_type['information'].mean()
        # out: Pandas series with prob grouped by action_v type
        cols = []
        for action_type in p_unv_giv_drill.index:
            cols.append(f'p_unv_if_{action_type}')
        p_hide_unv_g_drill_df = pd.DataFrame(
            [p_unv_giv_drill.values], columns=cols)
        self.cond_frequ_df = pd.concat(
            [self.cond_frequ_df, p_hide_unv_g_drill_df], axis=1)

    def p_visible_hide_giv_tr_disc(self):
        """
        Probability (i.e. relative frequency) of that the current position
        was an unveiled hiding spot,
        when a treasure was discovered.

        This kind of describes the percentage of treasures found on an
        unveiled and visible hiding spot versus
        not unveiled hiding spots
        """
        groupby_tr_disc = self.events_df.groupby('r')
        # group trial in those, where treasure was discovered and those
        # where no treasure was discovered
        p_was_unv_hide_giv_tr_disc = groupby_tr_disc[
            'tr_found_on_blue'].mean()
        # out: Pandas series with probs that a spot was an unveiled hiding
        # spot grouped by treasure discovered in current trial or not
        cols = []
        for tr_disc_status in p_was_unv_hide_giv_tr_disc.index:
            cols.append(f'p_visible_hide_giv_{tr_disc_status}_tr_disc')
        p_was_unv_hide_giv_tr_disc_df = pd.DataFrame(
            [p_was_unv_hide_giv_tr_disc.values], columns=cols)
        self.cond_frequ_df = pd.concat(
            [self.cond_frequ_df, p_was_unv_hide_giv_tr_disc_df], axis=1)

    def compute_all_p(self):
        """Compute all relative frequencies"""
        self.tr_giv_n_hides()
        self.p_unv_giv_action_v()
        self.p_visible_hide_giv_tr_disc()


class DescrStats(Demographics, Counts, ConditionalFrequencies):
    """_summary_

    Args:
        Demographics (_type_): _description_
        Counts (_type_): _description_
        ConditionalFrequencies (_type_): _description_
    """
    def __init__(self, events_df, dataset, subject=np.nan, part_fn=None):
        self.dataset = dataset  # 'beh' or 'sim'
        # self.subject = subject  # Can either be sub-ID or group
        if self.dataset == 'sim':
            self.agent = events_df.iloc[0]['agent']
        self.subject = subject
        if self.dataset == 'exp' and self.subject == 'whole_sample':
            Demographics.__init__(self, part_fn)
        Counts.__init__(self, events_df)
        ConditionalFrequencies.__init__(self, events_df)
        self.this_agents_events_df = events_df

    def perform_descr_stats(self):
        """Create one participant row with descriptive statistics"""
        stats_df = pd.DataFrame(index=range(1))

        stats_df['sub_id'] = self.subject
        if self.dataset == 'sim':
            stats_df['agent'] = self.agent
            stats_df['tau_gen'] = self.this_agents_events_df.iloc[0]["tau_gen"]
            stats_df['lambda_gen'] = self.this_agents_events_df.iloc[0][
                "lambda_gen"]

        # Evaluate demographics
        if self.dataset == 'exp' and self.subject == 'whole_sample':
            Demographics.compute_demographics(self)
            stats_df = pd.concat([stats_df, self.demographics_df], axis=1)

        # Evaluate counts
        Counts.count_all(self)
        stats_df = pd.concat([stats_df, self.counts_df], axis=1)

        # Evaluate relative frequencies
        ConditionalFrequencies.compute_all_p(self)
        stats_df = pd.concat([stats_df, self.cond_frequ_df], axis=1)

        return stats_df


class GroupStats(DescrStats):
    """Class for group level statistics"""

    def __init__(self, events_all_subs_df, dataset, descr_df):
        self.events_all_subs_df = events_all_subs_df
        self.descr_df = descr_df
        DescrStats.__init__(self, events_all_subs_df, dataset)
        self.n_subs = np.nan

    def eval_t_wise_stats(self, groupby) -> object:
        """Compute trial-wise action frequencies"""
        group_by_trial = self.events_all_subs_df.groupby([groupby])
        t_wise_counts_df = pd.DataFrame()

        for trial, trial_df in group_by_trial:
            action_type_counts = trial_df['action_type'].value_counts()
            n_action_valid = trial_df['action_v'].mask(
                trial_df['action_v'] == 'esc', np.nan).count()
            t_wise_counts_df.at[trial - 1, 'trial'] = trial
            t_wise_counts_df.at[trial - 1, 'round_'] = 1 + np.floor(
                (trial - 1) / 12)
            if 'step' in trial_df['action_type'].values:
                t_wise_counts_df.at[
                    trial - 1, 'n_steps'] = action_type_counts['step']
                t_wise_counts_df.at[  # TODO: not quite correct ?!?
                    trial - 1, 'p_steps'] = (
                        action_type_counts['step'] / n_action_valid)
                # test = np.mean(action_type_counts['step'])
                t_wise_counts_df.at[
                    trial - 1, 'mean_drill'] = np.nanmean(
                    trial_df['action_type_num'])
                t_wise_counts_df.at[
                    trial - 1, 'std_drill'] = np.nanstd(
                    trial_df['action_type_num'])
                t_wise_counts_df.at[
                    trial - 1, 'var_drill'] = np.var(
                    trial_df['action_type_num'])

            else:
                t_wise_counts_df.at[trial - 1, 'n_steps'] = 0
                t_wise_counts_df.at[trial - 1, 'p_steps'] = 0
            if 'drill' in trial_df['action_type'].values:
                t_wise_counts_df.at[
                    trial - 1, 'n_drills'] = action_type_counts['drill']
                t_wise_counts_df.at[
                    trial - 1, 'p_drills'] = (action_type_counts['drill']
                                              / n_action_valid)
            else:
                t_wise_counts_df.at[trial - 1, 'n_drills'] = 0
                t_wise_counts_df.at[trial - 1, 'p_drills'] = 0

        return t_wise_counts_df

    def eval_r_wise_stats(self):
        """Evaluate roundwise descriptive statistics"""

        self.n_subs = len(list(set(self.events_all_subs_df['sub_id'])))
        r_wise_stats_df = pd.DataFrame()

        # Drills per round
        roundwise_drills_col_names = [
            col for col in self.descr_df.columns if 'n_drills_r' in col]

        dic = {}
        for col in roundwise_drills_col_names:
            dic[int(col[col.find('n_drills_r') + 10:])] = self.descr_df[
                col].to_numpy()
            this_r_stats_df = pd.DataFrame(index=range(1))
            this_r_stats_df['round_'] = int(col[col.find('n_drills_r') + 10:])
            this_r_stats_df['drills_mean'] = np.mean(self.descr_df[col])
            this_r_stats_df['drills_sdt'] = np.std(self.descr_df[col])
            this_r_stats_df['drills_sem'] = (this_r_stats_df.loc[
                                                 0, 'drills_sdt']
                                             / np.sqrt(self.n_subs))
            r_wise_stats_df = r_wise_stats_df.append(
                this_r_stats_df, ignore_index=True)

        return r_wise_stats_df

    def perform_group_descr_stats(self, group_by: str) -> pd.DataFrame:
        """Method to compute group level descriptive stats

        Args:
            group_by (str): _description_

        Returns:
            pd.Dataframe: _description_
        """
        group_descr_stats_df = pd.DataFrame()

        for group_, group_df in self.events_df.groupby([group_by]):
            self.subject = group_  # assign group label
            self.n_subs = len(list(group_df.sub_id.unique()))

            if self.dataset == 'sim':
                self.agent = group_df.iloc[0]['agent']
            Counts.__init__(self, group_df)
            ConditionalFrequencies.__init__(self, group_df)
            self.events_df = group_df  # Change current event_df being analyzed

            descr_stats_this_group_df = self.perform_descr_stats()
            # If experimental group --> only one group take whole
            # descr_allsubs dataframe
            if self.dataset == 'exp':
                descr_stats_this_group_df['mean_tr_over_subs'] = np.nanmean(
                    self.descr_df['n_tr'])
                descr_stats_this_group_df['std_tr_over_subs'] = np.nanstd(
                    self.descr_df['n_tr'])
                descr_stats_this_group_df['min_tr_over_subs'] = min(
                    self.descr_df['n_tr'])
                descr_stats_this_group_df['max_tr_over_subs'] = max(
                    self.descr_df['n_tr'])

                # Compute treausures mean over blocks and subs
                tr_per_block_col_names = [
                    col_name for col_name in
                    descr_stats_this_group_df.columns if 'n_tr_b' in col_name]
                for col in tr_per_block_col_names:
                    descr_stats_this_group_df[
                        f'mean_tr_over_sub_bw_{col}'] = np.nanmean(
                        self.descr_df[col])
                mean_tr_per_block_col_names = [
                    col_name for col_name in descr_stats_this_group_df.columns
                    if 'mean_tr_over_sub_bw_' in col_name]
                mean_tr_per_block_df = descr_stats_this_group_df[
                    mean_tr_per_block_col_names]
                descr_stats_this_group_df['mean_tr_over_b'] = np.nanmean(
                    mean_tr_per_block_df)
                descr_stats_this_group_df['std_tr_over_b'] = np.nanstd(
                    mean_tr_per_block_df)
                descr_stats_this_group_df['min_tr_over_b'] = np.min(
                    mean_tr_per_block_df)
                descr_stats_this_group_df['max_tr_over_b'] = np.max(
                    mean_tr_per_block_df)

                # Compute action choices average across participants
                descr_stats_this_group_df[
                    'mean_drills_over_subs'] = np.nanmean(
                    self.descr_df['mean_drills'])
                descr_stats_this_group_df[
                    'std_drills_over_subs'] = np.nanstd(
                    self.descr_df['mean_drills'])
                descr_stats_this_group_df[
                    'var_drills_over_subs'] = np.var(
                    self.descr_df['mean_drills'])

            # If simulation group, 6 groups, --> take group-spec descr_allsubs
            elif self.dataset == 'sim':
                this_groups_descr_stats_allsubs = self.descr_df.loc[
                    self.descr_df['agent'].str.match(group_)]
                descr_stats_this_group_df['mean_tr_over_subs'] = np.nanmean(
                    this_groups_descr_stats_allsubs['n_tr'])
                descr_stats_this_group_df['std_tr_over_subs'] = np.nanstd(
                    this_groups_descr_stats_allsubs['n_tr'])
                descr_stats_this_group_df[
                    'mean_drills_over_subs'] = np.nanmean(
                    this_groups_descr_stats_allsubs['mean_drills'])
                descr_stats_this_group_df[
                    'std_drills_over_subs'] = np.nanstd(
                    this_groups_descr_stats_allsubs['mean_drills'])
                descr_stats_this_group_df['var_drills_over_subs'] = np.var(
                    this_groups_descr_stats_allsubs['mean_drills'])

                # Compute means over blocks
                tr_per_block_col_names = [
                    col_name for col_name
                    in this_groups_descr_stats_allsubs.columns
                    if 'n_tr_b' in col_name]
                n_tr_per_block_df = this_groups_descr_stats_allsubs[
                    tr_per_block_col_names]
                descr_stats_this_group_df['mean_tr_over_b'] = np.nanmean(
                    n_tr_per_block_df.T)
                descr_stats_this_group_df['std_tr_over_b'] = np.nanstd(
                    n_tr_per_block_df.T)

                drills_per_blocks_col_names = [
                    col_name for col_name
                    in this_groups_descr_stats_allsubs.columns
                    if 'mean_drills_b' in col_name]
                drills_per_block_df = this_groups_descr_stats_allsubs[
                    drills_per_blocks_col_names]
                descr_stats_this_group_df['mean_drills_over_b'] = np.nanmean(
                    drills_per_block_df.T)
                descr_stats_this_group_df['std_drills_over_b'] = np.nanstd(
                    drills_per_block_df.T)

            group_descr_stats_df = group_descr_stats_df.append(
                descr_stats_this_group_df, ignore_index=True)

        return group_descr_stats_df

    def perform_grp_lvl_stats(self, group_by):
        """Perform group level descriptive statistics"""
        group_lvl_stats_df = pd.DataFrame()

        for group_, group_df in self.descr_df.groupby([group_by]):
            self.subject = group_
            self.n_subs = len(list(group_df.sub_id.unique()))

            this_groups_df = pd.DataFrame(index=range(1))
            this_groups_df['group'] = group_

            for col_name, col_data in group_df.iteritems():
                if col_name in ['sub_id', 'agent', 'interrupted', 'duration']:
                    continue

                # TODO: better to check for dtype == string, but don't know how
                this_groups_df[f'{col_name}_mean'] = np.mean(col_data.values)
                this_groups_df[f'{col_name}_std'] = np.std(col_data.values)
                this_groups_df[f'{col_name}_sem'] = (this_groups_df.loc[
                                                         0, f'{col_name}_std']
                                                     / np.sqrt(self.n_subs))

            group_lvl_stats_df = group_lvl_stats_df.append(
                this_groups_df, ignore_index=True)

        return group_lvl_stats_df
