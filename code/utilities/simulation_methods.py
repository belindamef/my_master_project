"""
This script contains classes and methods to simulate agent behavioral data.

Author: Belinda Fleischmann
"""

# from dataclasses import dataclass
import time
import pandas as pd
import numpy as np
from .task import Task, TaskStatesConfigurator, TaskNGridParameters
from .agent import AgentAttributes, Agent, StochasticMatrices
from .modelling import BehavioralModel
from .config import humanreadable_time
np.set_printoptions(linewidth=500)


class GenModelNParameterSpaces:
    """Class to store and manage data generating agent model and parameter
    spaces

    Attributes:
    -----------
    agent_space_gen (list): Data generating agent model space
    tau_space_gen (list): Data generating tau parameter space
    lambda_gen_space (list): Data generating lambda parameter space
    current_agent_gen_init_obj (AgentAttributes): Instance of class
        AgenAttributes containing attributes of current simulation's data
            generating agent model.
    current_agent_gen (str): Current simulation's data generating agent model.
    current_tau_gen (float): Current simulation's data generating tau value.
    current_lambda_gen (float): Current simulation's data generating lambda
        value
    """

    agent_space_gen: list
    tau_space_gen: list
    lambda_gen_space: list
    current_agent_gen_init_obj: AgentAttributes
    current_agent_gen: str
    current_tau_gen: float
    current_lambda_gen: float

    n_reps: int
    repetition_numbers: range
    n_participants: int
    participant_numbers: range
    current_rep: int
    current_part: int

    def get_params_from_args(self, args):
        """Method to fetch simulation parameters from command line or bash
        script arguments.

        Args:
            args (TODO): TODO

        Returns:
            TODO: TODO
        """
        self.agent_space_gen = args.agent_model
        self.tau_space_gen = args.tau_gen
        self.lambda_gen_space = args.lambda_gen

        self.repetition_numbers = args.repetition
        self.participant_numbers = args.participant
        self.n_participants = len(self.participant_numbers)
        self.n_reps = len(self.repetition_numbers)

        return self

    def define_params_manually(self, agent_gen_space=None,
                               tau_gen_space=None,
                               lambda_gen_space=None):
        """Method to manually set data generating model and parameter space.

        Args:
        -----
            agent_space_gen (list): Data generating agent model space
            tau_space_gen (list): Data generating tau parameter space
            lambda_gen_space (list): Data generating lambda parameter space
        """

        if agent_gen_space is None:
            self.agent_space_gen = ["C1", "C2", "C3", "A1", "A2", "A3"]
        else:
            self.agent_space_gen = agent_gen_space

        if tau_gen_space is None:
            self.tau_space_gen = np.linspace(0.01, 0.5, 20).tolist()
        else:
            self.tau_space_gen = tau_gen_space

        if lambda_gen_space is None:
            self.lambda_gen_space = np.linspace(0, 1, 20).tolist()
        else:
            self.lambda_gen_space = lambda_gen_space

    def define_numbers(self, n_rep: int = 1, n_part: int = 1,):
        """Method to define number of repetitions and participants to
        class instance.

        Parameters
        ----------
        n_rep : int
            Number of repetitions. Default value is 1
        n_part : int
            Number of participants. Default value is 1
            """
        self.n_reps = n_rep
        self.repetition_numbers = range(self.n_reps)
        self.n_participants = n_part
        self.participant_numbers = range(self.n_participants)

    def create_agent_sub_id(self, current_part: int, current_rep: int) -> str:
        """Create id for this subject. More than one subject id per agent
        possible if >1 repetition per agent

        Args:
        ----
            current_part (int): Current data generating agent participant
            current_rep (int): Current simulation repetition

        Returns:
        ------
            str: Subject ID
        """
        sub_id = (
            f"{self.current_agent_gen_init_obj.name}_" +
            f"rep-{current_rep}_" +
            "tau-" + f"{self.current_tau_gen * 1000}"[:4] +
            "_" +
            "lambda-" + f"{self.current_lambda_gen * 1000}"[:4] +
            "_" +
            f"part-{current_part}"
            ).replace(".", "")

        sub_id.replace(".", "")

        return sub_id


class Recorder:
    """A class to store and iteratively add trial-wise simulation data that
    will be saved to events.tsv file

    Attributes:
    --------
        variable_list (list): Variables that are recorded during simulation

        data_one_round (dict of str: np.ndarray): Recorded data. <key> (str):
            variable name. <value> (np.ndarray): (1 x (n_trials + 1)-array of
                behavioral data of respective variable

        sim_data_this_block (pd.DataFrame):
            (((n_trial + 1)*n_rounds) x n_variables)- Dataframe containing
                behavioral data

        sim_data (pd.DataFrame):
            ((((n_trial + 1)*n_rounds)*n_blocks) x n_variables)- Dataframe
                containing behavioral data

    Args:
    ----
        *args: Variable length argument list of additional variables to be
            recorded
    """

    variable_list = []
    data_one_round = {}
    sim_data_this_block: pd.DataFrame
    sim_data: pd.DataFrame = pd.DataFrame()

    def __init__(self, *args):
        self.define_sim_out_list(*args)

    def define_sim_out_list(self, *args: str):
        """Define a list of variables that are saved to output data,

        Args
        ----------
        *args : Variable length of input arguments

        Returns
        -------
        list: list of variables to be recorded
        """
        self.variable_list = [
            "s1_t", "s2_t", "s3_t",
            "o_t", "a_giv_s1", "o_giv_s2", "p_o_giv_o", "kl_giv_a_o",
            "v", "d", "a", "log_p_a_giv_h", "r_t",
            "marg_s3_posterior", "marg_s3_prior_t0",
            "marg_s4_posterior", "marg_s4_prior_t0",
            "max_s3_belief", "argsmax_s3_belief",
            "min_dist_argsmax_s3_belief", "closest_argsmax_s3_belief",
            "node_colors"]
        self.variable_list += [*args]

    def create_rec_df_one_block(self):
        """Create new empty dataframe for this rounds data"""
        self.sim_data_this_block = pd.DataFrame()

    def create_rec_arrays_thisround(self, n_trials):
        """Create a dictionary with recording arrays for one round

        Args:
        -----
        n_trials (int): Number of trials per round
        """
        for var in self.variable_list:
            self.data_one_round[var] = np.full(
                n_trials + 1, np.nan, dtype=object)

    def record_trial_start(self, trial: int, task: Task):
        """Record all states and observations before agent makes decision,
        beh_model returns action and task evaluates state transition

        Args:
        ----------
        trial (int): Current trial number
        task (Task): Instance of class Task
        """
        self.data_one_round["s1_t"][trial] = task.s1_t
        self.data_one_round["s2_t"][trial] = task.s2_t
        self.data_one_round["s3_t"][trial] = task.s3_t
        self.data_one_round["node_colors"][trial] = task.node_colors
        self.data_one_round["o_t"][trial] = task.o_t

    def record_trial_ending(self, trial: int, task: Task, agent: Agent,
                            beh_model: BehavioralModel):
        """Record all task states and model dynamic values after agent
        made decision, behavioral model returned action and task evaluated
        state transition

        Args:
        ----
            trial (int): Current trial number
            task (Task): Task object
            agent (Agent): Agent object
            beh_model (BehavioralModel): Behavioral model object
        """

        self.data_one_round["a_giv_s1"][trial] = agent.a_s1
        self.data_one_round["o_giv_s2"][trial] = agent.o_s2
        self.data_one_round["p_o_giv_o"][trial] = agent.p_o_giv_o
        self.data_one_round["kl_giv_a_o"][trial] = agent.kl_giv_a_o
        self.data_one_round["v"][trial] = agent.valence_t
        self.data_one_round["d"][trial] = agent.decision_t
        self.data_one_round["a"][trial] = beh_model.action_t
        self.data_one_round["log_p_a_giv_h"][trial] = beh_model.log_likelihood
        self.data_one_round["r_t"][trial] = task.r_t
        self.data_one_round["marg_s3_posterior"][trial] = agent.marg_tr_belief
        self.data_one_round["marg_s3_prior_t0"][
            trial] = agent.marg_tr_belief_prior
        self.data_one_round["marg_s4_posterior"][
            trial] = agent.marg_hide_belief
        self.data_one_round["marg_s4_prior_t0"][
            trial] = agent.marg_hide_belief_prior
        self.data_one_round["max_s3_belief"][trial] = agent.max_s3_b_value
        self.data_one_round["argsmax_s3_belief"][
            trial] = agent.max_tr_b_node_indices
        self.data_one_round["min_dist_argsmax_s3_belief"][
            trial] = agent.shortest_dist_to_max_s3_b
        self.data_one_round["closest_argsmax_s3_belief"][
            trial] = agent.closest_max_s3_b_nodes
        self.data_one_round["s3_t"][trial] = task.s3_t

    def append_this_round_to_block_df(self, this_round: int, n_trials: int):
        """Append this round's dataframe to this block's dataframe

        Args:
        ----------
        this_round (int): Current round number
        n_trials (int): Number of trials per round
        """
        # Create a dataframe from recording array dictionary
        sim_data_this_round = pd.DataFrame(self.data_one_round)
        sim_data_this_round.insert(0, "trial", pd.Series(  # add trial column
            range(1, n_trials + 2)))
        sim_data_this_round.insert(0, "round_", this_round + 1)  # add round
        # col
        self.sim_data_this_block = pd.concat(
            [self.sim_data_this_block, sim_data_this_round],
            ignore_index=True)

    def append_this_block_to_simdata_df(self, this_block):
        """Append this block's dataframe to the overall dataframe for this
        agent and this repetition

        Args
        ----------
        this_block (int): Current block number
        """
        self.sim_data_this_block.insert(0, "block", this_block + 1)
        self.sim_data = pd.concat([self.sim_data, self.sim_data_this_block],
                                  ignore_index=True)

    def wrap_up_data(self, tau_gen: float, lambda_gen: float,
                     agent_name: str):
        """Finalize data set by adding columns for agent and tau parameter

        Args:
        -----
            tau_gen (float): Data generating tau parameter value
            lambda_gen (float): Data generating lambda parameter value
            agent_name (str): Data generating agent model name
        """
        self.sim_data.insert(0, "lambda_gen", lambda_gen)
        self.sim_data.insert(0, "tau_gen", tau_gen)
        self.sim_data.insert(0, "agent", agent_name)


class Simulator():
    """Class to simulate agent behavioral data in interaction with the
    treasure hunt tas.

    Attributes:
    ---------
        agent (Agent): Instance of class Agent
        task: (Task): Instance of class Task
        beh_model (BehavioralModel): Instance of class BehavioralModel
        task_configs (TaskConfigurator): Object storing Bayesian model
            components.
        bayesian_comps (BayesianModelComps): Object storing Bayesian model
            components object.
        Args:
        -----
            task_configs (TaskConfigurator): Task configuration object
            bayesian_comps (BayesianModelComps): Bayesian model components
                object
    """
    agent: Agent
    task: Task
    beh_model: BehavioralModel

    def __init__(self, state_values: TaskStatesConfigurator,
                 agent_stoch_matrices: StochasticMatrices,
                 task_params: TaskNGridParameters = TaskNGridParameters()):

        self.state_values = state_values
        self.task_params = task_params
        self.agent_stoch_matrices = agent_stoch_matrices

    def create_interacting_objects(self, agent_name: str, this_block: int,
                                   tau_gen: float, lambda_gen: float):
        """Method to create beh_model objects that interact in each trial

        Args:
        ------
            agent_name (str): Behavioral model name
            this_block (int): Block number
            tau_gen (float): Generating tau value
            lambda_gen (float): Generating lambda value
        """

        self.task = Task(state_values=self.state_values,
                         task_params=self.task_params)
        self.task.start_new_block(this_block)

        agent_attributes = AgentAttributes(agent_name)
        self.agent = Agent(agent_attr=agent_attributes,
                           task_object=self.task,
                           lambda_=lambda_gen)

        if agent_attributes.is_bayesian:
            self.agent.attach_stoch_matrices(
                stoch_matrices=self.agent_stoch_matrices)

        self.beh_model = BehavioralModel(tau_gen=tau_gen,
                                         agent_object=self.agent)

    def simulate_trial_start(self, this_trial: int):
        """Method to simulate agent task interaction at beginning of a trial.

        Args:
        ------
            this_trial (int): Current trial number
        """
        self.task.t = this_trial
        self.agent.start_new_trial()
        self.task.eval_obs_func_g()
        start = time.time()
        self.agent.update_belief_state(current_action=self.beh_model.action_t)
        end = time.time()
        print(f"trial {this_trial} agent belief state update took "
              f":  {humanreadable_time(end-start)}\n")

    def simulate_trial_interaction(self):
        """Method to simulate the agent-task interaction,
        i.e. agent decision, resulting action, and task state transition
        """
        self.agent.make_decision()
        action_t = self.beh_model.return_action()
        self.task.eval_action(action_t)

    def simulate_trial_interation_for_llh_evaluation(self, data_s_action):
        """Method to simulate trial-wise interaction between agent, task and
        behavioral model to evaluate likelihood of given data's action
        values.

        More specifically, this method frist let's the agent evaluate action
        valences and the behavioral model evaluate the conditional probability
        distribution of all actions given the history of actions and
        observations*. Subsequently, the behavioral model evaluates the
        probability of this trials action as given by the data based on that
        distribution.

        * Note that this history is based on actions and observations as given
        by the data.

        Args:
            data_s_action (np.int64): this trials action value from data
        """
        self.agent.make_decision()
        self.beh_model.eval_p_a_giv_tau()
        self.beh_model.eval_p_a_giv_h_this_action(data_s_action)

        # Embedd data action in behavioral and task model, so that next
        # iteraton in trial simulation has same state as in data
        self.beh_model.action_t = data_s_action
        self.task.eval_action(data_s_action)

    def simulate_beh_data(self, sim_params) -> pd.DataFrame:
        """Run behavioral data simulation routine. Saves data to instance
        attribute self.data.

        Returns:
            pd.DataFrame: _description_
        """
        recorder = Recorder()  # Initialize data recorder

        for this_block in range(self.task_params.n_blocks):
            recorder.create_rec_df_one_block()
            self.create_interacting_objects(sim_params.current_agent_gen,
                                            this_block,
                                            sim_params.current_tau_gen,
                                            sim_params.current_lambda_gen)

            for this_round in range(self.task_params.n_rounds):
                recorder.create_rec_arrays_thisround(
                    n_trials=self.task_params.n_trials)
                self.task.start_new_round(this_block, this_round)
                self.agent.start_new_round(this_round)

                for this_trial in range(self.task_params.n_trials):
                    self.simulate_trial_start(this_trial)
                    recorder.record_trial_start(this_trial, self.task)
                    self.simulate_trial_interaction()
                    recorder.record_trial_ending(
                        this_trial, self.task, self.agent, self.beh_model)

                    # End round, if treasure discovered
                    if self.task.r_t == 1:
                        # Evaluate observation and belief update for t + 1
                        self.task.eval_obs_func_g()
                        self.agent.update_belief_state(self.beh_model.action_t)
                        recorder.record_trial_start(this_trial + 1, self.task)
                        break

                recorder.append_this_round_to_block_df(
                    this_round, self.task.params.n_trials)
            recorder.append_this_block_to_simdata_df(this_block)
        recorder.wrap_up_data(sim_params.current_tau_gen,
                              sim_params.current_lambda_gen,
                              sim_params.current_agent_gen)
        return recorder.sim_data

    def sim_to_eval_llh(self, candidate_tau: float, candidate_lambda: float,
                        candidate_agent: str, data: pd.DataFrame) -> float:
        """Simulate trialwise interactions between agent and task to evaluate
        the conditional loglikelihood (llh) function value for a given
        theta=(tau, lambda) and given data, as sum of llh values over all
        trials.

        Args:
        -----
            candidate_tau (float): _description_
            candidate_lambda (float): _description_
            candidate_agent (str): _description_
            data (pd.DataFrame): _description_

        Returns:
        -------
            float: Candidate models conditional llh function value for given
                dataset

        Notes:
        -----
            Conditional loglikelihood (llh) function refers to the formal
            expression p^{theta}(a_n|a_{1:n}, o_{1:n})
        """

        # Read task design parameters from dataset
        n_blocks = data.block.max()
        n_rounds = data.round_.max()
        n_trials = data.trial.max() - 1  # subtract 1 bc data n_trials = 13

        # Initialize llh recording array
        llhs_all_blocks = np.full((n_blocks, 1), np.nan)

        for block in range(n_blocks):
            self.create_interacting_objects(candidate_agent, block,
                                            candidate_tau, candidate_lambda)
            llhs_all_rounds = np.full((n_rounds, 1), np.nan)  # Init record df

            for round_ in range(n_rounds):

                llhs_all_trials = np.full((n_trials, 1), np.nan)  # Init rec df

                # Extract this round's data
                data_this_round = data[(
                            data.block == block + 1) & (
                            data.round_ == round_ + 1)]

                # Let task and agent object start new round
                self.task.start_new_round(block, round_)
                self.agent.start_new_round(round_)

                for trial in range(n_trials):

                    # Extract this trial's data
                    data_this_trial = data_this_round.query(
                        'trial == (@trial + 1)')

                    # Simulate agent-task-interaction
                    self.simulate_trial_start(trial)

                    self.simulate_trial_interation_for_llh_evaluation(
                        data_s_action=data_this_trial.a.item()
                        )

                    # End round, if treasure discovered
                    if self.task.r_t == 1:
                        # Evaluate observation and belief update for t + 1
                        self.task.eval_obs_func_g()
                        self.agent.update_belief_state(self.beh_model.action_t)
                        break

                    # Record log likelihood
                    # ------------------------

                    # For control agents, use action valence
                    if np.isnan(candidate_tau):
                        prob_data_giv_action = self.agent.valence_t

                        # Work-around to avoid divide zero error
                        prob_data_giv_action[prob_data_giv_action == 0] = 0.001
                        prob_data_giv_action = prob_data_giv_action / sum(
                            prob_data_giv_action)

                        llhs_all_trials[trial] = np.log(
                            prob_data_giv_action[np.where(
                                self.agent.a_s1 == self.beh_model.action_t)]
                                )
                    # For Bayesian agents, use conditional llh function value
                    else:
                        llhs_all_trials[trial] = self.beh_model.log_likelihood

                llhs_all_rounds[round_] = np.nansum(llhs_all_trials)
            llhs_all_blocks[block] = np.nansum(llhs_all_rounds)

        llh_sum_over_all_blocks = np.nansum(llhs_all_blocks)

        return llh_sum_over_all_blocks
