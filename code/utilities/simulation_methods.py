"""
This script contains classes and methods to simulate agent behavioral data.

Author: Belinda Fleischmann
"""

# from dataclasses import dataclass
import time
import pandas as pd
import numpy as np
from .task import Task
from .agent import AgentAttributes, Agent, BayesianModelComps
from .modelling import BehavioralModel
from .config import TaskConfigurator
np.set_printoptions(linewidth=500)


class Recorder:
    """A class to store and iteratively add trial-wise simulation data"""

    out_var_list = []
    data_one_round = {}
    sim_data_this_block: pd.DataFrame
    sim_data = pd.DataFrame()

    def __init__(self, *args):
        self.define_sim_out_list(*args)

    def define_sim_out_list(self, *args: str):
        """Define a list of variables that are saved to output data
        Parameters
        ----------
        *args : Variable length of input arguments
        Returns
        -------
        out_va_list : list
        """
        self.out_var_list = [
            "s1", "s2", "s3", "s4",
            "o", "a_giv_s1", "o_giv_s2", "p_o_giv_o", "kl_giv_a_o",
            "v", "d", "a", "log_p_a_giv_h", "r", "information",
            "tr_found_on_blue",
            "marg_s3_posterior", "marg_s3_prior_t0",
            "marg_s4_posterior", "marg_s4_prior_t0",
            "max_s3_belief", "argsmax_s3_belief",
            "min_dist_argsmax_s3_belief", "closest_argsmax_s3_belief",
            "hiding_spots", "n_black", "n_grey", "n_blue"]
        self.out_var_list += [*args]

    def create_rec_df_one_block(self):
        """Create new empty dataframe for this rounds data"""
        self.sim_data_this_block = pd.DataFrame()

    def create_rec_arrays_thisround(self, n_trials):
        """Create a dictionary with recording arrays for one round

        Returns
        -------
        rec_arrays : dict
            dictionary storing all recording arrays
        """
        for var in self.out_var_list:
            self.data_one_round[var] = np.full(
                n_trials + 1, np.nan, dtype=object)

    def record_trial_start(self, trial, task: Task):
        """Record all states and observations before agent makes decision,
        beh_model returns action and task evaluates state transition

        Parameters
        ----------
        trial: int
        task: obj
        """
        self.data_one_round["s1"][trial] = task.s1_t
        self.data_one_round["s2"][trial] = task.s2_t
        self.data_one_round["s3"][trial] = task.s3_c
        self.data_one_round["s4"][trial] = task.s4_b
        self.data_one_round["n_black"][trial] = task.n_black
        self.data_one_round["n_grey"][trial] = task.n_grey
        self.data_one_round["n_blue"][trial] = task.n_blue
        self.data_one_round["o"][trial] = task.o_t

    def record_trial_ending(self, trial, task: Task, agent: Agent,
                            beh_model: BehavioralModel):
        """Record all states and agent beh_model values after agent made
        decision, beh_model returned action and task evaluated state
        transition"""
        self.data_one_round["a_giv_s1"][trial] = agent.a_s1
        self.data_one_round["o_giv_s2"][trial] = agent.o_s2
        self.data_one_round["p_o_giv_o"][trial] = agent.p_o_giv_o
        self.data_one_round["kl_giv_a_o"][trial] = agent.kl_giv_a_o
        self.data_one_round["v"][trial] = agent.valence_t
        self.data_one_round["d"][trial] = agent.decision_t
        self.data_one_round["a"][trial] = beh_model.action_t
        self.data_one_round["log_p_a_giv_h"][trial] = beh_model.log_likelihood
        self.data_one_round["r"][trial] = task.r_t
        self.data_one_round["information"][trial] = task.drill_finding
        self.data_one_round["tr_found_on_blue"] = task.tr_found_on_blue
        self.data_one_round["marg_s3_posterior"][trial] = agent.marg_s3_b
        self.data_one_round["marg_s3_prior_t0"][trial] = agent.marg_s3_prior
        self.data_one_round["marg_s4_posterior"][trial] = agent.marg_s4_b
        self.data_one_round["marg_s4_prior_t0"][trial] = agent.marg_s4_prior
        self.data_one_round["max_s3_belief"][trial] = agent.max_s3_b_value
        self.data_one_round["argsmax_s3_belief"][trial] = agent.max_s3_b_nodes
        self.data_one_round["min_dist_argsmax_s3_belief"][
            trial] = agent.shortest_dist_to_max_s3_b
        self.data_one_round["closest_argsmax_s3_belief"][
            trial] = agent.closest_max_s3_b_nodes
        self.data_one_round["hiding_spots"][trial] = task.hides_loc

    def append_this_round_to_block_df(self, this_round, n_trials):
        """Append this round's dataframe to this block's dataframe

        Parameters
        ----------
        this_round: int
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

        Parameters
        ----------
        this_block: int
        """
        self.sim_data_this_block.insert(0, "block", this_block + 1)
        self.sim_data = pd.concat([self.sim_data, self.sim_data_this_block],
                                  ignore_index=True)

    def wrap_up_data(self, tau, lambda_, agent_name):
        """Finalize data set by adding columns for agent and tau parameter"""
        self.sim_data.insert(0, "lambda_gen", lambda_)
        self.sim_data.insert(0, "tau_gen", tau)
        self.sim_data.insert(0, "agent", agent_name)


class SimulationParameters:
    """Class to store and manage parameters for behavioral data simulation"""

    n_repetitions: int
    repetition_numbers: range
    n_participants: int
    participant_numbers: range
    agent_space_gen: list
    tau_space_gen: list
    lambda_gen_space: list
    current_rep: int
    current_agent_gen_init_obj: AgentAttributes
    current_agent_gen: str
    current_tau_gen: float
    current_lambda_gen: float
    current_part: int

    def get_params_from_args(self, args):
        """Method to fetch simulation parameters from command line or bash
        script arguments."""
        self.repetition_numbers = args.repetition
        self.agent_space_gen = args.agent_model
        self.tau_space_gen = args.tau_value
        self.lambda_gen_space = args.lambda_value
        self.participant_numbers = args.participant
        self.n_participants = len(self.participant_numbers)
        self.n_repetitions = len(self.repetition_numbers)
        return self

    def define_n_reps_and_participants_manually(self, n_rep: int = 1,
                                                n_part: int = 1,):
        """Method to define pass number of repetitions and participants to
        class instance.

        Parameters
        ----------
        n_rep : int
            Number of repetitions. Default value is 1
        n_part : int
            Number of participants. Default value is 1
            """
        self.n_repetitions = n_rep
        self.repetition_numbers = range(self.n_repetitions)
        self.n_participants = n_part
        self.participant_numbers = range(self.n_participants)

    def define_params_manually(self, agent_gen_space=None,
                               tau_gen_space=None,
                               lambda_gen_space=None):
        """Method to manually set data generating model and parameter spacec.

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


class Timer:
    """Class to time start and stop of simulations"""

    start_of_block: float
    end_of_block: float

    def __init__(self, sim_params: SimulationParameters, block):
        """
        Parameters
        ----------
        sim_obj: Simulator
        """
        self.this_block = block
        self.this_repetition = sim_params.current_rep
        self.agent_model = sim_params.current_agent_gen
        self.tau_gen = sim_params.current_tau_gen
        self.lambda_gen = sim_params.current_lambda_gen
        self.participant = sim_params.current_part

    def start(self):
        """Method to start timer.

        Returns:
            __self__: Class instance
        """
        self.start_of_block = time.time()
        print(f"Starting simulation for agent {self.agent_model}, "
              f"participant {self.participant}, "
              f"repetition no. {self.this_repetition} with "
              f"tau: {self.tau_gen}, lambda: {self.lambda_gen}")
        return self

    def end(self):
        """Method to end timer"""
        self.end_of_block = time.time()
        time_this_block = self.end_of_block - self.start_of_block
        print(f"agent {self.agent_model} finished block "
              f"{self.this_block + 1} in"
              f" {round(time_this_block, ndigits=2)} sec")


class Simulator():
    """Class for behavioral data simulation objects

    Attributes:
        agent
    """
    agent: Agent
    task: Task
    beh_model: BehavioralModel
    data: pd.DataFrame

    def __init__(self, task_configs: TaskConfigurator,
                 bayesian_comps: BayesianModelComps,
                 sim_params=SimulationParameters()):
        """_summary_

        Args:
            task_configs (TaskConfigurator): Task configuration object
            bayesian_comps (BayesianModelComps): Bayesian model components
                object
            sim_params (SimulationParameters, optional): simulation parameters.
                Defaults to SimulationParameters().
        """
        self.task_configs = task_configs
        self.bayesian_comps = bayesian_comps
        self.sim_params = sim_params

    def create_agent_sub_id(self) -> str:
        """Create id for this subject. More than one subject id per agent
        possible if >1 repetition per agent

        Returns:
            str: Subject ID
        """
        sub_id = (
            f"{self.sim_params.current_agent_gen_init_obj.name}_" +
            f"rep-{self.sim_params.current_rep}_" +
            "tau-" + f"{self.sim_params.current_tau_gen * 1000}"[:4] +
            "_" +
            "lambda-" + f"{self.sim_params.current_lambda_gen * 1000}"[:4] +
            "_" +
            f"part-{self.sim_params.current_part}"
            ).replace(".", "")

        sub_id.replace(".", "")

        return sub_id

    def create_interacting_objects(self, agent_name: str, this_block: int,
                                   tau_gen: float, lambda_gen: float):
        """Create beh_model objects that interact in each trial

        Args:
            agent_name (str): Behavioral model name
            this_block (int): Block number
            tau_gen (float): Generating tau value
            lambda_gen (float): Generating lambda value
        """
        if agent_name is None:
            agent_attributes = self.sim_params.current_agent_gen_init_obj
        else:
            agent_attributes = AgentAttributes(agent_name)
        self.task = Task(self.task_configs)
        self.task.start_new_block(this_block)

        self.agent = Agent(agent_attributes, self.task, lambda_gen)
        if agent_attributes.is_bayesian:
            self.agent.add_bayesian_model_components(self.bayesian_comps)

        self.beh_model = BehavioralModel(tau_gen, self.agent)

    def simulate_trial_start(self, this_trial: int):
        """Simulate agent task interaction at beginning of a new trial

        Args:
            this_trial (int): Trial number
        """
        self.task.start_new_trial(this_trial)
        self.agent.start_new_trial()
        self.task.return_observation()
        self.agent.update_belief_state(self.beh_model.action_t)

    def simulate_trial_interaction(self):
        """Simulate the agent-task interaction, i.e. agent decision, resulting
        action, and task state transition
        """
        self.agent.make_decision()
        action_t = self.beh_model.return_action()
        self.task.eval_action(action_t)

    def simulate_trial_interation_for_llh_evaluation(self, data_s_action):
        """Method to simulate trial-wise interaction between agent, task and
        behavioral model for the evaluation of data likelihood. More
        specifically, this method frist let's the agent make a decision and the
        behavioral model evaluate the conditional probability distribution of
        All actions given the history of actions and observations*.
        Subsequently the behavioral model evaluates the probability of this
        trials action as given by the data based on theat distribution.

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

    def simulate_beh_data(self):
        """Run behavioral data simulation routine. Saves data to instance
        attribute self.data.
        """
        recorder = Recorder()

        for this_block in range(self.task_configs.params.n_blocks):
            timer = Timer(self.sim_params, this_block).start()
            recorder.create_rec_df_one_block()
            self.create_interacting_objects(self.sim_params.current_agent_gen,
                                            this_block,
                                            self.sim_params.current_tau_gen,
                                            self.sim_params.current_lambda_gen)

            for this_round in range(self.task_configs.params.n_rounds):
                recorder.create_rec_arrays_thisround(
                    self.task_configs.params.n_trials)
                self.task.start_new_round(this_block, this_round)
                self.agent.start_new_round(this_round)

                for this_trial in range(self.task_configs.params.n_trials):
                    self.simulate_trial_start(this_trial)
                    recorder.record_trial_start(this_trial, self.task)
                    self.simulate_trial_interaction()
                    recorder.record_trial_ending(
                        this_trial, self.task, self.agent, self.beh_model)

                    # End round, if treasure discovered
                    if self.task.r_t == 1:
                        # Evaluate observation and belief update for t + 1
                        self.task.return_observation()
                        self.agent.update_belief_state(self.beh_model.action_t)
                        recorder.record_trial_start(this_trial + 1, self.task)
                        break

                recorder.append_this_round_to_block_df(
                    this_round, self.task_configs.params.n_trials)
            recorder.append_this_block_to_simdata_df(this_block)
            timer.end()
        recorder.wrap_up_data(self.sim_params.current_tau_gen,
                              self.sim_params.current_lambda_gen,
                              self.sim_params.current_agent_gen)
        self.data = recorder.sim_data

    def sim_to_eval_llh(self, candidate_tau: float, candidate_lambda: float,
                        candidate_agent: str, data: pd.DataFrame) -> float:
        """Simulate trialwise interactions between agent and task to evaluate
        the llh function value for a given tau and lambda and given data,
        as sum over all trials
        """

        n_blocks = data.block.max()
        n_rounds = data.round_.max()
        n_trials = data.trial.max() - 1  # subtract 1 bc data trials = 13

        llhs_all_blocks = np.full((n_blocks, 1), np.nan)

        for block in range(n_blocks):
            self.create_interacting_objects(candidate_agent, block,
                                            candidate_tau, candidate_lambda)
            llhs_all_rounds = np.full((n_rounds, 1), np.nan)

            for round_ in range(n_rounds):

                llhs_all_trials = np.full((n_trials, 1), np.nan)

                data_this_round = data[(
                            data.block == block + 1) & (
                            data.round_ == round_ + 1)]

                self.task.start_new_round(block, round_)
                self.agent.start_new_round(round_)

                for trial in range(n_trials):

                    data_this_trial = data_this_round.query(
                        'trial == (@trial + 1)')

                    self.simulate_trial_start(trial)

                    self.simulate_trial_interation_for_llh_evaluation(
                        data_s_action=data_this_trial.a.item()
                    )

                    # End round, if treasure discovered
                    if self.task.r_t == 1:
                        # Evaluate observation and belief update for t + 1
                        self.task.return_observation()
                        self.agent.update_belief_state(self.beh_model.action_t)
                        break

                    if np.isnan(candidate_tau):
                        prob_data_giv_action = self.agent.valence_t
                        prob_data_giv_action[prob_data_giv_action == 0] = 0.001
                        prob_data_giv_action = prob_data_giv_action / sum(
                            prob_data_giv_action)

                        llhs_all_trials[trial] = np.log(
                            prob_data_giv_action[np.where(
                                self.agent.a_s1 == self.beh_model.action_t)]
                                )

                    else:
                        llhs_all_trials[trial] = self.beh_model.log_likelihood

                llhs_all_rounds[round_] = np.nansum(llhs_all_trials)
            llhs_all_blocks[block] = np.nansum(llhs_all_rounds)

        llh_sum_over_all_blocks = np.nansum(llhs_all_blocks)

        return llh_sum_over_all_blocks
