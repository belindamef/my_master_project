"""
This script contains classes and methods to simulate agent behavioral data.

Author: Belinda Fleischmann
"""

# from dataclasses import dataclass
import time
import pandas as pd
import numpy as np
from .task import Task
from .agent import Agent
from .modelling import BehavioralModel
np.set_printoptions(linewidth=500)


class Recorder:
    """A class to store and iteratively add trial-wise simulation data"""

    out_var_list = []
    data_one_round = {}
    sim_data_this_block = None
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
        self.data_one_round["s4"][trial] = task.s4
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

    n_repetitions: int
    repetition_numbers: range
    n_participants: int
    participant_numbers: range
    agent_space_gen: list
    tau_space_gen: list
    lambda_gen_space: list
    current_rep: int = None
    current_agent_attributes: object = None
    current_agent_model: str = None
    current_tau_gen: float = None
    current_lambda_gen: float = None
    current_part: int = None

    def get_params_from_args(self, args):
        self.repetition_numbers = args.repetition
        self.agent_space_gen = args.agent_model
        self.tau_space_gen = args.tau_value
        self.lambda_gen_space = args.lambda_value
        self.participant_numbers = args.participant
        return self

    def define_n_reps_and_participants_manually(self, n_rep: int = 1,
                                                n_part: int = 1,):
        self.n_repetitions = n_rep
        self.repetition_numbers = range(self.n_repetitions)
        self.n_participants = n_part
        self.participant_numbers = range(self.n_participants)

    def define_params_manually(self, agent_gen_space: list = None,
                               tau_gen_space: list = None,
                               lambda_gen_space: list = None):

        if agent_gen_space is None:
            self.agent_space_gen = ["C1", "C2", "C3", "A1", "A2", "A3"]
        else:
            self.agent_space_gen = agent_gen_space

        if tau_gen_space is None:
            self.tau_space_gen = np.linspace(0, 2, 21).tolist()
        else:
            self.tau_space_gen = tau_gen_space

        if lambda_gen_space is None:
            self.lambda_gen_space = np.linspace(0, 1, 11)
        else:
            self.lambda_gen_space = lambda_gen_space


class Timer:

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
        self.agent_model = sim_params.current_agent_attributes.name
        self.tau_gen = sim_params.current_tau_gen
        self.participant = sim_params.current_part

    def start(self):
        self.start_of_block = time.time()
        print(f"Starting simulation for agent {self.agent_model}, "
              f"participant {self.participant}, "
              f"repetition no. {self.this_repetition} with "
              f"tau: {self.tau_gen}")
        return self

    def end(self):
        self.end_of_block = time.time()
        time_this_block = self.end_of_block - self.start_of_block
        print(f"agent {self.agent_model} finished block "
              f"{self.this_block + 1} in"
              f" {round(time_this_block, ndigits=2)} sec")


class Simulator():

    agent: Agent = None
    task: Task = None
    beh_model: BehavioralModel = None
    data: pd.DataFrame = None

    def __init__(self, task_configs, bayesian_comps,
                 sim_params=SimulationParameters()):
        self.task_configs = task_configs
        self.bayesian_comps = bayesian_comps
        self.sim_params = sim_params

    def create_interacting_objects(self, this_block, tau_gen, lambda_gen):
        """Create beh_model objects that interact in each trial

        Parameters
        ----------
        this_block: int
        agent_attributes: AgentInitObject
        """
        self.task = Task(self.task_configs)
        self.task.start_new_block(this_block)
        self.agent = Agent(self.sim_params.current_agent_attributes, self.task,
                           lambda_gen)
        if self.sim_params.current_agent_attributes.is_bayesian:
            self.agent.add_bayesian_model_components(self.bayesian_comps)
        self.beh_model = BehavioralModel(tau_gen, self.agent)

        # Connect interacting model
        self.task.beh_model = self.beh_model

    def simulate_trial_start(self, this_trial):
        """Simulate the beginning of a trial, when an
        initial observation is made, but before decision is made

        Parameters
        ----------
        this_trial: int
        """
        self.task.start_new_trial(this_trial)
        self.agent.start_new_trial(this_trial)
        self.task.return_observation()
        self.agent.update_belief_state(self.beh_model.action_t)

    def simulate_trial_interaction(self):
        """Simulate the agent-task interaction, i.e. agent decision, resulting
        action, and task state transition
        """
        self.agent.make_decision()
        self.beh_model.return_action()
        self.task.eval_action()

    def simulate_beh_data(self):
        recorder = Recorder()

        for this_block in range(self.task_configs.params.n_blocks):
            timer = Timer(self.sim_params, this_block).start()
            recorder.create_rec_df_one_block()
            self.create_interacting_objects(this_block,
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
                              self.sim_params.current_agent_attributes.name)
        self.data = recorder.sim_data

    def sim_to_eval_llh(self, current_tau_analyze,
                        current_lambda_analyze) -> float:
        """Simulate trialwise interactions between agent and task to evaluate
        the llh function value for a given tau and lambda and given data,
        as sum over all trials
        """

        # TODO: alle sim und task parameters aus Datensatz lesen?
        llhs_all_blocks = np.full(
            (self.task_configs.params.n_blocks, 1), np.nan)

        for block in range(self.task_configs.params.n_blocks):
            self.create_interacting_objects(
                block, current_tau_analyze, current_lambda_analyze)
            llhs_all_rounds = np.full(
                (self.task_configs.params.n_rounds, 1), np.nan)

            for round_ in range(self.task_configs.params.n_rounds):

                llhs_all_trials = np.full(
                    (self.task_configs.params.n_trials, 1), np.nan)

                data_this_round = self.data[(
                            self.data.block == block + 1) & (
                            self.data.round_ == round_ + 1)]

                self.task.start_new_round(block, round_)
                self.agent.start_new_round(round_)

                for trial in range(self.task_configs.params.n_trials):

                    data_this_trial = data_this_round.query(
                        'trial == (@trial + 1)')

                    self.simulate_trial_start(trial)

                    # TODO: this is a part of simulate_trial_interaction()
                    # Find more elegant way
                    # ----------------------------------
                    # Let agent evaluate valence function.
                    # TODO: resulting decision value is redundant
                    self.agent.make_decision()
                    # fetch action from simulated data
                    data_action_t = data_this_trial.a.item()
                    # Evaluate conditional probability distribution of actions
                    # given tau, aka likeholood of this tau
                    self.beh_model.eval_p_a_giv_tau()
                    # Evaluate conditional probability of data action given
                    # this tau, (aka likelihood of this tau)
                    self.beh_model.eval_p_a_giv_h_this_action(data_action_t)

                    # Embedd data action in behavioral and task model, so that
                    # next iteraton in trial simulation has same state as is
                    # data
                    self.beh_model.action_t = data_action_t
                    self.task.eval_action()

                    # TODO: check if task reward now matches with data
                    # self.task.r_t = data_this_trial.r.item()
                    # End round, if treasure discovered
                    if self.task.r_t == 1:
                        # Evaluate observation and belief update for t + 1
                        self.task.return_observation()
                        self.agent.update_belief_state(self.beh_model.action_t)
                        break

                    llhs_all_trials[trial] = self.beh_model.log_likelihood
                llhs_all_rounds[round_] = np.nansum(llhs_all_trials)
            llhs_all_blocks[block] = np.nansum(llhs_all_rounds)

            llh_sum_over_all_blocks = np.nansum(llhs_all_blocks)

            return llh_sum_over_all_blocks
