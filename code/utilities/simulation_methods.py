import pandas as pd
import numpy as np
import time
from .config import DirectoryManager, TaskConfigurator
from .task import Task
from .agent import Agent
from .modelling import AgentInitObject, BehavioralModel, BayesianModelComps
np.set_printoptions(linewidth=500)


class Recorder:
    """A class to store and iteratively add trial-wise simulation data"""

    out_var_list = []
    data_one_round = {}

    def __init__(self, task_design_params, *args):
        self.sim_data = pd.DataFrame()
        self.sim_data_this_block = None
        self.define_sim_out_list(*args)
        self.task_design_params = task_design_params

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
            "v", "d", "a", "r", "information", "tr_found_on_blue",
            "marg_s3_posterior", "marg_s3_prior_t0",
            "marg_s4_posterior", "marg_s4_prior_t0",
            "max_s3_belief", "argsmax_s3_belief",
            "min_dist_argsmax_s3_belief", "closest_argsmax_s3_belief",
            "hiding_spots", "n_black", "n_grey", "n_blue"]
        self.out_var_list += [*args]

    def create_rec_df_one_block(self):
        """Create new empty dataframe for this rounds data"""
        self.sim_data_this_block = pd.DataFrame()

    def create_rec_arrays_thisround(self):
        """Create a dictionary with recording arrays for one round

        Returns
        -------
        rec_arrays : dict
            dictionary storing all recording arrays
        """
        for var in self.out_var_list:
            self.data_one_round[var] = np.full(
                self.task_design_params.n_trials + 1, np.nan, dtype=object)

    def record_trial_start(self, trial, task):
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

    def record_trial_ending(self, trial, sim_obj):
        """Record all states and agent beh_model values after agent made
        decision, beh_model returned action and task evaluated state
        transition"""
        self.data_one_round["a_giv_s1"][trial] = sim_obj.agent.a_s1
        self.data_one_round["o_giv_s2"][trial] = sim_obj.agent.o_s2
        self.data_one_round["p_o_giv_o"][trial] = sim_obj.agent.p_o_giv_o
        self.data_one_round["kl_giv_a_o"][trial] = sim_obj.agent.kl_giv_a_o
        self.data_one_round["v"][trial] = sim_obj.agent.v
        self.data_one_round["d"][trial] = sim_obj.agent.d
        self.data_one_round["a"][trial] = sim_obj.beh_model.a_t
        self.data_one_round["r"][trial] = sim_obj.task.r_t
        self.data_one_round["information"][trial] = sim_obj.task.drill_finding
        self.data_one_round["tr_found_on_blue"] = sim_obj.task.tr_found_on_blue
        self.data_one_round["marg_s3_posterior"][trial] = \
            sim_obj.agent.marg_s3_b
        self.data_one_round["marg_s3_prior_t0"][trial] = \
            sim_obj.agent.marg_s3_prior
        self.data_one_round["marg_s4_posterior"][trial] = \
            sim_obj.agent.marg_s4_b
        self.data_one_round["marg_s4_prior_t0"][trial] = \
            sim_obj.agent.marg_s4_prior
        self.data_one_round["max_s3_belief"][trial] = \
            sim_obj.agent.max_s3_b_value
        self.data_one_round["argsmax_s3_belief"][trial] = \
            sim_obj.agent.max_s3_b_nodes
        self.data_one_round["min_dist_argsmax_s3_belief"][trial] = \
            sim_obj.agent.shortest_dist_to_max_s3_b
        self.data_one_round["closest_argsmax_s3_belief"][trial] = \
            sim_obj.agent.closest_max_s3_b_nodes
        self.data_one_round["hiding_spots"][trial] = sim_obj.task.hides_loc

    def append_this_round_to_block_df(self, this_round):
        """Append this round's dataframe to this block's dataframe

        Parameters
        ----------
        this_round: int
        """
        # Create a dataframe from recording array dictionary
        sim_data_this_round = pd.DataFrame(self.data_one_round)
        sim_data_this_round.insert(0, "trial", pd.Series(  # add trial column
            range(1, self.task_design_params.n_trials + 2)))
        sim_data_this_round.insert(0, "round", this_round + 1)  # add round col
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

    def save_data_to_tsv(self, paths, agent_model):
        """Safe dataframe to tsv file

        Parameters
        ----------
        agent_model: str
        paths: Paths
        """
        self.sim_data.insert(0, "agent", agent_model)
        # Save data
        with open(f"{paths.out_filename}.tsv", "w", encoding="utf8") as \
                tsv_file:
            tsv_file.write(self.sim_data.to_csv(sep="\t", na_rep=np.NaN,
                                                index=False))


class Timer:

    start_of_block: float
    end_of_block: float

    def __init__(self, this_block, this_repetition, agent_model):
        """
        Parameters
        ----------
        this_block: int
        this_repetition: int
        agent_model: str
        """
        self.this_block = this_block
        self.this_repetition = this_repetition
        self.agent_model = agent_model

    def start(self):
        self.start_of_block = time.time()
        print(f"Starting simulation for agent {self.agent_model}, "
              f"repetition no. {self.this_repetition + 1} of "
              f"block {self.this_block + 1}")
        return self

    def end(self):
        self.end_of_block = time.time()
        time_this_block = self.end_of_block - self.start_of_block
        print(f"agent {self.agent_model} finished block "
              f"{self.this_block + 1} in"
              f" {round(time_this_block, ndigits=2)} sec")


class Simulator:

    dir_mgr = DirectoryManager()
    task_configs = TaskConfigurator(dir_mgr.paths).get_config()
    bayesian_comps = BayesianModelComps(task_configs.params).get_comps()
    n_repetitions: int = 50  # no. of simulation repetitions for stochastic
    agent_model_space = ["C1", "C2", "C3", "A1", "A2", "A3"]

    taus: list = None  # post-decision noise parameter
    tau: float = None

    agent: Agent = None
    task: Task = None
    beh_model: BehavioralModel = None

    def __init__(self, mode):
        self.mode = mode

    def create_interacting_objects(self, this_block, agent_attributes):
        """Create beh_model objects that interact in each trial

        Parameters
        ----------
        this_block: int
        agent_attributes: AgentInitObject
        """
        self.task = Task(self.task_configs)
        self.task.start_new_block(this_block)
        self.agent = Agent(agent_attributes, self.task)
        if agent_attributes.is_bayesian:
            self.agent.add_bayesian_model_components(self.bayesian_comps)
        self.beh_model = BehavioralModel(self.mode, self.tau)

        # Connect interacting models
        self.beh_model.agent = self.agent
        self.task.behavioral_model = self.beh_model
        self.agent.beh_model = self.beh_model

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
        self.agent.update_belief_state()

    def simulate_trial_interaction(self):
        """Simulate the agent-task interaction, i.e. agent decision, resulting
        action, and task state transition
        """
        self.agent.make_decision()
        self.beh_model.return_action()
        self.task.eval_action()

    def simulate(self, agent_attr):

        for rep in range(self.n_repetitions):
            self.dir_mgr.prepare_output(agent_attr.name, rep)
            recorder = Recorder(self.task_configs.params)

            for block in range(self.task_configs.params.n_blocks):
                timer = Timer(block, rep, agent_attr.name).start()
                recorder.create_rec_df_one_block()
                self.create_interacting_objects(block, agent_attr)

                for round_ in range(self.task_configs.params.n_rounds):
                    recorder.create_rec_arrays_thisround()
                    self.task.start_new_round(block, round_)
                    self.agent.start_new_round(round_)

                    for trial in range(self.task_configs.params.n_trials):
                        self.simulate_trial_start(trial)
                        recorder.record_trial_start(trial, self.task)
                        self.simulate_trial_interaction()
                        recorder.record_trial_ending(trial, self)

                        # End round, if treasure discovered
                        if self.task.r_t == 1:
                            # Evaluate observation and belief update for t + 1
                            self.task.return_observation()
                            self.agent.update_belief_state()
                            recorder.record_trial_start(trial + 1, self.task)
                            break

                    recorder.append_this_round_to_block_df(round_)
                recorder.append_this_block_to_simdata_df(block)
                timer.end()
            recorder.save_data_to_tsv(self.dir_mgr.paths, agent_attr.name)
