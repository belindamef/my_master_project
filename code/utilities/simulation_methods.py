from dataclasses import dataclass
import pandas as pd
import numpy as np
np.set_printoptions(linewidth=500)


@dataclass
class DefaultSimulationParameters:
    n_repetitions: int = 50  # no. of simulation repetitions for stochastic
    # models
    agent_models = ["C1", "C2", "C3", "A1", "A2", "A3"]  # agent model
    # space


class Recorder:
    """A class to store and iteratively add trial-wise simulation data"""

    out_var_list = []
    sim_data_this_round = {}

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

    def create_rec_df_thisblock(self):
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
            self.sim_data_this_round[var] = np.full(
                self.task_design_params.n_trials + 1, np.nan, dtype=object)

    def record_trial_start(self, trial, task):
        """Record all states and observations before agent makes decision,
        model returns action and task evaluates state transition

        Parameters
        ----------
        trial: int
        task: obj
        """
        self.sim_data_this_round["s1"][trial] = task.s1_t
        self.sim_data_this_round["s2"][trial] = task.s2_t
        self.sim_data_this_round["s3"][trial] = task.s3_c
        self.sim_data_this_round["s4"][trial] = task.s4
        self.sim_data_this_round["n_black"][trial] = task.n_black
        self.sim_data_this_round["n_grey"][trial] = task.n_grey
        self.sim_data_this_round["n_blue"][trial] = task.n_blue
        self.sim_data_this_round["o"][trial] = task.o_t

    def record_trial_ending(self, trial, agent, task, model):
        """Record all states and agent model values after agent made
        decision, model returned action and task evaluated state transition"""
        self.sim_data_this_round["a_giv_s1"][trial] = agent.a_s1
        self.sim_data_this_round["o_giv_s2"][trial] = agent.o_s2
        self.sim_data_this_round["p_o_giv_o"][trial] = agent.p_o_giv_o
        self.sim_data_this_round["kl_giv_a_o"][trial] = agent.kl_giv_a_o
        self.sim_data_this_round["v"][trial] = agent.v
        self.sim_data_this_round["d"][trial] = agent.d
        self.sim_data_this_round["a"][trial] = model.a_t
        self.sim_data_this_round["r"][trial] = task.r_t
        self.sim_data_this_round["information"][trial] = task.drill_finding
        self.sim_data_this_round["tr_found_on_blue"] = task.tr_found_on_blue
        self.sim_data_this_round["marg_s3_posterior"][trial] = agent.marg_s3_b
        self.sim_data_this_round["marg_s3_prior_t0"][trial] = \
            agent.marg_s3_prior
        self.sim_data_this_round["marg_s4_posterior"][trial] = agent.marg_s4_b
        self.sim_data_this_round["marg_s4_prior_t0"][trial] = \
            agent.marg_s4_prior
        self.sim_data_this_round["max_s3_belief"][trial] = agent.max_s3_b_value
        self.sim_data_this_round["argsmax_s3_belief"][trial] = \
            agent.max_s3_b_nodes
        self.sim_data_this_round["min_dist_argsmax_s3_belief"][trial] = \
            agent.shortest_dist_to_max_s3_b
        self.sim_data_this_round["closest_argsmax_s3_belief"][trial] = \
            agent.closest_max_s3_b_nodes
        self.sim_data_this_round["hiding_spots"][trial] = task.hides_loc

    def append_this_round_to_block_df(self, this_round):
        """Append this round's dataframe to this block's dataframe

        Parameters
        ----------
        this_round: int
        """
        # Create a dataframe from recording array dictionary
        sim_data_this_round = pd.DataFrame(self.sim_data_this_round)
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

    def save_data_to_tsv(self, out_filename, agent_model):
        """Safe dataframe to tsv file

        Parameters
        ----------
        agent_model: str
        out_filename: str
        """
        self.sim_data.insert(0, "agent", agent_model)
        # Save data
        with open(f"{out_filename}.tsv", "w", encoding="utf8") as tsv_file:
            tsv_file.write(self.sim_data.to_csv(sep="\t", na_rep=np.NaN,
                                                index=False))
