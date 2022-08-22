import pandas as pd
import numpy as np
import os

from .define_agent_models import define_agent_model
from .task import Task
from .agent import Agent
from .model import Model
import time

np.set_printoptions(linewidth=500)


def start_agent_task_interaction(
    working_dir,
    output_dir,
    n_blocks,
    n_rounds,
    n_trials,
    dim,
    n_hides,
    agent_models,
    task_configs,
    model_comps,
):
    """This function simulates the interaction between
    an agent object and a task object under a behavioral model
    and writes a tabular file with simulated data to output_dir.

    Parameters
    ----------
    working_dir : str
        Current working directory
    output_dir: str
        Output directory to which simulated data are written to
    n_blocks : int
        Number of blocks (i.e. number games with distinct task configurations)
    n_rounds : int
        Number of rounds within one block
    n_trials : int
        Number of trials within one round
    dim : int
        Dimensionality of the gridworld
    n_hides : int
        Number of hiding spots
    agent_models : list of str
        List of agent models
    task_configs : dict
        {'s_1', 's_3_tr_loc', 'hides_loc'}
    model_comps : obj
        Object of class ModelComponents
    """

    n_nodes = dim**2  # number of nodes in the gridworld

    # Iterate generating models (for one task configuration)
    for agent_model in agent_models:

        # If control agent (which are probabilistic) repeat 3 times
        if "C" in agent_model:
            repetitions = 50
        else:
            repetitions = 1

        for repetition in range(repetitions):

            # Create output dir and fn
            sim_data = pd.DataFrame()
            if "C" in agent_model:
                sub_id = f"{agent_model}_{repetition}"
            else:
                sub_id = agent_model

            sub_dir = os.path.join(output_dir, f"sub-{sub_id}", "beh")

            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            fn_stem = os.path.join(sub_dir, f"sub-{sub_id}_task-th_beh")

            start = time.time()

            # Define generating agent model initialization object
            bayesian, exploring = define_agent_model(agent_model)

            # -----Start Routine "simulation" ------

            for this_block in range(n_blocks):
                print(
                    f"Agent {agent_model} (rep: {repetition} "
                    f"playing block {this_block} ..."
                )
                # ------Prepare Routine "block"-------

                # Initialize dataframe for data recording
                # (data of all rounds in current block)
                sim_dat_b = pd.DataFrame()

                # Instantiate task object
                task = Task(
                    working_dir=working_dir,
                    n_rounds=n_rounds,
                    n_trials=n_trials,
                    dim=dim,
                    n_hides=n_hides,
                )

                # Get task configuration and set s_4 accordingly
                task.hides_loc = task_configs["hides_loc"][this_block]
                task.eval_s_4()

                # Initialize agent and model object
                agent = Agent(
                    agent=agent_model,
                    bayesian=bayesian,
                    exploring=exploring,
                    model_comps=model_comps,
                    task_object=task,
                )

                model = Model()

                # Connect interacting models
                model.agent = agent  # Embed agent object in model init. object
                task.model = model  # Embed model object in task
                agent.model = model  # Embed model object in agent

                # Record hiding spots
                hides_loc_t = task.hides_loc
                s_4_hide_node = task.s4

                # ------Start Routine "block" -------

                for this_round in range(n_rounds):

                    # ------Prepare Routine "round"-------

                    # Fetch task configuration
                    task.s1_t = task_configs["s_1"][this_block, this_round]
                    task.s3_c = task_configs["s_3_tr_loc"][this_block,
                                                           this_round]

                    # Initialize arrays, objects for trialwise data recording
                    # Task states
                    s_1_t = np.full(n_trials + 1, np.nan)  # Current position
                    s_2_node_color_t = np.full((n_trials + 1, n_nodes), np.nan)

                    # Variables for computations
                    tr_disc_t = np.full(n_trials + 1, np.nan)  # tr discovery
                    n_black_t = np.full(n_trials + 1, np.nan)
                    n_grey_t = np.full(n_trials + 1, np.nan)
                    n_blue_t = np.full(n_trials + 1, np.nan)
                    drill_finding_t = np.full(n_trials + 1, np.nan)
                    tr_found_on_hide_t = np.full(
                        n_trials + 1, np.nan
                    )  # whether tr was found on hide
                    zero_sum_denom_t = np.full(
                        n_trials + 1, np.nan
                    )  # whether post belief update denominator sum was zero

                    # Observation, marginal beliefs, valence, decision, action
                    o_t = np.full(
                        n_trials + 1, np.nan
                    )  # observation of treasure discovery
                    marg_s3_b_t = np.full(
                        (n_trials + 1, n_nodes), np.nan
                    )  # belief state hiding spots
                    marg_s4_b_t = np.full(
                        (n_trials + 1, n_nodes), np.nan
                    )  # belief state hiding spots
                    a_s1_t = np.full(
                        n_trials + 1, np.nan, dtype=object
                    )  # state-dependent action set
                    o_s2_t = np.full(
                        n_trials + 1, np.nan, dtype=object
                    )  # state-dependent action set
                    v_t = np.full(
                        n_trials + 1, np.nan, dtype=object
                    )  # decision valence values in each trial
                    d_t = np.full(n_trials + 1, np.nan)  # agent's decision
                    a_t = np.full(n_trials + 1, np.nan)  # action
                    p_o_giv_o_t = np.full(
                        n_trials + 1, np.nan, dtype=object
                    )  # b-dep posterior predictive distribution
                    kl_giv_a_o = np.full(
                        n_trials + 1, np.nan, dtype=object
                    )  # b-dep posterior predictive distribution
                    agent_t = np.full(
                        (n_trials + 1, 1), np.nan
                    )  # Recordings for sanity checks

                    # Max belief state computations
                    max_s3_b_value_t = np.full(n_trials + 1, np.nan)
                    max_s3_b_nodes_t = np.full(n_trials + 1,
                                               np.nan,
                                               dtype=object)
                    shortest_dist_to_max_s3_b_t = np.full(n_trials + 1, np.nan)
                    closest_max_s3_b_nodes_t = np.full(
                        n_trials + 1, np.nan, dtype=object
                    )

                    # -----Start Routine "round"------------------------

                    # Start new round and reset those states that are not
                    # transferred round-to-round
                    agent.c = this_round
                    task.start_new_round()
                    agent.start_new_round()

                    # Record treasure location and marginal priors
                    s_3_tr_loc = np.full(n_trials + 1, task.s3_c)

                    marg_s3_prior_c = agent.marg_s3_prior
                    marg_s4_prior_c = agent.marg_s4_prior

                    for this_t in range(n_trials):

                        # -----Prepare Routine "trial"-----------
                        # Seems like there is nothing prepare

                        # ------Start Routine "trial", agent-task interaction--
                        agent.t = this_t
                        task.t = this_t

                        # ---------------------------------------------------------------------------
                        # trial BEGINNING recordings
                        agent_t[this_t] = agent.t  # agent's trial count
                        s_1_t[this_t] = task.s1_t  # current position
                        s_2_node_color_t[
                            this_t
                        ] = task.s2_t  # reflects s at end of the last trial
                        n_black_t[this_t] = task.n_black
                        n_grey_t[this_t] = task.n_grey
                        n_blue_t[this_t] = task.n_blue
                        o_t[this_t] = task.o_t  # Record agent observation
                        # ---------------------------------------------------------------------------

                        # ---------------------------------------------------------------------------
                        # task-agent-model-Interaction
                        task.start_new_trial()
                        agent.start_new_trial()
                        task.return_observation()
                        agent.update_belief_state()
                        agent.make_decision()
                        model.return_action()
                        task.eval_action()

                        # ---------------------------------------------------------------------------
                        # trial END recordings
                        marg_s3_b_t[this_t] = agent.marg_s3_b
                        marg_s4_b_t[this_t] = agent.marg_s4_b
                        zero_sum_denom_t[this_t] = agent.zero_sum_denominator
                        a_s1_t[this_t] = agent.a_s1  # state-dep. action set
                        o_s2_t[this_t] = agent.o_s2  # state-dep. obs. set
                        p_o_giv_o_t[this_t] = agent.p_o_giv_o
                        kl_giv_a_o[this_t] = agent.kl_giv_a_o
                        max_s3_b_value_t[this_t] = agent.max_s3_b_value
                        max_s3_b_nodes_t[this_t] = agent.max_s3_b_nodes
                        shortest_dist_to_max_s3_b_t[
                            this_t
                        ] = agent.shortest_dist_to_max_s3_b
                        closest_max_s3_b_nodes_t[this_t] = \
                            agent.closest_max_s3_b_nodes
                        v_t[this_t] = agent.v  # valences
                        d_t[this_t] = agent.d  # decision
                        a_t[this_t] = model.a_t  # action
                        tr_disc_t[
                            this_t
                        ] = task.tr_disc_t  # tr. discovery of # this trial
                        drill_finding_t[this_t] = task.drill_finding
                        tr_found_on_hide_t[this_t] = task.tr_found_on_blue
                        # -----------------------------------------------------

                        # End round, if treasure discovered
                        if task.tr_disc_t == 1:
                            # Return t+1 obs. and make add. belief update
                            task.return_observation()
                            agent.update_belief_state()

                            # Record otherwise trial BEGINNING recordings
                            marg_s3_b_t[this_t + 1] = agent.marg_s3_b
                            marg_s4_b_t[this_t + 1] = agent.marg_s4_b

                            agent_t[this_t + 1] = agent.t
                            s_1_t[this_t + 1] = task.s1_t  # current position
                            s_2_node_color_t[
                                this_t + 1
                            ] = task.s2_t  # s at end of the last trial
                            n_black_t[
                                this_t + 1
                            ] = task.n_black
                            n_grey_t[this_t + 1] = task.n_grey
                            n_blue_t[this_t + 1] = task.n_blue

                            break

                    # ------Ending Routine "round"------
                    sim_dat_c = pd.DataFrame(
                        index=range(0, n_trials + 1)
                    )  # Create dataframe
                    sim_dat_c["agent"] = f"Agent {agent_model}"
                    sim_dat_c["block"] = this_block + 1
                    sim_dat_c["round"] = this_round + 1
                    sim_dat_c["trial"] = range(1, n_trials + 2)

                    sim_dat_c["s1_pos"] = s_1_t  # Current node position
                    sim_dat_c["s2_node_color"] = np.full(n_trials + 1, np.nan)
                    sim_dat_c["s2_node_color"] = sim_dat_c[
                        "s2_node_color"].astype("object")
                    for trial in range(n_trials + 1):
                        sim_dat_c.at[trial, "s2_node_color"] = \
                            s_2_node_color_t[trial]
                    sim_dat_c["s3_tr_loc"] = s_3_tr_loc  # Treasure location
                    sim_dat_c["s4_hide_node"] = np.full(n_trials + 1, np.nan)
                    sim_dat_c["s4_hide_node"] = sim_dat_c[
                        "s4_hide_node"].astype("object")
                    for trial in range(n_trials + 1):
                        sim_dat_c.at[trial, "s4_hide_node"] = s_4_hide_node

                    sim_dat_c["o"] = o_t

                    sim_dat_c["a_s1"] = np.full(n_trials + 1, np.nan)
                    sim_dat_c["a_s1"] = sim_dat_c["a_s1"].astype("object")
                    for t in range(n_trials + 1):
                        sim_dat_c.at[t, "a_s1"] = a_s1_t[t]

                    sim_dat_c["max_s3_b_value"] = max_s3_b_value_t

                    sim_dat_c["max_s3_b_nodes"] = np.full(n_trials + 1, np.nan)
                    sim_dat_c["max_s3_b_nodes"] = sim_dat_c[
                        "max_s3_b_nodes"].astype("object")
                    for t in range(n_trials + 1):
                        sim_dat_c.at[t, "max_s3_b_nodes"] = max_s3_b_nodes_t[t]

                    sim_dat_c["shortest_dist_max_s3_b"] = \
                        shortest_dist_to_max_s3_b_t

                    sim_dat_c["closest_max_s3_b_nodes"] = np.full(n_trials + 1,
                                                                  np.nan)
                    sim_dat_c["closest_max_s3_b_nodes"] = sim_dat_c[
                        "closest_max_s3_b_nodes"].astype("object")
                    for t in range(n_trials + 1):
                        sim_dat_c.at[t, "closest_max_s3_b_nodes"] = \
                            closest_max_s3_b_nodes_t[t]

                    sim_dat_c["o_s2"] = np.full(n_trials + 1, np.nan)
                    sim_dat_c["o_s2"] = sim_dat_c["o_s2"].astype("object")
                    for t in range(n_trials + 1):
                        sim_dat_c.at[t, "o_s2"] = o_s2_t[t]

                    sim_dat_c["p_o_giv_o"] = np.full(n_trials + 1, np.nan)
                    sim_dat_c["p_o_giv_o"] = sim_dat_c["p_o_giv_o"].astype(
                        "object")
                    for t in range(n_trials + 1):
                        sim_dat_c.at[t, "p_o_giv_o"] = p_o_giv_o_t[t]

                    sim_dat_c["kl_giv_a_o"] = np.full(n_trials + 1, np.nan)
                    sim_dat_c["kl_giv_a_o"] = sim_dat_c[
                        "kl_giv_a_o"].astype("object")
                    for t in range(n_trials + 1):
                        sim_dat_c.at[t, "kl_giv_a_o"] = kl_giv_a_o[t]

                    sim_dat_c["v"] = np.full(n_trials + 1, np.nan)
                    sim_dat_c["v"] = sim_dat_c["v"].astype("object")
                    for trial in range(n_trials + 1):
                        sim_dat_c.at[trial, "v"] = v_t[trial]
                    sim_dat_c["d"] = d_t

                    sim_dat_c["action"] = a_t

                    sim_dat_c["tr_disc"] = tr_disc_t  # Treasure discovery
                    sim_dat_c["drill_finding"] = drill_finding_t
                    sim_dat_c["tr_found_on_blue"] = tr_found_on_hide_t

                    sim_dat_c["n_black"] = n_black_t
                    sim_dat_c["n_grey"] = n_grey_t
                    sim_dat_c["n_blue"] = n_blue_t

                    sim_dat_c["marg_b_s3"] = np.full(n_trials + 1, np.nan)
                    sim_dat_c["marg_b_s3"] = sim_dat_c["marg_b_s3"].astype(
                        "object")
                    for t in range(n_trials + 1):
                        sim_dat_c.at[t, "marg_b_s3"] = marg_s3_b_t[t]

                    sim_dat_c["marg_b_s4"] = np.full(n_trials + 1, np.nan)
                    sim_dat_c["marg_b_s4"] = sim_dat_c["marg_b_s4"].astype(
                        "object")
                    for t in range(n_trials + 1):
                        sim_dat_c.at[t, "marg_b_s4"] = marg_s4_b_t[t]

                    sim_dat_c["hiding_spots"] = np.full(n_trials + 1, np.nan)
                    sim_dat_c["hiding_spots"] = sim_dat_c[
                        "hiding_spots"].astype("object")  # hiding spots arrays
                    for trial in range(n_trials + 1):
                        sim_dat_c.at[trial, "hiding_spots"] = hides_loc_t

                    sim_dat_c["marg_s3_prior"] = np.full(n_trials + 1, np.nan)
                    sim_dat_c["marg_s3_prior"] = sim_dat_c[
                        "marg_s3_prior"].astype("object")
                    for trial in range(n_trials + 1):
                        sim_dat_c.at[trial, "marg_s3_prior"] = marg_s3_prior_c

                    sim_dat_c["marg_s4_prior"] = np.full(n_trials + 1, np.nan)
                    sim_dat_c["marg_s4_prior"] = sim_dat_c[
                        "marg_s4_prior"].astype("object")
                    for trial in range(n_trials + 1):
                        sim_dat_c.at[trial, "marg_s4_prior"] = marg_s4_prior_c

                    sim_dat_c["agent_r_count"] = agent.c
                    sim_dat_c["agent_t_count"] = agent_t

                    sim_dat_c["zerosum_denom"] = zero_sum_denom_t
                    # Append df from 'this_round' to df from 'this_block'
                    sim_dat_b = sim_dat_b.append(sim_dat_c, ignore_index=True)
                    # completed all repeats of "rounds"

                # ------Ending Routine "block"-------
                # Append dataframe from 'this_block' to entire Dataframe
                sim_data = sim_data.append(sim_dat_b, ignore_index=True)

                end = time.time()
                print(
                    f"Agent {agent_model} finished block {this_block} in"
                    f" {end-start} sec"
                )

            # Save data
            with open(f"{fn_stem}.tsv", "w") as tsv_file:
                tsv_file.write(sim_data.to_csv(sep="\t", na_rep=np.NaN,
                                               index=False))
