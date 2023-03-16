#!/usr/bin/env python3
"""
This script starts the simulation of agent-task-interactions for given
experimental parameter and agent models.

Task configurations will be either loaded from existing configuration files
or sampled at random if not existing for given experimental parameters.
This script calls the method ./utilities/simulation_methods.py which
iterates over the here given set of agents and performs the interactions
for each agent and task configuration.

Author: Belinda Fleischmann
"""

import numpy as np
import time
import utilities.config as config
import utilities.modelling as model
from utilities.task import Task
from utilities.agent import Agent
from utilities.modelling import BehavioralModel
from utilities.simulation_methods import DefaultSimulationParameters, Recorder

np.set_printoptions(linewidth=500)


def prepare_sim(mode, validation_parameters):
    """Prepare paths variables, fetch task design parameters and task
    configurations and create the output directory

    Parameters
    ----------
    validation_parameters: obj
    mode: str
    """
    paths = config.Paths()  # get paths of in, out, code , etc. directories
    task_params = config.TaskDesignParameters()  # get task parameters
    task_configs = config.TaskConfigurator(paths, task_params).get_config()
    config.DirectoryManager.create_data_out_dir(paths)

    if mode == "validation":
        sim_params = validation_parameters.simulation_parameters
    else:
        sim_params = DefaultSimulationParameters()

    return paths, task_params, task_configs, sim_params


def prepare_output(paths, agent_model, this_repetition):
    sub_id = config.create_agent_id(agent_model, this_repetition)
    paths.sub_dir = config.define_and_create_sub_out_dir(
        paths.this_sim_out, sub_id)
    return config.define_out_filename(paths.sub_dir, sub_id)


class TimePrinter:

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

    def start_block(self):
        self.start_of_block = time.time()
        print(f"Starting simulation for agent {self.agent_model}, "
              f"repetition no. {self.this_repetition + 1} of "
              f"block {self.this_block + 1}")
        return self

    def end_block(self):
        self.end_of_block = time.time()
        time_this_block = self.end_of_block - self.start_of_block
        print(f"Agent {self.agent_model} finished block "
              f"{self.this_block + 1} in"
              f" {round(time_this_block, ndigits=2)} sec")


def create_interacting_objects(this_block, paths, task_params, task_configs,
                               agent_attributes):
    # Create task, agent and model objects
    task = Task(paths.code, task_params)
    task.start_new_block(task_configs, this_block)
    # task starting a new block?
    agent = Agent(agent_attributes, task)
    agent.Get_bayesian_model_components(paths, task_params)
    behavioral_model = BehavioralModel()

    # Connect interacting models
    behavioral_model.agent = agent  # Embed agent object in model init. object
    task.behavioral_model = behavioral_model  # Embed model object in task
    agent.behavioral_model = behavioral_model  # Embed model object in agent

    return agent, task, behavioral_model


def simulate_trial_start(this_trial, task, agent):
    """Simulate the beginning of a trial, when an
    initial observation is made, but before decision is made

    Parameters
    ----------
    this_trial: int
    task: Task
    agent: Agent
    """
    task.start_new_trial(this_trial)
    agent.start_new_trial(this_trial)
    task.return_observation()
    agent.update_belief_state()


def simulate_trial_interaction(agent, behavioral_model, task):
    """Simulate the agent-task interaction, i.e. agent decision, resulting
    action, and task state transition

    Parameters
    ----------
    agent: Agent
    behavioral_model: BehavioralModel,
    task: Task"""
    agent.make_decision()
    behavioral_model.return_action()
    task.eval_action()


def main(mode="simulation", validation_parameters=None):
    # Prepare paths, simulation parameters, and task configuration
    paths, task_params, task_configs, sim_params = prepare_sim(
        mode=mode, validation_parameters=validation_parameters)

    for agent_model in sim_params.agent_models:
        agent_attr = model.define_agent_attributes(agent_model)
        if agent_attr.is_deterministic:
            sim_params.n_repetitions = 1

        for this_repetition in range(sim_params.n_repetitions):
            out_filename = prepare_output(paths, agent_model, this_repetition)
            recorder = Recorder(task_params)

            for this_block in range(task_params.n_blocks):
                time_printer = TimePrinter(this_block, this_repetition,
                                           agent_model).start_block()
                recorder.create_rec_df_thisblock()
                agent, task, beh_model = create_interacting_objects(
                    this_block, paths, task_params, task_configs, agent_attr)

                for this_round in range(task_params.n_rounds):
                    task.start_new_round(task_configs, this_block, this_round)
                    agent.start_new_round(this_round)
                    recorder.create_rec_arrays_thisround()

                    for this_trial in range(task_params.n_trials):
                        simulate_trial_start(this_trial, task, agent)
                        recorder.record_trial_start(this_trial, task)
                        simulate_trial_interaction(agent, beh_model, task)
                        recorder.record_trial_ending(
                            this_trial, agent, task, beh_model)

                        # End round, if treasure discovered
                        if task.r_t == 1:
                            # Evaluate observation and belief update for t + 1
                            task.return_observation()
                            agent.update_belief_state()
                            recorder.record_trial_start(this_trial + 1, task)
                            break

                    recorder.append_this_round_to_block_df(this_round)
                recorder.append_this_block_to_simdata_df(this_block)
                time_printer.end_block()
            recorder.save_data_to_tsv(out_filename, agent_model)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total time for simulation: {round((end-start), ndigits=2)} sec.")
