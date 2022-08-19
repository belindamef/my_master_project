"""
This script starts the simulation of agent-task-interactions for given
experimental parameter and agent models.

Task configurations will be either loaded from existing configuration files
or sampled at random if not existing for given experimental parameters.
This script calls the method ./utilities/agent_task_interaction.py which
iterates over the here given set of agents and performs the interactions
for each agent and task configuration.

Author: Belinda Fleischmann
"""

import os
from utilities.agent_task_interaction import start_agent_task_interaction
from utilities.model_comp import ModelComps
from utilities.create_task_config import TaskConfigurator

# Specify directories
working_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.sep.join(working_dir.split(os.sep)[:4])  # ~/treasure-hunt
data_dir = os.path.join(project_dir, "data")  # data directory
raw_sim_data_dir = os.path.join(data_dir, "rawdata", "sim")  # simulated data

if not os.path.exists(raw_sim_data_dir):
    os.makedirs(raw_sim_data_dir)

# Get and create simulation output main data directory
sim_name = str(input("Enter simulation folder name: "))

# Create simulation folder (if not existing) and output filenames
raw_sim_data_dir = os.path.join(raw_sim_data_dir, f"{sim_name}")
if not os.path.exists(raw_sim_data_dir):
    os.makedirs(raw_sim_data_dir)

# Specify experimental parameter
n_blocks = int(input("Enter number of blocks: "))  # No. of task blocks
n_rounds = 10  # No. of hunting rounds per task block
n_trials = 12
dim = 5  # dimension: No. of rows and columns of gridworld
n_hides = 6  # No. of  hiding spots in gridworld
n_nodes = dim**2  # No. of fields in the gridworld

# Define model space
# models = ['A1']
agent_models = ["C1", "C2", "C3", "A1", "A2", "A3"]


config_files_dir = os.path.join(
    working_dir,
    "task_config",
    f"b-{n_blocks}_r-{n_rounds}_" f"t-{n_trials}",
    f"{sim_name}",
)

# Create or load if existent components
model_comp = ModelComps(working_dir=working_dir, dim=dim, n_hides=n_hides)

# Create task config (loads config file if existing for given params)
task_configurator = TaskConfigurator(
    task_config_dir=config_files_dir,
    n_blocks=n_blocks,
    n_rounds=n_rounds,
    dim=dim,
    n_hides=n_hides,
)
task_configs = task_configurator.return_task_configuration()

# Start simulation
start_agent_task_interaction(
    working_dir=working_dir,
    output_dir=raw_sim_data_dir,
    n_blocks=n_blocks,
    n_rounds=n_rounds,
    n_trials=n_trials,
    dim=dim,
    n_hides=n_hides,
    agent_models=agent_models,
    task_configs=task_configs,
    model_comps=model_comp,
)
