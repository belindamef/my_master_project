"""
This script evaluates and visualizes model recovery simulations for.

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

# Create simulation folder (if not existing) and output filenames
val_sim_data_dir = os.path.join(raw_sim_data_dir, "validation_sims")
if not os.path.exists(raw_sim_data_dir):
    os.makedirs(val_sim_data_dir)

# Specify experimental parameter
exp_params = {"n_blocks" : 3, "n_rounds" : 10 , "n_trials" : 12,
                  "dim" : 5, "n_hides" : 6
              }
# Define model space
#agent_models = ["C1", "C2", "C3", "A1", "A2", "A3"]
agent_models = ["A3"]

config_files_dir = os.path.join(working_dir, "task_config",
    f"b-{exp_params['n_blocks']}_r-{exp_params['n_rounds']}_-{['n_trials']}",
                                "main")

# Create or load if existent components
model_comp = ModelComps(working_dir=working_dir, dim=exp_params["dim"],
                        n_hides=exp_params["n_hides"])

# Create task config (loads config file if existing for given params)
task_configurator = TaskConfigurator(
    task_config_dir=config_files_dir,
    n_blocks=exp_params['n_blocks'],
    n_rounds=exp_params['n_rounds'],
    dim=exp_params['dim'],
    n_hides=exp_params['n_hides'],
)
task_configs = task_configurator.return_task_configuration()

# Start simulation
start_agent_task_interaction(
    working_dir=working_dir,
    output_dir=val_sim_data_dir,
    data_dir=data_dir,
    n_blocks=exp_params['n_blocks'],
    n_rounds=exp_params['n_rounds'],
    n_trials=exp_params['n_trials'],
    dim=exp_params['dim'],
    n_hides=exp_params['n_hides'],
    agent_models=agent_models,
    task_configs=task_configs,
    model_comps=model_comp,
    mode='eval_lklh'
)
