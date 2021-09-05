from utilities.abm_structure import AbmStructure
from utilities.agent_task_interaction import simulation
from utilities.model_comp import ModelComp
from utilities.create_task_config import TaskConfigurator

import os
import numpy as np

"""
This script evaluates and visualizes face validity simulations
"""

# Specify directories
working_dir = os.getcwd()  # working dir
project_dir = os.sep.join(working_dir.split(os.sep)[:4])  # Should be Users/<{$USER}>/<{$PROJECT_FOLDER}>
data_dir = os.path.join(project_dir, 'data')  # data directory
sim_data_dir = os.path.join(data_dir, 'sim')  # directory for generated data
results_dir = os.path.join(project_dir, 'results')  # results directory

if not os.path.exists(sim_data_dir):
    os.makedirs(sim_data_dir)

# Get and create simulation output main data directory
sim_name = str(input("Enter simulation folder name: "))

# Create simulation folder (if not existing) and output filenames
sim_data_dir = os.path.join(sim_data_dir, f'{sim_name}')
if not os.path.exists(sim_data_dir):
    os.makedirs(sim_data_dir)

# Specify experimental parameter
blocks = 3  # No. of task blocks (each block has different tr location, but same hiding spots
rounds = 10  # No. of hunting rounds per task block
dim = 5  # dimension: No. of rows and columns of gridworld
trials = 12
n_hides = 6  # No. of  hiding spots in gridworld
n_nodes = dim ** 2  # No. of fields in the gridworld

# Specify simulation parameter
n_task_configs = 100  # number of task configurations, if sampling tasks
#task_config_type = 'sample'
task_config_type = 'create_n_save'

# Initialize components stimulus creation
gridsize = 10  # Size of gridworld in cm
cube_size = gridsize / dim  # size of one cube in the gridworld

# Define model space
#models = ['A1']
models = ['C1', 'C2', 'C3', 'A1', 'A2', 'A3']  # generating and analysis model space

sim = AbmStructure()  # initialize simulation object
sim.sim_data_dir = sim_data_dir
sim.blocks = blocks  # number of blocks
sim.rounds = rounds  # number of rounds
sim.trials = trials  # number of trials

sim.working_dir = working_dir
sim.mode = 'simulation'

# Paradigm parameters
sim.dim = dim  # dimensionality (No. of rows and columns of the gridworld
sim.gridsize = gridsize  # size of the gridworld given in cm
sim.n_hides = n_hides
sim.n_nodes = n_nodes

# Prepare task configuration object
config_params = AbmStructure()
config_params.task_config_type = task_config_type
if config_params.task_config_type == 'create_n_save':
    config_params.blocks = blocks
elif config_params.task_config_type == 'sample':
    config_params.blocks = n_task_configs
    sim.blocks = n_task_configs
config_params.config_file_path = os.path.join(working_dir,
                                              'task_config',
                                              f'b-{blocks}_r-{rounds}_t-{trials}',
                                              f'{sim_name}')
config_params.rounds = rounds
config_params.n_nodes = n_nodes
config_params.n_hides = n_hides


# Create or load if existent components
sim.model_comp = ModelComp(sim)

sim.models = models

# Create or fetch task config
sim.task_configuration = TaskConfigurator(config_params)

# Start simulation
sim = simulation(sim)  # data simulation
