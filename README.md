# My Master Project
---
*NOTE:*  All contents in this repo are still work in progress and incomplete

---
## Table of contents
* [General Info](#general-info)
* [Spatial Search Task](#spatial-search-task-treasure-hunt-game)
* [Task Model](#task-model)
* [Behavioural Models](#behavioural-models) 
* [Simulations](#simulations)
* [Data-Analyses](#data-analyses)
* [References](#references)



## General Info 
This repository contains all scripts and tools for my master project.

In my master project I am investigating human sequential decision making in an exploration-exploitation dilemma.

For this I designed a [spatial search task](#spatial_search_task)) and developed a set of probabilistic agent-based [behavioural models](#behavioural_models) to compare different decision making strategies that participants may use for the task. This set includes explorative and exploitative Bayesian agents as well as control agents.

All scripts and tools I used for [simulations](#simulations) of agent-task-interactions and [data analyses](#data_analyses) are stored in this repository.




## Spatial Search Task (Treasure Hunt Game)
#### Scripts
* Script to run the rask: [run_task.py](https://github.com/belindamef/my_master_project/blob/main/code/run_task.py)
* task model: [task.py](https://github.com/belindamef/my_master_project/blob/main/code/utilities/task.py)

In this spatial search task participants had to make sequential decisions under uncertainty. Available actions were associated with varying expected reward values and varying information gains. 

The **task goal** was to find as many treasures as possible in a 5x5 grid-world, which was represented from a bird's eye perspective. One game consisted of 10 rounds. In each round, the participant had to find a treasure within 15 moves. One round ended if either the participant succeeded or the move limit was reached after 15 trials. Consequently, they could find between 0 and 10 treasures in one game. 
Importantly, the treasure could only be hidding on one of 6 "**hiding spots**", which, at the beginning of the game, were invisible for the participant. However, they had the option to unveil hiding spots by trading one move for a "drill". Unveiled hiding spots remained visible throughout all rounds of one game.


#### Trial layout
![alt text](https://github.com/belindamef/my_master_project/blob/main/figures/trial_layout.png)

On each trial participants were first presented with their current position (state s<sup>1</sup><sub>t</sub>)  and the prompt to choose an action. If participants chose the un-informative action a<sup>t</sup> to take a step, they were presented with their new position (state s<sup>1</sup><sub>t+1</sub>), and received the information whether they found the treasure or not. If participants chose the informative action to drill, they were first presented with an animated stimulus representing'drilling' and received the information whether they detected a hiding spot or not. (A) Scenario in which the participant decided to take a step in three consecutive trials. (B) Scenario in which the participant alternated between drills (trials 1 and 3) and steps (trial 2). Note that unveiled hiding spots remain visible over all rounds of one game as shown in the very right images.

## Task Model
The model of the task is formulated using concepts from the theory of partially observable Markov decision processes (PoMDP) [[1]](#references)

## Behavioural Models
#### Scripts
* Agent models: [agent.py](https://github.com/belindamef/my_master_project/blob/main/code/utilities/agent.py)
* Model components (prior and likelihood arrays): [model_comp.py](https://github.com/belindamef/my_master_project/blob/main/code/utilities/model_comp.py)

## Simulations
#### Scripts
* Script to run simulation with fixed or random task configurations: [run_simulation.py](https://github.com/belindamef/my_master_project/blob/main/code/run_simulation.py)
* method to simulate agent-task interaction: [agent_task_interaction.py](https://github.com/belindamef/my_master_project/blob/main/code/utilities/agent_task_interaction.py)

## Data Analyses

## References
[1] Bertsekas, D. P. (2005)._Dynamic Programming and Optimal Control_.Belmont, Mass: AthenaScientific, 3rd edition

## Contact
fl.belinda@gmail.com

