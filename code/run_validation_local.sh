#!/bin/bash
# v3.0

timestamp=$(date +"%Y%m%d_%H%M")

# Make sure floating numbers are comma seperated
export LC_NUMERIC="en_US.UTF-8"

# Define simulation behavioral model space
declare -a all_agents=(C1 C2 C3 A1) # A2 A3)
declare -a bayesian_agents=(A1) # A2 A3)
declare -a control_agents=(C1 C2 C3)
#declare -a agent_models=(A3)

# Specify number of repetions and participants
n_repetitions=1
n_participants=1

# Parameterspace
tau_max=0.5
tau_resolution_step=0.1
lambda_resolution_step=0.2

# Iterate over repetitions
for repetition in $(seq 1 ${n_repetition}); do

    # Iterate over data generating agent models
    for agent_model in ${all_agents[@]}; do

        # Define tau space # TODO
        if [[ " ${bayesian_agents[*]} " == *" ${agent_model} "* ]]; then
            tau_gen_sapce=$(seq 0 ${tau_resolution_step} ${tau_max})
        else
            tau_gen_sapce=(nan)
        fi

        # Iterate over data generating tau parameter values
        for tau_value in ${tau_gen_sapce}; do

            # Define lambda space
            if [ ${agent_model} = A3 ]; then
                lambda_gen_space=$(seq 0.0 ${lambda_resolution_step} 1.0)
            else
                lambda_gen_space=(nan)
            fi

            # Iterate over data generating lambda parameter values
            for lambda_value in ${lambda_gen_space}; do

                # Iterate over participants
                for participant in $(seq 1 ${n_participants}); do

                    # Run validation
                    cd ~/treasure-hunt
                    python code/run_validation.py --parallel_computing --repetition ${repetition} --agent_model ${agent_model} --tau_value ${tau_value} --lambda_value ${lambda_value} --participant ${participant}
                done
            done
        done
    done
done
