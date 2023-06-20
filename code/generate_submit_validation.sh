#!/bin/bash
# v3.0

timestamp=$(date +"%Y%m%d_%H%M")
logs_dir=/home/data/treasure_hunt/logs/tests/model_recov_${timestamp}
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

# print the .submit header
printf "# The environment
universe       = vanilla
getenv         = True
request_cpus   = 1
request_memory = 8G

# Execution
initial_dir    = /home/data/treasure_hunt/treasure-hunt/code
executable     = /home/data/treasure_hunt/treasure-hunt/code/wrapper_script_validation.sh\n"

# Make sure floating numbers are comma seperated
export LC_NUMERIC="en_US.UTF-8"

# Define simulation parameter spaces
declare -a agent_models=(C2 C3 A1 A2 A3)
#declare -a agent_models=(A2 A3)
n_repetitions=1
tau_resolution_step=0.4
n_participants=5

# Iterate over repetitions
for repetition in $(seq 1 ${n_repetition}); do

    # Iterate over data generating agent models
    for agent_model in ${agent_models[@]}; do

        # Iterate over data generating tau parameter values
        for tau_value in $(seq .1 ${tau_resolution_step} 2.0); do

            # Define lambda space
            if [ ${agent_model} = A3 ]; then
                lambda_gen_space=$(seq 0.1 0.2 0.9)
            else
                lambda_gen_space=(0.5)
            fi

            # Iterate over data generating lambda parameter values
            for lambda_value in ${lambda_gen_space}; do

                # Iterate over participants
                for participant in $(seq 0 ${n_participants}); do

                    # Create a job for each subject file
                    printf "arguments = --parallel_computing --repetition ${repetition} --agent_model ${agent_model} --tau_value ${tau_value} --lambda_value ${lambda_value} --participant ${participant}\n"
                    printf "log       = ${logs_dir}/\$(Cluster).\$(Process)_rep-${repetition}_sub-${agent_model}_tau-${tau_value}_lambda-${lambda_value}_p-${participant}.log\n"
                    printf "output    = ${logs_dir}/\$(Cluster).\$(Process)_rep-${repetition}_sub-${agent_model}_tau-${tau_value}_lambda-${lambda_value}_p-${participant}.out\n"
                    printf "error     = ${logs_dir}/\$(Cluster).\$(Process)_rep-${repetition}_sub-${agent_model}_tau-${tau_value}_lambda-${lambda_value}_p-${participant}.err\n"
                    printf "Queue\n\n"
                done
            done
        done
    done
done
