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
request_memory = 16G
notify_user    = belinda.fleischmann@ovgu.de
notification   = Error

# Execution
initial_dir    = /home/data/treasure_hunt/treasure-hunt/code
executable     = /home/data/treasure_hunt/treasure-hunt/code/wrapper_script_validation.sh\n"

# Make sure floating numbers are comma seperated
export LC_NUMERIC="en_US.UTF-8"

# Define data generating model space
declare -a all_agents_gen=(C1 C2 C3 A1 A2 A3)
declare -a bayesian_agents_gen=(A1 A2 A3)
declare -a control_agents_gen=(C1 C2 C3)
#declare -a agent_models=(A3)

# Define analzing candidate model space
declare -a all_agents_cand=(C1 C2 C3 A1 A2 A3)
declare -a bayesian_agents_cand=(A1 A2 A3)
declare -a control_agents_cand=(C1 C2 C3)

# Specify number of repetions and participants
n_repetitions=1
n_participants=1

# Parameterspace
tau_min=0.01
tau_max=0.5
tau_resolution_step=0.05
lambda_min=0.25
lambda_max=0.75
lambda_resolution_step=0.25

# Iterate over repetitions
for repetition in $(seq 1 ${n_repetition}); do

    # Iterate over data generating agent models
    for agent_model in ${all_agents_gen[@]}; do

        # Define tau space # TODO
        if [[ " ${bayesian_agents_gen[*]} " == *" ${agent_model} "* ]]; then
            tau_gen_space=$(seq ${tau_min} ${tau_resolution_step} ${tau_max})
        else
            tau_gen_space=(nan)
        fi

        # Iterate over data generating tau parameter values
        for tau_value in ${tau_gen_space}; do

            # Define lambda space
            if [ ${agent_model} = A3 ]; then
                lambda_gen_space=$(seq ${lambda_min} ${lambda_resolution_step} ${lambda_max})
            else
                lambda_gen_space=(nan)
            fi

            # Iterate over data generating lambda parameter values
            for lambda_value in ${lambda_gen_space}; do

                # Iterate over participants
                for participant in $(seq 1 ${n_participants}); do

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
