#!/bin/bash
# v3.0

timestamp=$(date +"%Y%m%d_%H%M")
logs_dir=/home/data/treasure_hunt/logs/tests/model_fit_${timestamp}
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
executable     = /home/data/treasure_hunt/treasure-hunt/code/wrapper_script_model_fit.sh\n"

# Make sure floating numbers are comma seperated
export LC_NUMERIC="en_US.UTF-8"

# Specify number of repetions and participants
n_repetitions=1
n_participants=50  # TODO: read from folder

# Iterate over repetitions
for repetition in $(seq 1 ${n_repetition}); do

    # Iterate over participants
    for participant in $(seq 1 ${n_participants}); do

        # Create a job for each subject file
        printf "arguments = --parallel_computing --repetition ${repetition} --participant ${participant}\n"
        printf "log       = ${logs_dir}/\$(Cluster).\$(Process)_rep-${repetition}_sub-${participant}.log\n"
        printf "output    = ${logs_dir}/\$(Cluster).\$(Process)_rep-${repetition}_sub-${participant}.out\n"
        printf "error     = ${logs_dir}/\$(Cluster).\$(Process)_rep-${repetition}_sub-${participant}.err\n"
        printf "Queue\n\n"
    done
done
