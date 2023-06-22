#!/bin/bash
ARGUMENTS="$@"

# Anything that should be done before the script call should be added here

# Load environment
source /home/data/treasure_hunt/.th_env/bin/activate

# Run script
python3 /home/data/treasure_hunt/treasure-hunt/code/run_validation_testing.py $ARGUMENTS
