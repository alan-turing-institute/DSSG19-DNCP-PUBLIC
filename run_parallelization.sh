#/ bin/bash

# Run it from root!!!!

macroexperiment=$1

# Split
python src/pipeline/parallelization/parallelization.py initialise_parallel_experiments --experiment_file=$macroexperiment

# Generate commands
source src/pipeline/parallelization/generate_commands.sh $macroexperiment

# Run parallel
parallel --tmux < src/pipeline/parallelization/list_of_commands.sh

# Delete script file
rm src/pipeline/parallelization/list_of_commands.sh 

# Remove tmp folder with experiment splits
python src/pipeline/parallelization/parallelization.py remove_experiment_splits --experiment_file=$macroexperiment
