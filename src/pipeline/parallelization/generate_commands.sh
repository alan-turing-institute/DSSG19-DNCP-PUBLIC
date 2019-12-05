#!/bin/bash

# Delete script file
rm src/pipeline/parallelization/list_of_commands.sh 

macroyaml=$1

# Generate one line to be executed per yaml
for d in experiments/parallelization/${macroyaml/'.yaml'/''}/*; 
do
    file="$(basename "$d")"
    echo python src/pipeline/pipeline.py run_experiment --experiment_file=parallelization/${macroyaml/'.yaml'/''}/$file >> src/pipeline/parallelization/list_of_commands.sh 
done