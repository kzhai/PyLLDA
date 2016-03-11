#!/bin/bash

PYANN_HOME=$HOME/Workspace/PyANN

if [ $# == 1 ]; then
    INPUT_DIRECTORY=$1
    ITERATIONS=1000
    SNAPSHOT=100
elif [ $# == 2 ]; then
    INPUT_DIRECTORY=$1
    ITERATIONS=$2
    SNAPSHOT=$2
elif [ $# == 3 ]; then
    INPUT_DIRECTORY=$1
    ITERATIONS=$2
    SNAPSHOT=$3
else
    echo "usage: train_and_test.sh INPUT_DIRECTORY [ITERATIONS=1000] [SNAPSHOT=100]"
    exit
fi

OUTPUT_DIRECTORY=$HOME/temp.$RANDOM

python -um PyLLDA.launch_train --input_directory=$INPUT_DIRECTORY --output_directory=$OUTPUT_DIRECTORY --training_iterations=$ITERATIONS --snapshot_interval=$SNAPSHOT

for CORPUS_DIRECTORY in $OUTPUT_DIRECTORY/*
do
    if [ -f "$CORPUS_DIRECTORY" ]; then
		continue
    fi
    
    for MODEL_DIRECTORY in $CORPUS_DIRECTORY/*
	do
		if [ -f "$MODEL_DIRECTORY" ]; then
			continue
    	fi
    	
    	python -um PyLLDA.launch_test --model_directory=$MODEL_DIRECTORY --input_directory=$INPUT_DIRECTORY --snapshot_index=$ITERATIONS
    done
done

rm -r $OUTPUT_DIRECTORY