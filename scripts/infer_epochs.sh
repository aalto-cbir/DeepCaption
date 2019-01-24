#!/bin/bash

MODEL_PATH=$1
DATASET=$2
EP_BEGIN=$3
EP_END=$4
# The line below allows user to supply arbitrary many arguments starting with $5
# that can be supplied directly to the ./infer.py script:
MISC_PARAMS=${@:5}

echo "Evaluating model: $MODEL_PATH"
echo "Epoch start: $EP_BEGIN"
echo "Epoch end: $EP_END"

for EPOCH in $(eval echo {$EP_BEGIN..$EP_END}); do
    MODEL_FILE="$MODEL_PATH/ep${EPOCH}.model"
    MODEL_NAME=$(basename $MODEL_PATH)
    
    RESULTS_FILE="results/${MODEL_NAME}-ep${EPOCH}.json"
    
    if [ -s $RESULTS_FILE ]; then
        echo "File $RESULTS_FILE already exists. Please remove it if you want to infer this model again"
        continue
    fi

    if [ -s $MODEL_FILE ]; then
        echo "Inferring from model: $MODEL_FILE using dataset: $DATASET..."
        python3 infer.py --model $MODEL_FILE --dataset $DATASET \
                         --output_format json --num_workers 4 $MISC_PARAMS
    else
        echo "File $MODEL_FILE doesn't exist, skipping."
    fi
done
