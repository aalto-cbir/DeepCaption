#!/bin/bash

MODEL_PATH=$1
DATASET=$2
EP_BEGIN=$3
EP_END=$4

echo "Evaluating model: $MODEL_PATH"
echo "Epoch start: $EP_BEGIN"
echo "Epoch end: $EP_END"

for EPOCH in $(eval echo {$EP_BEGIN..$EP_END}); do
    MODEL_FILE="$MODEL_PATH/ep${EPOCH}.model"
    if [ -f $MODEL_FILE ]; then
        echo "Inferring from model: $MODEL_FILE using dataset: $DATASET..."
        python3 infer.py --model $MODEL_FILE --dataset $DATASET \
                         --output_format json --num_workers 4
    else
        echo "File $MODEL_FILE doesn't exist, skipping."
    fi
done
