#!/bin/bash

MODEL_PATH=$1
DATASET=$2
EP_BEGIN=$3
EP_END=$4

echo "Evaluating model: $MODEL_PATH"
echo "Epoch start: $EP_BEGIN"
echo "Epoch end: $EP_END"

for EPOCH in $(eval echo {$EP_BEGIN..$EP_END}); do
    python3 infer.py --model $MODEL_PATH/$EPOCH.model --dataset $DATASET \
                     --data_format json --num_workers 4
done
