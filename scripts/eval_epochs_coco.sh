#!/bin/bash

MODEL_NAME=$1
REFS_FILE=$2
EP_BEGIN=$3
EP_END=$4

echo "Evaluating model: $MODEL_NAME"
echo "Epoch start: $EP_BEGIN"
echo "Epoch end: $EP_END"

for EPOCH in $(eval echo {$EP_BEGIN..$EP_END}); do
    RESULTS_FILE="results/${MODEL_NAME}-ep${EPOCH}.json"
    if [ -f $RESULTS_FILE ]; then
        echo "Processing $RESULTS_FILE"
        OUTPUT_FILE="${RESULTS_FILE%.json}.eval"
        if [ -f $OUTPUT_FILE ]; then
            echo "$OUTPUT_FILE already exists, proceeding to next file"
            continue
        fi
        python2.7 eval_coco.py $RESULTS_FILE --ground_truth $REFS_FILE
    else
        echo "$RESULTS_FILE does not exist, moving on to next file"
    fi
done
