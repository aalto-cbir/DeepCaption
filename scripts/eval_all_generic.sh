#!/bin/bash

# Evaluate all epochs matching a given model file

RESULTS_FILE_PREFIX=$1 #../image_captioning_dev/results/${MODEL}
REFS_FILE=$2 # /proj/mediaind/picsom/databases/visualgenome/download/im2p/paragraphs_v1.json

echo "Evaluating all results matching results file prefix: $RESULTS_FILE_PREFIX"

cd ../caption-scorer
for RESULTS_FILE in ${RESULTS_FILE_PREFIX}*.json; do
    if [ -f $RESULTS_FILE ]; then
        echo "Processing $RESULTS_FILE"
        OUTPUT_FILE="${RESULTS_FILE%.json}.eval"
        OUTPUT_PATH=$(dirname "${OUTPUT_FILE}")
        if [ -s $OUTPUT_FILE ]; then
            echo "$OUTPUT_FILE already exists, proceeding to next file"
            continue
        fi

        python2 scorer.py --database VGIM2P \
            --references $REFS_FILE \
            --results $RESULTS_FILE \
            --eval_results_path $OUTPUT_PATH --range_to_100
    else
        echo "$RESULTS_FILE does not exist, moving on to next file"
    fi
done
