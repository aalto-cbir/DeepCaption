#!/bin/bash

COCOEVAL=~/src/coco-caption/cocoEval.py
RESULTS_JSON=$1

if [ -z "$RESULTS_JSON" ]; then
    echo "Usage: $0 results.json"
    exit 1
fi

OUTPUT_FNAME="$(basename $RESULTS_JSON .json).result"

$COCOEVAL $RESULTS_JSON | tee $OUTPUT_FNAME
echo "Wrote $OUTPUT_FNAME"
