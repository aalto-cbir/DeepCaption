#!/bin/bash

PYTHON2=/appl/opt/python/2.7.10-gcc493-shared/bin/python2
COCOEVAL=~/appl_taito/coco-caption/cocoEval.py
RESULTS_JSON=$1

if [ -z "$RESULTS_JSON" ]; then
    echo "Usage: $0 results.json"
    exit 1
fi

OUTPUT_FNAME="$(basename $RESULTS_JSON .json).result"

$PYTHON2 $COCOEVAL $RESULTS_JSON | tee $OUTPUT_FNAME
echo "Wrote $OUTPUT_FNAME"
