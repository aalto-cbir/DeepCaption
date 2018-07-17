#!/bin/bash

PYTHON2=python2
COCOEVAL=eval_coco.py

if [ -z "$*" ]; then
    echo "Usage: $0 results1.json results2.json ..."
    exit 1
fi

for RESULTS_JSON in "$@"
do
    BASENAME=$(basename $RESULTS_JSON .json)
    OUTPUT_FNAME="${BASENAME}.result"
    echo -n "${BASENAME} "

    $PYTHON2 $COCOEVAL $RESULTS_JSON &> $OUTPUT_FNAME

    METEOR=$(grep "METEOR:" $OUTPUT_FNAME | tail -n1)
    CIDER=$(grep "CIDEr:" $OUTPUT_FNAME | tail -n1)
    echo "${METEOR} ${CIDER}"
done
