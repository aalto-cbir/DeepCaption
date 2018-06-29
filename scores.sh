#!/bin/bash

PYTHON2=python2

if [[ $(hostname -s) == taito* ]]; then
    PYTHON2=/appl/opt/python/2.7.10-gcc493-shared/bin/python2
    COCOEVAL=~/appl_taito/coco-caption/cocoEval.py
elif [[ $(hostname -s) == cs-299 ]]; then 
    COCOEVAL=~/src/coco-caption/cocoEval.py
else
    echo "You need to specify the location of cocoEval.py in the script..."
    exit 1
fi

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
