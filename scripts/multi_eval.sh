#!/bin/bash

PYTHON2=python2
EVAL_SCRIPT=''

if [ -z "$*" ]; then
    echo "Usage: $0 DATASET_NAME results1.json results2.json ..."
    exit 1
fi

if [[ $1 == 'coco2014' ]]; then
    EVAL_SCRIPT=eval_coco.py
else
    exit 1
fi

# $HOSTNAME works inside CSC compute nodes:
if [[ $HOSTNAME == taito* ]]; then 
    echo "Loading Python 2 environment:"
    module purge
    module load python-env/2.7.10
elif [[ $(hostname -s) == cs-299 ]]; then 
    COCOEVAL=~/src/coco-caption/cocoEval.py
else
    echo "You need to specify the location of cocoEval.py in the script..."
    exit 1
fi

for RESULTS_JSON in "${@:2}"
do
    echo "Processing $RESULTS_JSON"

    $PYTHON2 $EVAL_SCRIPT $RESULTS_JSON
done
