#!/bin/bash

if [ -z "$*" ]; then
    echo "Usage: $0 model1.ckpt model2.ckpt ... --common_param1 val1 ... --common_paramN valN"
    exit 1
fi

COMMON_PARAMS_OFFSET=1

# Find out where the common parameters begin (first param starting 
# starting with "--":)
for _PARAM in "$@"
do
    if [[ $_PARAM =~ ^-- ]]; then
        break
    else
        COMMON_PARAMS_OFFSET=${COMMON_PARAMS_OFFSET}+1
    fi
done

MODELS=${@:1:${COMMON_PARAMS_OFFSET}-1} # TODO: check indices
COMMON_PARAMS=${@:$COMMON_PARAMS_OFFSET}

# Run infer.py for every model:
for MODEL in $MODELS; do
    MODEL_PARAM="--model $MODEL"
    ALL_PARAMS="$MODEL_PARAM $COMMON_PARAMS"
    (set -x; sbatch --time=0-8 submit.sh infer.py ${ALL_PARAMS})
    echo $ALL_PARAMS
    sleep 1
done
