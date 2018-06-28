#!/bin/bash

if [ -z "$*" ]; then
    echo "Usage: $0 model1.ckpt model2.ckpt ..."
    exit 1
fi

for MODEL in "$@"
do
    (set -x; sbatch --time=0-5 submit.sh eval.py --model ${MODEL})
    sleep 1
done
