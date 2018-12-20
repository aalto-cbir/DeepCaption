#!/bin/bash
#SBATCH --mem=8GB
#SBATCH --time=0-1
#SBATCH --mail-user=arturs.polis@aalto.fi
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH -o "eval-%j.out"
#SBATCH -n 4

# Set billing project
# newgrp mvsjober

# research-support@csc.fi suggested for pytorch 0.4.0
module purge
module load python-env/2.7.10

EPOCH=$1 # 11
MODELS=$2 # model1 model2 model3
RESULTS_TEMPLATE=$3 # ../image_captioning_dev/results/${MODEL}-ep${EPOCH}.json
REFERENCE_CAPTIONS=$4 # /proj/mediaind/picsom/databases/visualgenome/download/im2p/paragraphs_v1.json
DATABASE=$5 #VGIM2P
OUTPUT_PATH=$6 # ../image_captioning_dev/results/

cd ../caption-scorer
for NAME in $MODELS; do
    RESULTS_FILE=$(eval echo $RESULTS_TEMPLATE)
    if [ -f $RESULTS_FILE ]; then
        echo "Processing $RESULTS_FILE"
        OUTPUT_FILE="${RESULTS_FILE%.json}.eval"
        if [ -f $OUTPUT_FILE ]; then
            echo "$OUTPUT_FILE already exists, proceeding to next file"
            continue
        fi

        srun python2 scorer.py --database VGIM2P \
            --references $REFERENCE_CAPTIONS \
            --results $RESULTS_FILE \
            --eval_results_path $OUTPUT_PATH --range_to_100
    else
        echo "$RESULTS_FILE does not exist, moving on to next file"
    fi
done
wait
